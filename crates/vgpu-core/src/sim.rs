//! Event-driven CTA scheduler with a fluid resource-sharing phase model.
//!
//! Events happen only at CTA phase boundaries (launch, prologue, steady-state
//! loop, epilogue). Between events every active phase progresses at a constant
//! rate, recomputed from the current set of active phases:
//!
//! - tensor throughput and non-MMA issue are shared among the CTAs of one SM;
//! - within a CTA, an iteration's non-MMA work follows its MMAs in series;
//! - L2 and DRAM bandwidth are shared max-min fairly among all phases on the
//!   GPU, with the achievable bandwidth taken from the profile curve for the
//!   current number of active SMs.
//!
//! See `docs/components/model-core.md`, sections 3.1 to 3.5.

use crate::kernel::{Access, Kernel, Spill, SpillScope, Work};
use crate::occupancy::{occupancy, Occupancy, OccupancyError};
use crate::profile::{interpolate, MlpPoint, Profile, SpillReloadPoint, WarpRate};

/// Where the bytes of an access are served from.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Traffic {
    /// Already in L2 when requested: L2 bandwidth, L2 latency.
    pub resident: f64,
    /// Merged into a DRAM fill another resident CTA has in flight: L2
    /// bandwidth only, but the request waits for the fill (DRAM latency).
    pub merged: f64,
    /// Fetched from DRAM.
    pub dram: f64,
}

impl Traffic {
    pub fn total(&self) -> f64 {
        self.resident + self.merged + self.dram
    }
}

impl std::ops::Add for Traffic {
    type Output = Traffic;
    fn add(self, o: Traffic) -> Traffic {
        Traffic {
            resident: self.resident + o.resident,
            merged: self.merged + o.merged,
            dram: self.dram + o.dram,
        }
    }
}

/// Spill traffic of all resident threads of one SM per execution of the spill
/// scope, by the level that serves it, and the L1 miss rate of the frames.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SpillSplit {
    pub l1: f64,
    pub l2: f64,
    pub dram: f64,
    pub miss: f64,
}

/// The core's view of the L2 model.
pub trait L2 {
    fn begin_tile(&mut self, _sm: u32, _program: u32, _tile: u32, _t_ns: f64) {}
    fn end_tile(&mut self, _sm: u32, _program: u32, _tile: u32, _t_ns: f64) {}
    /// Traffic of one access over steps `[k_begin, k_end)` of a phase.
    #[allow(clippy::too_many_arguments)]
    fn split(
        &mut self,
        sm: u32,
        program: u32,
        tile: u32,
        access: &Access,
        k_begin: u32,
        k_end: u32,
        is_write: bool,
        t_ns: f64,
    ) -> Traffic;

    /// Spill traffic of all resident threads of one SM per execution of the
    /// spill scope, at the SM's shared memory carveout.
    fn spill(
        &mut self,
        _sm: u32,
        active_threads_in_sm: u32,
        _carveout_bytes: u64,
        s: &Spill,
    ) -> SpillSplit {
        SpillSplit {
            l1: s.bytes_per_thread * f64::from(active_threads_in_sm),
            ..SpillSplit::default()
        }
    }
}

/// Fixed resident hit ratio for loads, nominal bytes, stores go to DRAM. For
/// tests.
pub struct ConstantHitRatio(pub f64);

impl L2 for ConstantHitRatio {
    fn split(
        &mut self,
        _: u32,
        _: u32,
        _: u32,
        a: &Access,
        k0: u32,
        k1: u32,
        is_write: bool,
        _: f64,
    ) -> Traffic {
        let bytes = a.nominal_bytes * f64::from(k1 - k0);
        let hit = if is_write { 0.0 } else { self.0 };
        Traffic {
            resident: bytes * hit,
            merged: 0.0,
            dram: bytes * (1.0 - hit),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    Occupancy(OccupancyError),
    MissingMma(String),
    EmptyCurve(&'static str),
    Unsupported(String),
}

/// Profile values at one draw, in ns, bytes and FLOP units.
#[derive(Debug, Clone)]
pub struct Machine {
    pub sm_count: u32,
    /// (active SMs, bytes per ns).
    pub dram_bw: Vec<(f64, f64)>,
    pub l2_bw: Vec<(f64, f64)>,
    /// Load latency at idle and at saturation of each level.
    pub dram_latency_ns: f64,
    pub dram_latency_loaded_ns: f64,
    pub l2_latency_ns: f64,
    pub l2_latency_loaded_ns: f64,
    /// (bytes in flight, delivered bytes per ns) of each level, indexed by
    /// issue mode: `[asynchronous copies, loads into registers]`. Empty if
    /// the profile has no MLP curves.
    pub dram_mlp: [Vec<(f64, f64)>; 2],
    pub l2_mlp: [Vec<(f64, f64)>; 2],
    /// (MMA-issuing warps on the SM, FLOP per ns for the whole SM).
    pub mma: Vec<(f64, f64)>,
    /// Warp instructions per ns per SM.
    pub issue_rate: f64,
    /// (warps on the SM issuing them, warp instructions per ns for the whole
    /// SM) of the special-function and shuffle pipes.
    pub sfu: Vec<(f64, f64)>,
    pub shfl: Vec<(f64, f64)>,
    /// Global memory warp instructions per ns per SM; infinite without a
    /// measurement.
    pub lsu_rate: f64,
    /// Shared memory bytes per ns per SM.
    pub smem_bw: f64,
    /// L1-resident local memory (spill) bytes per ns per SM.
    pub local_bw: f64,
    pub launch_overhead_ns: f64,
    pub cta_launch_ns: f64,
    /// Multiplier on per-iteration spill traffic whose loop placement is assumed.
    pub assumed_spill_scale: f64,
    /// (local memory footprint per SM, (reloads, extra ns per iteration)) of
    /// batched spill reloads served by L1 and by L2.
    pub spill_reload_l1: Vec<(f64, Vec<(f64, f64)>)>,
    pub spill_reload_l2: Vec<(f64, Vec<(f64, f64)>)>,
    /// (warps, stage buffers, ns): per-stage synchronization cost.
    pub fixed_sync: Vec<(u32, u32, f64)>,
    /// (active AI Cores, ns): cross-core AIC <-> AIV handoff latency (empty on non-Ascend).
    pub cv_sync_hop_ns: Vec<(f64, f64)>,
    /// True on architectures with physically separate Cube and Vector units (Ascend).
    pub cube_vector_split: bool,
    /// Effective L2 capacity in bytes (used to check inter-core workspace L2 residency).
    pub l2_effective_capacity_bytes: f64,
}

fn curve<T>(
    points: &[T],
    f: impl Fn(&T) -> (f64, f64),
    name: &'static str,
) -> Result<Vec<(f64, f64)>, ModelError> {
    let mut v: Vec<(f64, f64)> = points.iter().map(f).collect();
    if v.is_empty() {
        return Err(ModelError::EmptyCurve(name));
    }
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(v)
}

/// A measured (warps, rate) curve, or a flat one at `fallback`.
fn warp_curve(points: &[WarpRate], fallback: f64) -> Vec<(f64, f64)> {
    curve(
        points,
        |q| (q.warps_per_sm.into(), q.rate.value),
        "warp_rate",
    )
    .unwrap_or_else(|_| vec![(1.0, fallback)])
}

/// Bytes of one spill reload: the reload microbenchmark uses 4 B loads, and
/// the assembler's static report gives bytes, not instructions.
pub const SPILL_RELOAD_BYTES: f64 = 4.0;

/// Batched reload points of one regime, by footprint per SM, each sorted by
/// reload count.
fn reload_table(points: &[SpillReloadPoint], regime: &str) -> Vec<(f64, Vec<(f64, f64)>)> {
    let mut table: Vec<(f64, Vec<(f64, f64)>)> = Vec::new();
    for q in points
        .iter()
        .filter(|q| q.regime == regime && q.dependency_mode == "batch_before_dep")
    {
        let (Some(frame), Some(threads)) = (q.frame_bytes_per_thread, q.resident_threads_per_sm)
        else {
            continue;
        };
        let footprint = (frame * threads) as f64;
        let point = (f64::from(q.reload_count), q.extra_ns_vs_zero.value.max(0.0));
        match table.iter_mut().find(|e| e.0 == footprint) {
            Some(e) => e.1.push(point),
            None => table.push((footprint, vec![point])),
        }
    }
    for e in &mut table {
        e.1.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    table.sort_by(|a, b| a.0.total_cmp(&b.0));
    table
}

/// Extra ns for `count` reloads at `footprint`: interpolated over the
/// count, beyond the largest count at its average cost per reload, then
/// over the footprint, clamped at both ends.
fn reload_table_ns(table: &[(f64, Vec<(f64, f64)>)], count: f64, footprint: f64) -> f64 {
    let at: Vec<(f64, f64)> = table
        .iter()
        .map(|(f, c)| {
            let (n, ns) = c[c.len() - 1];
            let v = if count > n && n > 0.0 {
                ns * count / n
            } else {
                interpolate(c, count)
            };
            (*f, v)
        })
        .collect();
    interpolate(&at, footprint)
}

/// Throughput of each of `u` CTAs using a pipe on one SM, at index `u`.
#[derive(Debug, Clone, Copy)]
struct PerCta {
    mma: f64,
    sfu: f64,
    shfl: f64,
}

/// MLP curves per issue mode, `[cp_async, ld_global_v4]`. A missing mode
/// borrows the other's curve.
fn mlp_curves(points: &[MlpPoint]) -> [Vec<(f64, f64)>; 2] {
    let pick = |mode: &str| {
        let mut v: Vec<(f64, f64)> = points
            .iter()
            .filter(|q| q.issue_mode == mode)
            .map(|q| (q.bytes_in_flight as f64, q.bw_gbps.value))
            .collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    };
    let (a, b) = (pick("cp_async"), pick("ld_global_v4"));
    match (a.is_empty(), b.is_empty()) {
        (false, true) => [a.clone(), a],
        (true, false) => [b.clone(), b],
        _ => [a, b],
    }
}

impl Machine {
    /// Little's law latencies with the given bytes in flight at each level:
    /// bytes in flight over the bandwidth the MLP curve delivers for them.
    /// Below the first measured point the latency is that point's.
    fn mlp_latency(&self, mode: usize, dram_in_flight: f64, l2_in_flight: f64) -> Latency {
        let little = |curve: &[(f64, f64)], f: f64| {
            let (f0, x0) = curve[0];
            if f <= f0 {
                f0 / x0
            } else {
                f / interpolate(curve, f)
            }
        };
        Latency {
            dram: little(&self.dram_mlp[mode], dram_in_flight),
            l2: little(&self.l2_mlp[mode], l2_in_flight),
        }
    }

    /// Latencies when DRAM and L2 deliver the given fractions of their peak
    /// bandwidth: linear between the idle and the saturated measurement.
    fn latency(&self, dram_util: f64, l2_util: f64) -> Latency {
        let lerp = |a: f64, b: f64, u: f64| a + (b - a) * u.clamp(0.0, 1.0);
        Latency {
            dram: lerp(self.dram_latency_ns, self.dram_latency_loaded_ns, dram_util),
            l2: lerp(self.l2_latency_ns, self.l2_latency_loaded_ns, l2_util),
        }
    }

    /// A round trip takes at least the idle latency plus streaming its own
    /// bytes through the SM's port, and at least the time the port needs for
    /// the bytes of all the SM's CTAs, whose round trips run out of phase.
    /// The MLP curves are indexed by bytes in flight over the whole GPU and
    /// do not see one SM receiving them all.
    fn port_floor(&self, lat: Latency, d: &Demand, resident: u32) -> Latency {
        if d.round_trips == 0.0 {
            return lat;
        }
        let port = interpolate(&self.l2_bw, 1.0);
        let own = d.load_bytes / d.round_trips / port;
        let shared = own * f64::from(resident);
        Latency {
            dram: lat.dram.max(self.dram_latency_ns + own).max(shared),
            l2: lat.l2.max(self.l2_latency_ns + own).max(shared),
        }
    }

    /// Per-stage synchronization cost for the measured warp count and buffer
    /// count nearest the kernel's, or zero if the profile has none.
    fn sync_ns(&self, warps: u32, buffers: u32) -> f64 {
        self.fixed_sync
            .iter()
            .min_by_key(|&&(w, t, _)| (w.abs_diff(warps), t.abs_diff(buffers)))
            .map_or(0.0, |&(_, _, ns)| ns)
    }

    /// Latency that `count` batched spill reloads per thread add to a CTA's
    /// dependent chain, for frames of `footprint` bytes per SM of which the
    /// fraction `miss` is served by L2. Zero without measured reloads.
    pub fn spill_reload_ns(&self, count: f64, footprint: f64, miss: f64) -> f64 {
        if count <= 0.0 || self.spill_reload_l1.is_empty() {
            return 0.0;
        }
        let hit = reload_table_ns(&self.spill_reload_l1, count, footprint);
        if self.spill_reload_l2.is_empty() {
            return hit;
        }
        let l2 = reload_table_ns(&self.spill_reload_l2, count, footprint);
        let miss = miss.clamp(0.0, 1.0);
        (1.0 - miss) * hit + miss * l2
    }

    fn peak(curve: &[(f64, f64)]) -> f64 {
        curve[curve.len() - 1].1
    }

    /// The unperturbed profile (draw 0).
    pub fn nominal(p: &Profile, mma_key: &str) -> Result<Machine, ModelError> {
        let m = &p.measured;
        let mma_points = m
            .compute
            .mma_tflops_per_sm_by_warps
            .get(mma_key)
            .ok_or_else(|| ModelError::MissingMma(mma_key.to_string()))?;
        Ok(Machine {
            sm_count: p.specs.sm_count,
            dram_bw: curve(
                &m.memory.dram_bw_by_active_sms,
                |q| (q.active_sms.into(), q.bw_gbps.value),
                "dram_bw_by_active_sms",
            )?,
            l2_bw: curve(
                &m.memory.l2_bw_by_active_sms,
                |q| (q.active_sms.into(), q.bw_gbps.value),
                "l2_bw_by_active_sms",
            )?,
            dram_latency_ns: m.memory.dram_latency_ns.value,
            dram_latency_loaded_ns: m
                .memory
                .dram_latency_loaded_ns
                .as_ref()
                .map_or(m.memory.dram_latency_ns.value, |v| v.value),
            l2_latency_ns: m.memory.l2_latency_ns.value,
            l2_latency_loaded_ns: m
                .memory
                .l2_latency_loaded_ns
                .as_ref()
                .map_or(m.memory.l2_latency_ns.value, |v| v.value),
            dram_mlp: mlp_curves(&m.memory.dram_bw_by_bytes_in_flight),
            l2_mlp: mlp_curves(&m.memory.l2_bw_by_bytes_in_flight),
            mma: curve(
                mma_points,
                |q| (q.mma_warps_per_sm.into(), q.tflops_per_sm.value * 1e3),
                "mma_tflops_per_sm_by_warps",
            )?,
            issue_rate: m.compute.non_mma_issue_rate_per_sm_gips.value,
            // Without a measured curve, the special-function pipe gets the
            // lane rate over the SMs and the 32 lanes of a warp, and the
            // shuffle pipe the issue rate.
            sfu: warp_curve(
                &m.compute.sfu_issue_rate_per_sm_gips,
                m.compute
                    .sfu_tflops
                    .as_ref()
                    .map_or(m.compute.non_mma_issue_rate_per_sm_gips.value, |v| {
                        v.value * 1e3 / (f64::from(p.specs.sm_count) * 32.0)
                    }),
            ),
            shfl: warp_curve(
                &m.compute.shfl_issue_rate_per_sm_gips,
                m.compute.non_mma_issue_rate_per_sm_gips.value,
            ),
            // The narrowest loads measured, where issue rather than bytes
            // limits; the fastest rate the SM sustained for them.
            lsu_rate: m
                .memory
                .gmem_load_issue_controlled
                .iter()
                .min_by_key(|q| q.load_width_bytes)
                .map(|q| q.load_width_bytes)
                .map_or(f64::INFINITY, |w| {
                    m.memory
                        .gmem_load_issue_controlled
                        .iter()
                        .filter(|q| q.load_width_bytes == w)
                        .map(|q| q.issue_rate_gwarp_inst_per_sm.value)
                        .fold(0.0, f64::max)
                }),
            smem_bw: m.memory.smem_bw_per_sm_gbps.value,
            local_bw: m
                .spill
                .get("local_mem_l1_hit_bw_per_sm_gbps")
                .and_then(Option::as_ref)
                .map_or(m.memory.smem_bw_per_sm_gbps.value, |v| v.value),
            launch_overhead_ns: m.overhead.launch_overhead_us.value * 1e3,
            cta_launch_ns: m.overhead.per_cta_launch_latency_ns.value,
            assumed_spill_scale: 1.0,
            spill_reload_l1: reload_table(&m.memory.spill_reload_latency_by_count, "l1_hit"),
            spill_reload_l2: reload_table(&m.memory.spill_reload_latency_by_count, "l2_spill"),
            fixed_sync: m
                .pipeline
                .fixed_sync_ns
                .iter()
                .map(|q| (q.num_warps, q.tokens, q.overhead_ns.value))
                .collect(),
            cv_sync_hop_ns: curve(
                &m.pipeline.cv_sync_hop_ns_by_active_sms,
                |q| (q.active_sms.into(), q.latency_ns.value),
                "cv_sync_hop_ns_by_active_sms",
            )
            .unwrap_or_default(),
            cube_vector_split: p
                .specs
                .capabilities
                .get("cube_vector_split")
                .copied()
                .unwrap_or(false),
            l2_effective_capacity_bytes: m.memory.l2_effective_capacity_bytes.value,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// Launch overhead plus makespan.
    pub time_ns: f64,
    pub makespan_ns: f64,
    pub occupancy: Occupancy,
    /// Programs divided by CTA slots on the GPU.
    pub waves: f64,
    /// Busy SM time divided by SM count times makespan.
    pub tail_efficiency: f64,
    pub load_bytes: f64,
    pub l2_hit_bytes: f64,
    /// Part of `l2_hit_bytes` merged into in-flight DRAM fills.
    pub merged_bytes: f64,
    pub dram_bytes: f64,
    pub events: u64,
    /// Fraction of CTA-time during which each resource of `RESOURCES` was the
    /// bound of the CTA's current phase.
    pub bound_fraction: [f64; 13],
}

/// Resources a phase can be bound by, in `Prediction::bound_fraction` order.
pub const RESOURCES: [&str; 13] = [
    "mma",
    "issue",
    "smem",
    "local",
    "sfu",
    "shfl",
    "lsu",
    "sync",
    "l2_latency",
    "dram_latency",
    "l2_bandwidth",
    "dram_bandwidth",
    "launch",
];
const MMA: usize = 0;
const LOCAL: usize = 3;
const SYNC: usize = 7;
const L2_LATENCY: usize = 8;
const DRAM_LATENCY: usize = 9;
const L2_BANDWIDTH: usize = 10;
const DRAM_BANDWIDTH: usize = 11;
const LAUNCH: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Launch,
    Prologue,
    Main,
    Epilogue,
}

/// Total demand of one phase. Progress is a fraction from 1 down to 0.
#[derive(Debug, Clone, Default)]
struct Demand {
    flops: f64,
    /// Non-MMA warp instructions (issue slots); the special-function and
    /// shuffle ones among them also use their own pipes.
    warp_insts: f64,
    sfu_insts: f64,
    shfl_insts: f64,
    smem_bytes: f64,
    /// Global load and store warp instructions.
    mem_insts: f64,
    /// Spill bytes served by L1.
    local_bytes: f64,
    /// The part of `local_bytes` on the CTA's own chain: the stores when the
    /// reload latency is charged, since its measurement includes issuing the reloads.
    chain_local_bytes: f64,
    /// All bytes through L2 (loads and stores).
    bytes: f64,
    dram_bytes: f64,
    /// Latency that does not depend on memory load (CTA launch).
    fixed_latency_ns: f64,
    /// Load bytes of the phase.
    load_bytes: f64,
    /// Memory round trips not hidden by the pipeline.
    round_trips: f64,
    /// Fraction of the phase's compute, at the rate the CTA would get alone
    /// on its SM, that adds to the round trips instead of overlapping them.
    exposed_compute: f64,
    /// Steady-state stages of the phase (pipeline iterations), and the stage
    /// buffers each CTA cycles through them.
    stages: f64,
    tokens: u32,
    /// Spill reload latency over the phase, on the CTA's dependent chain.
    reload_ns: f64,
    /// Per-stage synchronization over the phase. It stalls the SM's pipes,
    /// so the other resident CTAs cannot fill it.
    sync_ns: f64,
    /// Fraction of loaded bytes already resident in L2: they wait L2
    /// latency. The rest, merged or fetched, waits DRAM latency.
    resident_fraction: f64,
    /// Fraction of loaded bytes fetched from DRAM: only these occupy DRAM
    /// in flight, merged requests ride on their fills.
    dram_fraction: f64,
    /// True when Cube (MMA) and Vector (non-MMA) units execute concurrently via a
    /// multi-buffered inter-core workspace ring on `cube_vector_split` hardware.
    cv_overlap: bool,
}

impl Demand {
    /// Exposed latency split into (launch, L2 round trips, DRAM round trips).
    fn latency_parts(&self, lat: Latency) -> (f64, f64, f64) {
        (
            self.fixed_latency_ns,
            self.round_trips * self.resident_fraction * lat.l2,
            self.round_trips * (1.0 - self.resident_fraction) * lat.dram,
        )
    }
}

/// Load latencies at the current memory utilization, in ns.
#[derive(Debug, Clone, Copy)]
struct Latency {
    dram: f64,
    l2: f64,
}

struct Cta {
    program: u32,
    sm: u32,
    tile: u32,
    tiles: u32,
    phase: Phase,
    remaining: f64,
    demand: Demand,
}

/// Max-min fair split of `capacity` among flows with the given caps.
///
/// Flows whose cap is below the fair share are frozen at their cap and the
/// rest is shared again; the fair share only grows, so one pass per freezing
/// round is enough, and rounds are few in practice.
fn waterfill(caps: &[f64], capacity: f64, out: &mut [f64]) {
    const OPEN: f64 = -1.0;
    out.fill(OPEN);
    let mut left = capacity;
    let mut open = caps.len();
    while open > 0 {
        let fair = left / open as f64;
        let before = open;
        for (o, &cap) in out.iter_mut().zip(caps) {
            if *o == OPEN && cap <= fair {
                *o = cap;
                left -= cap;
                open -= 1;
            }
        }
        if open == before {
            out.iter_mut()
                .filter(|o| **o == OPEN)
                .for_each(|o| *o = fair);
            return;
        }
    }
}

struct Sim<'a, L: L2> {
    k: &'a Kernel,
    m: &'a Machine,
    l2: &'a mut L,
    t: f64,
    load_bytes: f64,
    l2_hit_bytes: f64,
    merged_bytes: f64,
    dram_bytes: f64,
    /// Spill traffic of one CTA per execution of the spill scope, as
    /// `(L1 bytes, L2 bytes, DRAM bytes)`.
    spill: (f64, f64, f64),
    /// Spill reload latency of one CTA per execution of the spill scope.
    reload_ns: f64,
}

impl<'a, L: L2> Sim<'a, L> {
    /// Traffic of `accesses` over steps `[k0, k1)`, as (L2 hit bytes, DRAM bytes).
    #[allow(clippy::too_many_arguments)]
    fn traffic(
        &mut self,
        accesses: &[Access],
        k0: u32,
        k1: u32,
        is_write: bool,
        sm: u32,
        program: u32,
        tile: u32,
    ) -> Traffic {
        let mut sum = Traffic::default();
        if k1 > k0 {
            for a in accesses {
                sum = sum
                    + self
                        .l2
                        .split(sm, program, tile, a, k0, k1, is_write, self.t);
            }
        }
        sum
    }

    /// Phase demands under the KernelIR convention: the pipeline's first
    /// `prefetch_distance` loads are issued in the prologue (prefill), the
    /// steady state loads the rest, and the last `prefetch_distance`
    /// iterations only compute (drain). `prologue` holds only work outside
    /// the software pipeline.
    fn demand(&mut self, phase: Phase, sm: u32, program: u32, tile: u32) -> Demand {
        let k = self.k;
        let tl = &k.tile;
        let n = k.iterations(program, tile);
        let d = tl.prefetch_distance.min(n);
        let (compute, steps, loads, stores, round_trips) = match phase {
            Phase::Launch => {
                return Demand {
                    fixed_latency_ns: self.m.cta_launch_ns,
                    ..Demand::default()
                }
            }
            Phase::Prologue => {
                let own = self.traffic(&tl.prologue.loads, 0, 1, false, sm, program, tile);
                let prefill = self.traffic(&tl.per_iteration.loads, 0, d, false, sm, program, tile);
                let st = self.traffic(&tl.prologue.stores, 0, 1, true, sm, program, tile);
                (&tl.prologue, 1, own + prefill, st, 1.0)
            }
            Phase::Main => {
                let ld = self.traffic(&tl.per_iteration.loads, d, n, false, sm, program, tile);
                let st = self.traffic(&tl.per_iteration.stores, 0, n, true, sm, program, tile);
                // A closed loop of d + 1 stage buffers: a stage arrives a round
                // trip after it was issued, the CTA computes on it and only
                // then reissues the buffer. So d + 1 stages share each round
                // trip plus compute, and synchronous loads (d = 0) pay both
                // every iteration.
                let trips = f64::from(n - d) / f64::from(d + 1);
                (&tl.per_iteration, n, ld, st, trips)
            }
            Phase::Epilogue => {
                let ld = self.traffic(&tl.epilogue.loads, 0, 1, false, sm, program, tile);
                let st = self.traffic(&tl.epilogue.stores, 0, 1, true, sm, program, tile);
                (&tl.epilogue, 1, ld, st, 1.0)
            }
        };
        let (hits, misses) = (loads.resident + loads.merged, loads.dram);
        let stores = stores.total();
        self.merged_bytes += loads.merged;
        let load_bytes = hits + misses;
        self.load_bytes += load_bytes;
        self.l2_hit_bytes += hits;
        self.dram_bytes += misses + stores;
        let spill_executions = match (k.spill.scope, phase) {
            (SpillScope::Once, Phase::Prologue) => 1.0,
            (SpillScope::PerIteration { assumed: false }, Phase::Main) => f64::from(steps),
            (SpillScope::PerIteration { assumed: true }, Phase::Main) => {
                f64::from(steps) * self.m.assumed_spill_scale
            }
            _ => 0.0,
        };
        let (spill_l1, spill_l2, spill_dram) = self.spill;
        let (spill_l1, spill_l2, spill_dram) = (
            spill_l1 * spill_executions,
            spill_l2 * spill_executions,
            spill_dram * spill_executions,
        );
        self.dram_bytes += spill_dram;
        let local_bytes = spill_l1 + spill_l2 + spill_dram;
        let chain_local_bytes = if self.reload_ns > 0.0 && k.spill.bytes_per_thread > 0.0 {
            local_bytes * (1.0 - k.spill.load_bytes_per_thread / k.spill.bytes_per_thread)
        } else {
            local_bytes
        };
        let steps = f64::from(steps);
        let (ws_l2, ws_dram) = if phase == Phase::Main
            && self.m.cube_vector_split
            && k.cv_workspace.loop_bytes_per_iter > 0
        {
            let ws_total = k.cv_workspace.loop_bytes_per_iter as f64 * steps;
            let active_cores = f64::from(self.m.sm_count.min(k.grid).max(1));
            let ws_footprint = k.cv_workspace.bytes_per_core as f64 * active_cores;
            let dram_overflow = if ws_footprint <= self.m.l2_effective_capacity_bytes
                || self.m.l2_effective_capacity_bytes <= 0.0
            {
                0.0
            } else {
                let fit = (self.m.l2_effective_capacity_bytes / ws_footprint).clamp(0.0, 1.0);
                ws_total * (1.0 - fit)
            };
            self.l2_hit_bytes += ws_total - dram_overflow;
            self.dram_bytes += dram_overflow;
            (ws_total, dram_overflow)
        } else {
            (0.0, 0.0)
        };
        let total_bytes = if ws_l2 > 0.0 {
            load_bytes + stores + spill_l2 + spill_dram + ws_l2
        } else {
            load_bytes + stores + spill_l2 + spill_dram
        };
        let total_dram_bytes = if ws_dram > 0.0 {
            misses + stores + spill_dram + ws_dram
        } else {
            misses + stores + spill_dram
        };
        Demand {
            flops: compute.mma_flops * steps,
            warp_insts: (compute.alu_insts_per_warp
                + compute.sfu_insts_per_warp
                + compute.shfl_insts_per_warp)
                * f64::from(k.num_warps)
                * steps,
            sfu_insts: compute.sfu_insts_per_warp * f64::from(k.num_warps) * steps,
            shfl_insts: compute.shfl_insts_per_warp * f64::from(k.num_warps) * steps,
            smem_bytes: compute.smem_bytes * steps,
            mem_insts: compute
                .loads
                .iter()
                .chain(&compute.stores)
                .map(|a| a.nominal_bytes / (32.0 * a.width_bytes))
                .sum::<f64>()
                * steps,
            // Local accesses go through the SM's L1 and LSU wherever they are served.
            local_bytes,
            chain_local_bytes,
            bytes: total_bytes,
            dram_bytes: total_dram_bytes,
            fixed_latency_ns: 0.0,
            load_bytes,
            round_trips: if load_bytes > 0.0 { round_trips } else { 0.0 },
            stages: if phase == Phase::Main { steps } else { 0.0 },
            tokens: d + 1,
            reload_ns: self.reload_ns * spill_executions,
            sync_ns: {
                let base_sync = match phase {
                    Phase::Main => self.m.sync_ns(k.num_warps, d + 1) * steps,
                    _ => 0.0,
                };
                if phase == Phase::Main
                    && self.m.cube_vector_split
                    && k.cv_workspace.sync_hops_per_iter > 0
                    && !self.m.cv_sync_hop_ns.is_empty()
                {
                    let active_cores = f64::from(self.m.sm_count.min(k.grid).max(1));
                    let hop_ns = interpolate(&self.m.cv_sync_hop_ns, active_cores);
                    base_sync + f64::from(k.cv_workspace.sync_hops_per_iter) * hop_ns * steps
                } else {
                    base_sync
                }
            },
            exposed_compute: if load_bytes > 0.0 {
                round_trips / steps
            } else {
                0.0
            },
            resident_fraction: if load_bytes > 0.0 {
                loads.resident / load_bytes
            } else {
                0.0
            },
            dram_fraction: if load_bytes > 0.0 {
                misses / load_bytes
            } else {
                0.0
            },
            cv_overlap: phase == Phase::Main && self.m.cube_vector_split && k.cv_workspace.overlap,
        }
    }

    fn next_phase(phase: Phase) -> Option<Phase> {
        match phase {
            Phase::Launch => Some(Phase::Prologue),
            Phase::Prologue => Some(Phase::Main),
            Phase::Main => Some(Phase::Epilogue),
            Phase::Epilogue => None,
        }
    }

    /// Whether `phase` has no work at all, decided from the kernel alone so
    /// that phase transitions never need the L2 model.
    fn phase_empty(&self, phase: Phase, program: u32, tile: u32) -> bool {
        let k = self.k;
        let tl = &k.tile;
        let idle = |w: &Work| {
            w.mma_flops == 0.0
                && w.alu_insts_per_warp + w.sfu_insts_per_warp + w.shfl_insts_per_warp == 0.0
                && w.smem_bytes == 0.0
        };
        let n = k.iterations(program, tile);
        let spills = |scope: SpillScope| k.spill.scope == scope && k.spill.bytes_per_thread > 0.0;
        let d = tl.prefetch_distance.min(n);
        match phase {
            Phase::Launch => self.m.cta_launch_ns == 0.0,
            Phase::Prologue => {
                idle(&tl.prologue)
                    && tl.prologue.loads.is_empty()
                    && tl.prologue.stores.is_empty()
                    && (d == 0 || tl.per_iteration.loads.is_empty())
                    && !spills(SpillScope::Once)
            }
            Phase::Main => {
                let w = &tl.per_iteration;
                n == 0
                    || (idle(w)
                        && w.stores.is_empty()
                        && (w.loads.is_empty() || d == n)
                        && !matches!(k.spill.scope, SpillScope::PerIteration { .. }))
            }
            Phase::Epilogue => {
                idle(&tl.epilogue) && tl.epilogue.loads.is_empty() && tl.epilogue.stores.is_empty()
            }
        }
    }

    /// Move `c` to its next non-empty phase, recording tile boundaries in
    /// `events` instead of telling the L2 model. Returns false when the CTA
    /// is done.
    fn step(&self, c: &mut Cta, events: &mut Vec<TileEvent>) -> bool {
        loop {
            match Self::next_phase(c.phase) {
                Some(p) => {
                    if p == Phase::Prologue {
                        events.push(TileEvent::Begin(c.sm, c.program, c.tile));
                    }
                    c.phase = p;
                }
                None => {
                    events.push(TileEvent::End(c.sm, c.program, c.tile));
                    c.tile += 1;
                    if c.tile == c.tiles {
                        return false;
                    }
                    events.push(TileEvent::Begin(c.sm, c.program, c.tile));
                    c.phase = Phase::Prologue;
                }
            }
            if !self.phase_empty(c.phase, c.program, c.tile) {
                return true;
            }
        }
    }

    /// A new CTA in its first non-empty phase, or `None` if it has no work.
    fn launch(&self, program: u32, sm: u32, events: &mut Vec<TileEvent>) -> Option<Cta> {
        let tiles = self.k.tiles_of(program);
        if tiles == 0 {
            return None;
        }
        let mut c = Cta {
            program,
            sm,
            tile: 0,
            tiles,
            phase: Phase::Launch,
            remaining: 1.0,
            demand: Demand::default(),
        };
        if self.phase_empty(Phase::Launch, program, 0) && !self.step(&mut c, events) {
            return None;
        }
        Some(c)
    }

    /// Finish one timestamp: tell the L2 model about every tile that ended,
    /// then every tile that began, and only then compute the demands of the
    /// CTAs that changed phase. Every CTA of the batch sees the same resident
    /// set, whatever order the batch was processed in.
    fn settle(&mut self, events: &mut Vec<TileEvent>, changed: &mut Vec<Cta>, ctas: &mut Vec<Cta>) {
        events.sort_unstable();
        for e in events.drain(..) {
            match e {
                TileEvent::End(sm, program, tile) => self.l2.end_tile(sm, program, tile, self.t),
                TileEvent::Begin(sm, program, tile) => {
                    self.l2.begin_tile(sm, program, tile, self.t)
                }
            }
        }
        changed.sort_unstable_by_key(|c| (c.program, c.tile));
        for mut c in changed.drain(..) {
            c.demand = self.demand(c.phase, c.sm, c.program, c.tile);
            c.remaining = 1.0;
            ctas.push(c);
        }
    }
}

/// A tile boundary, applied to the L2 model at the end of a timestamp. Ends
/// sort before begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TileEvent {
    End(u32, u32, u32),
    Begin(u32, u32, u32),
}

/// One CTA's three paths through an iteration, by component.
#[derive(Debug, Clone, Copy, Default)]
struct Parts {
    /// Round-trip path: launch, L2 and DRAM round trips, and compute that
    /// cannot overlap them, with the `RESOURCES` index of that compute.
    launch: f64,
    l2_latency: f64,
    dram_latency: f64,
    exposed: f64,
    exposed_kind: usize,
    /// `RESOURCES` index of what dominates the dependent chain.
    chain_kind: usize,
    /// Shared compute path: pipes by `RESOURCES` index, plus synchronization.
    pipes: [f64; 7],
    sync: f64,
    /// One CTA's dependent chain, a floor under the shared path.
    chain: f64,
    /// Steady state as a closed queueing network, when it applies: the time
    /// mean value analysis gives instead of the round-trip bound.
    queued: Option<f64>,
}

/// Stage throughput of `customers` stage buffers cycling through one
/// queueing server (`service` per stage) and a delay (`delay` per stage):
/// exact mean value analysis of the closed network.
fn mva_throughput(customers: u32, service: f64, delay: f64) -> f64 {
    let (mut queue, mut x) = (0.0, 0.0);
    for n in 1..=customers {
        let response = service * (1.0 + queue);
        x = f64::from(n) / (response + delay);
        queue = x * response;
    }
    x
}

impl Parts {
    /// Round-trip path: the queueing-network time if it applies, else round
    /// trips plus the compute that cannot overlap them.
    fn round_trip(&self) -> f64 {
        self.queued
            .unwrap_or(self.launch + self.l2_latency + self.dram_latency + self.exposed)
    }

    fn compute(&self) -> f64 {
        (self.pipes.iter().fold(0.0f64, |a, &b| a.max(b)) + self.sync).max(self.chain)
    }

    /// What bounds the phase, given its bandwidth times. Bandwidth counts
    /// only when it binds: an unconstrained share equals the demand and
    /// would otherwise tie with the path that set the demand.
    fn bound(&self, t_l2: f64, t_dram: f64) -> usize {
        let (lat, cmp) = (self.round_trip(), self.compute());
        if t_l2.max(t_dram) > lat.max(cmp) * (1.0 + 1e-9) {
            return if t_l2 >= t_dram {
                L2_BANDWIDTH
            } else {
                DRAM_BANDWIDTH
            };
        }
        let argmax = |xs: &[(f64, usize)]| {
            xs.iter()
                .fold((f64::MIN, MMA), |b, &x| if x.0 > b.0 { x } else { b })
                .1
        };
        if lat >= cmp {
            argmax(&[
                (self.launch, LAUNCH),
                (self.l2_latency, L2_LATENCY),
                (self.dram_latency, DRAM_LATENCY),
                (self.exposed, self.exposed_kind),
            ])
        } else {
            let p = self.pipes;
            argmax(&[
                (p[0], 0),
                (p[1], 1),
                (p[2], 2),
                (p[3], 3),
                (p[4], 4),
                (p[5], 5),
                (p[6], 6),
                (self.sync, SYNC),
                (self.chain, self.chain_kind),
            ])
        }
    }
}

/// Reusable buffers for the rate computation.
#[derive(Default)]
struct Scratch {
    mma_users: Vec<u32>,
    issue_users: Vec<u32>,
    sfu_users: Vec<u32>,
    shfl_users: Vec<u32>,
    smem_users: Vec<u32>,
    local_users: Vec<u32>,
    resident: Vec<u32>,
    t_compute: Vec<f64>,
    caps: Vec<f64>,
    l2_share: Vec<f64>,
    dram_share: Vec<f64>,
    rate: Vec<f64>,
    /// Per CTA, the time each resource alone would need, in `RESOURCES` order.
    terms: Vec<Parts>,
    /// Per CTA, the `RESOURCES` index of what bounds its current phase.
    bound: Vec<usize>,
}

fn rates(ctas: &[Cta], m: &Machine, per_cta: &[PerCta], lat: Latency, s: &mut Scratch) {
    let n_sm = m.sm_count as usize;
    s.mma_users.clear();
    s.mma_users.resize(n_sm, 0);
    s.issue_users.clear();
    s.issue_users.resize(n_sm, 0);
    s.sfu_users.clear();
    s.sfu_users.resize(n_sm, 0);
    s.shfl_users.clear();
    s.shfl_users.resize(n_sm, 0);
    s.smem_users.clear();
    s.smem_users.resize(n_sm, 0);
    s.local_users.clear();
    s.local_users.resize(n_sm, 0);
    s.resident.clear();
    s.resident.resize(n_sm, 0);
    for c in ctas {
        let i = c.sm as usize;
        s.resident[i] += 1;
        s.mma_users[i] += u32::from(c.demand.flops > 0.0);
        s.issue_users[i] += u32::from(c.demand.warp_insts > 0.0);
        s.sfu_users[i] += u32::from(c.demand.sfu_insts > 0.0);
        s.shfl_users[i] += u32::from(c.demand.shfl_insts > 0.0);
        s.smem_users[i] += u32::from(c.demand.smem_bytes > 0.0);
        s.local_users[i] += u32::from(c.demand.local_bytes > 0.0);
    }
    let active_sms = s.resident.iter().filter(|&&r| r > 0).count() as f64;

    s.t_compute.clear();
    s.terms.clear();
    for c in ctas {
        let d = &c.demand;
        let i = c.sm as usize;
        let t_mma = if d.flops > 0.0 {
            d.flops / per_cta[s.mma_users[i] as usize].mma
        } else {
            0.0
        };
        let t_issue = if d.warp_insts > 0.0 {
            d.warp_insts * f64::from(s.issue_users[i]) / m.issue_rate
        } else {
            0.0
        };
        let t_smem = if d.smem_bytes > 0.0 {
            d.smem_bytes * f64::from(s.smem_users[i]) / m.smem_bw
        } else {
            0.0
        };
        let t_sfu = if d.sfu_insts > 0.0 {
            d.sfu_insts / per_cta[s.sfu_users[i] as usize].sfu
        } else {
            0.0
        };
        let t_shfl = if d.shfl_insts > 0.0 {
            d.shfl_insts / per_cta[s.shfl_users[i] as usize].shfl
        } else {
            0.0
        };
        let t_lsu = d.mem_insts * f64::from(s.resident[i]) / m.lsu_rate;
        let t_local = if d.local_bytes > 0.0 {
            d.local_bytes * f64::from(s.local_users[i]) / m.local_bw
        } else {
            0.0
        };
        // Round trips plus the compute that cannot overlap them. While a CTA
        // waits, the other CTAs on its SM use the pipes it leaves idle, so
        // that compute runs at the rate the CTA would get alone, and the
        // shared rate only bounds the average. Synchronization stalls the SM
        // for every resident CTA in turn.
        let alone = [
            d.flops / per_cta[1].mma,
            d.warp_insts / m.issue_rate,
            d.smem_bytes / m.smem_bw,
            d.chain_local_bytes / m.local_bw,
            d.sfu_insts / per_cta[1].sfu,
            d.shfl_insts / per_cta[1].shfl,
            // Issuing loads overlaps the wait for the ones in flight, so
            // memory issue bounds only the SM's throughput.
            0.0,
        ];
        // On unified SMs, other pipes depend on MMA results and the next MMAs
        // depend on that work (e.g., softmax between attention's two MMAs), so
        // one CTA takes the sum. On cube_vector_split architectures with a
        // multi-buffered inter-core workspace ring, Cube (AIC) and Vector (AIV)
        // execute in pipeline parallel, so steady-state service takes their max.
        let (other, &t_other) =
            alone[1..]
                .iter()
                .enumerate()
                .fold((0, &0.0), |b, x| if x.1 > b.1 { x } else { b });
        let kind = if alone[0] >= t_other { MMA } else { other + 1 };
        let longest = if d.cv_overlap {
            alone[0].max(t_other)
        } else {
            alone[0] + t_other
        };
        let (launch, l2_latency, dram_latency) =
            d.latency_parts(m.port_floor(lat, d, s.resident[i]));
        // Steady state with compute and loads: the stage buffers of all CTAs
        // on the SM queue for the SM's pipes (service: a stage's compute alone
        // plus its synchronization) and wait a round trip between stages.
        let queued = (d.stages > 0.0 && d.round_trips > 0.0 && longest > 0.0).then(|| {
            let users = s.mma_users[i].max(1);
            let service = (longest + d.sync_ns) / d.stages;
            let delay = (l2_latency + dram_latency) / d.round_trips + d.reload_ns / d.stages;
            let x = mva_throughput(users * d.tokens, service, delay);
            d.stages * f64::from(users) / x
        });
        let parts = Parts {
            launch,
            l2_latency,
            dram_latency,
            exposed: d.exposed_compute * (longest + d.reload_ns),
            exposed_kind: kind,
            chain_kind: if d.reload_ns > longest + d.sync_ns {
                LOCAL
            } else {
                kind
            },
            pipes: [t_mma, t_issue, t_smem, t_local, t_sfu, t_shfl, t_lsu],
            sync: d.sync_ns * f64::from(s.resident[i]),
            chain: longest + d.sync_ns + d.reload_ns,
            queued,
        };
        s.terms.push(parts);
        // Time without memory bandwidth limits: caps the bandwidth demand.
        s.t_compute.push(parts.round_trip().max(parts.compute()));
    }

    let n = ctas.len();
    s.l2_share.resize(n, 0.0);
    s.dram_share.resize(n, 0.0);
    s.caps.clear();
    s.caps.extend(
        ctas.iter()
            .zip(&s.t_compute)
            .map(|(c, &t)| c.demand.bytes / t),
    );
    waterfill(&s.caps, interpolate(&m.l2_bw, active_sms), &mut s.l2_share);
    s.caps.clear();
    s.caps.extend(
        ctas.iter()
            .zip(&s.t_compute)
            .zip(&s.l2_share)
            .map(|((c, &t), &b)| {
                let d = &c.demand;
                if d.bytes == 0.0 {
                    0.0
                } else {
                    d.dram_bytes * (1.0 / t).min(b / d.bytes)
                }
            }),
    );
    waterfill(
        &s.caps,
        interpolate(&m.dram_bw, active_sms),
        &mut s.dram_share,
    );

    s.rate.clear();
    s.bound.clear();
    for (i, c) in ctas.iter().enumerate() {
        let d = &c.demand;
        let t_l2 = if d.bytes > 0.0 {
            d.bytes / s.l2_share[i]
        } else {
            0.0
        };
        let t_dram = if d.dram_bytes > 0.0 {
            d.dram_bytes / s.dram_share[i]
        } else {
            0.0
        };
        let t_mem = t_l2.max(t_dram);
        let parts = s.terms[i];
        s.bound.push(parts.bound(t_l2, t_dram));
        // The loaded latency already contains queueing and transfer under
        // the current load, and DRAM and L2 serve whichever CTAs have requests
        // outstanding, so the bandwidth share only bounds the average rate,
        // for synchronous loads too.
        let t = parts.round_trip().max(parts.compute()).max(t_mem);
        // A phase with no demand finishes at once.
        s.rate.push(1.0 / t.max(1e-9));
    }
}

/// A phase whose remaining fraction is below this after an event is treated
/// as finished in that event. Merging nearly simultaneous completions cuts the
/// event count about fourfold on desynchronized grids, for a time error of at
/// most this fraction of one phase.
const COMPLETION_TOLERANCE: f64 = 1e-3;

/// Bisection steps for the latency and bandwidth equilibrium of an event:
/// DRAM utilization is resolved to 1/4096, or, with MLP curves, DRAM latency
/// to 0.2% over its range.
const EQUILIBRIUM_STEPS: u32 = 12;

/// Simulate one kernel variant at one profile draw.
pub fn simulate<L: L2>(
    k: &Kernel,
    p: &Profile,
    m: &Machine,
    l2: &mut L,
) -> Result<Prediction, ModelError> {
    let occ = occupancy(k, &p.specs).map_err(ModelError::Occupancy)?;
    let n_sm = m.sm_count;
    let slots = occ.ctas_per_sm * n_sm;
    // Spill traffic at full residency, computed once for the steady state.
    let threads_in_sm = occ.ctas_per_sm * k.num_warps * p.specs.warp_size;
    let split = match k.spill.scope {
        SpillScope::None => SpillSplit::default(),
        _ => l2.spill(0, threads_in_sm, occ.carveout_bytes, &k.spill),
    };
    let per_cta = f64::from(occ.ctas_per_sm);
    let spill = (split.l1 / per_cta, split.l2 / per_cta, split.dram / per_cta);
    let reload_ns = if k.spill.scope == SpillScope::None {
        0.0
    } else {
        m.spill_reload_ns(
            k.spill.load_bytes_per_thread / SPILL_RELOAD_BYTES,
            k.spill.stack_frame_bytes as f64 * f64::from(threads_in_sm),
            split.miss,
        )
    };
    let mut sim = Sim {
        k,
        m,
        l2,
        t: 0.0,
        load_bytes: 0.0,
        l2_hit_bytes: 0.0,
        merged_bytes: 0.0,
        dram_bytes: 0.0,
        spill,
        reload_ns,
    };

    // First wave: breadth-first round robin over SMs. Later: first SM to free a slot.
    let mut ctas: Vec<Cta> = Vec::with_capacity(slots as usize);
    let mut changed: Vec<Cta> = Vec::new();
    let mut tile_events: Vec<TileEvent> = Vec::new();
    let mut next = 0;
    while next < k.grid && (changed.len() as u32) < slots {
        let sm = changed.len() as u32 % n_sm;
        if let Some(c) = sim.launch(next, sm, &mut tile_events) {
            changed.push(c);
        }
        next += 1;
    }
    sim.settle(&mut tile_events, &mut changed, &mut ctas);

    let warps = f64::from(k.num_warps);
    let per_cta: Vec<PerCta> = (0..=occ.ctas_per_sm)
        .map(|u| {
            let users = f64::from(u.max(1));
            let at = |c: &[(f64, f64)]| interpolate(c, users * warps) / users;
            PerCta {
                mma: at(&m.mma),
                sfu: at(&m.sfu),
                shfl: at(&m.shfl),
            }
        })
        .collect();
    let mut s = Scratch::default();
    let mut freed: Vec<u32> = Vec::new();
    let mut busy_sm_ns = 0.0;
    let mut events = 0u64;
    let mut bound_ns = [0.0f64; 13];
    // Latency depends on the DRAM and L2 utilization and the utilization on
    // the rates, which depend on latency. DRAM utilization falls as the
    // latency it implies rises, so the equilibrium is unique and bisection
    // finds it even where latency is steep in load. L2 utilization, flatter
    // and secondary, follows the last pass.
    let (dram_peak, l2_peak) = (Machine::peak(&m.dram_bw), Machine::peak(&m.l2_bw));
    let mut l2_util = 0.0;
    let achieved = |ctas: &[Cta], rate: &[f64]| {
        ctas.iter().zip(rate).fold((0.0, 0.0), |(d, l), (c, &r)| {
            (d + r * c.demand.dram_bytes, l + r * c.demand.bytes)
        })
    };
    let mode = usize::from(!k.async_loads);
    let mlp = !m.dram_mlp[mode].is_empty() && !m.l2_mlp[mode].is_empty();
    while !ctas.is_empty() {
        if mlp {
            // Little's law on the measured MLP curves. The load bytes a level
            // holds in flight are its delivered load bandwidth times its
            // latency, and the curve turns them into that latency. Bytes in
            // flight grow with the latency assumed, so the DRAM latency that
            // reproduces itself is found by bisection; L2 latency, secondary,
            // follows the last pass.
            let (mut lo, mut hi) = (m.mlp_latency(mode, 0.0, 0.0).dram, 1e6f64);
            let mut l2_lat = m.mlp_latency(mode, 0.0, 0.0).l2;
            for _ in 0..EQUILIBRIUM_STEPS {
                let mid = (lo * hi).sqrt();
                rates(
                    &ctas,
                    m,
                    &per_cta,
                    Latency {
                        dram: mid,
                        l2: l2_lat,
                    },
                    &mut s,
                );
                let (dram_bw, l2_bw) =
                    ctas.iter()
                        .zip(&s.rate)
                        .fold((0.0, 0.0), |(d, l), (c, &r)| {
                            let x = r * c.demand.load_bytes;
                            (
                                d + x * c.demand.dram_fraction,
                                l + x * c.demand.resident_fraction,
                            )
                        });
                let implied = m.mlp_latency(mode, dram_bw * mid, l2_bw * l2_lat);
                l2_lat = implied.l2;
                if implied.dram > mid {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            rates(
                &ctas,
                m,
                &per_cta,
                Latency {
                    dram: (lo * hi).sqrt(),
                    l2: l2_lat,
                },
                &mut s,
            );
        } else {
            let (mut lo, mut hi) = (0.0, 1.0);
            for _ in 0..EQUILIBRIUM_STEPS {
                let mid = 0.5 * (lo + hi);
                rates(&ctas, m, &per_cta, m.latency(mid, l2_util), &mut s);
                let (d, l) = achieved(&ctas, &s.rate);
                l2_util = l / l2_peak;
                if d / dram_peak > mid {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            rates(
                &ctas,
                m,
                &per_cta,
                m.latency(0.5 * (lo + hi), l2_util),
                &mut s,
            );
        }
        let dt = ctas
            .iter()
            .zip(&s.rate)
            .map(|(c, &r)| c.remaining / r)
            .fold(f64::INFINITY, f64::min);
        sim.t += dt;
        busy_sm_ns += dt * s.resident.iter().filter(|&&r| r > 0).count() as f64;
        for &b in &s.bound {
            bound_ns[b] += dt;
        }
        events += 1;

        for (c, &r) in ctas.iter_mut().zip(&s.rate) {
            c.remaining -= r * dt;
        }
        freed.clear();
        let mut i = 0;
        while i < ctas.len() {
            if ctas[i].remaining > COMPLETION_TOLERANCE {
                i += 1;
                continue;
            }
            let mut c = ctas.swap_remove(i);
            if sim.step(&mut c, &mut tile_events) {
                changed.push(c);
            } else {
                freed.push(c.sm);
            }
        }
        freed.sort_unstable();
        for &sm in &freed {
            while next < k.grid {
                let program = next;
                next += 1;
                if let Some(c) = sim.launch(program, sm, &mut tile_events) {
                    changed.push(c);
                    break;
                }
            }
        }
        sim.settle(&mut tile_events, &mut changed, &mut ctas);
        #[cfg(test)]
        shuffle(&mut ctas, events);
    }

    let makespan = sim.t;
    Ok(Prediction {
        time_ns: m.launch_overhead_ns + makespan,
        makespan_ns: makespan,
        waves: f64::from(k.grid) / f64::from(slots),
        tail_efficiency: if makespan > 0.0 {
            busy_sm_ns / (f64::from(n_sm) * makespan)
        } else {
            1.0
        },
        occupancy: occ,
        load_bytes: sim.load_bytes,
        l2_hit_bytes: sim.l2_hit_bytes,
        merged_bytes: sim.merged_bytes,
        dram_bytes: sim.dram_bytes,
        events,
        bound_fraction: {
            let total: f64 = bound_ns.iter().sum();
            bound_ns.map(|b| if total > 0.0 { b / total } else { 0.0 })
        },
    })
}

// Tests permute the active CTAs after every event to check that results do
// not depend on the order CTAs are stored and processed in.
#[cfg(test)]
thread_local! {
    static SHUFFLE_SEED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn shuffle(ctas: &mut [Cta], salt: u64) {
    let seed = SHUFFLE_SEED.with(|s| s.get());
    if seed == 0 {
        return;
    }
    let mut x = seed ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for i in (1..ctas.len()).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ctas.swap(i, (x % (i as u64 + 1)) as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::from_ir::{kernel_from_ir, IrL2};
    use crate::ir::KernelIR;
    use crate::l2::{L2InitState, L2Options};

    fn run(ir: &KernelIR, p: &Profile, initial: L2InitState, seed: u64) -> Prediction {
        SHUFFLE_SEED.with(|s| s.set(seed));
        let k = kernel_from_ir(ir).unwrap();
        let m = Machine::nominal(p, &k.mma_key).unwrap();
        let mut l2 = IrL2::new(ir, p, initial, L2Options::default());
        let r = simulate(&k, p, &m, &mut l2).unwrap();
        SHUFFLE_SEED.with(|s| s.set(0));
        r
    }

    #[test]
    fn processing_order_within_a_timestamp_does_not_matter() {
        let ir = KernelIR::from_json_str(include_str!("../../../schemas/examples/gemm_sm89.json"))
            .unwrap();
        let p = Profile::from_json(include_str!("../../../profiles/rtx4070.json")).unwrap();
        for initial in [L2InitState::Cold, L2InitState::Warm] {
            let base = run(&ir, &p, initial, 0);
            for seed in [1, 2, 3] {
                assert_eq!(run(&ir, &p, initial, seed), base, "{initial:?} seed {seed}");
            }
        }
    }

    #[test]
    fn reload_latency_takes_spill_loads_off_the_chain() {
        let mut v: serde_json::Value =
            serde_json::from_str(include_str!("../../../schemas/examples/gemm_sm89.json")).unwrap();
        let r = &mut v["resources"];
        r["stack_frame_bytes"] = 512.into();
        r["spill_loads_bytes"] = 384.into();
        r["spill_stores_bytes"] = 128.into();
        r["spill_scope"] = "inner_loop_confirmed".into();
        let ir = KernelIR::from_json_str(&v.to_string()).unwrap();
        let p = Profile::from_json(include_str!("../../../profiles/rtx4070.json")).unwrap();
        let k = kernel_from_ir(&ir).unwrap();
        let m = Machine::nominal(&p, &k.mma_key).unwrap();
        let mut l2 = IrL2::new(&ir, &p, L2InitState::Cold, L2Options::default());
        let mut sim = Sim {
            k: &k,
            m: &m,
            l2: &mut l2,
            t: 0.0,
            load_bytes: 0.0,
            l2_hit_bytes: 0.0,
            merged_bytes: 0.0,
            dram_bytes: 0.0,
            spill: (1000.0, 0.0, 0.0),
            reload_ns: 0.0,
        };
        let d = sim.demand(Phase::Main, 0, 0, 0);
        assert!(d.local_bytes > 0.0);
        assert_eq!(d.chain_local_bytes, d.local_bytes);
        sim.reload_ns = 50.0;
        let d = sim.demand(Phase::Main, 0, 0, 0);
        assert!((d.chain_local_bytes - d.local_bytes * 0.25).abs() < 1e-9);
    }
}
