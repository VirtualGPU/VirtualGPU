//! Post-hoc: the model on the sync-loop-overlap, sync-loop-bytes and sync-loop-miss microbenchmark points.
//! Ignored by default. Run one test with PROFILE (profile JSON), POINTS (benchmark JSON) and
//! OUT (output text), for example:
//! PROFILE=profiles/gb10.json POINTS=bytes.json OUT=model.txt \
//!   cargo test -p vgpu-core --test posthoc_sync_loop sync_loop_bytes -- --ignored
use serde_json::Value;
use vgpu_core::kernel::{Access, Kernel, Spill, Tile, TripCounts, Work};
use vgpu_core::profile::Profile;
use vgpu_core::sim::{simulate, ConstantHitRatio, Machine, Traffic, L2};

const MMA_KEY: &str = "mma.sync:f16:f16:f32:16x8x16";
const STEPS: u32 = 512;

struct Point {
    grid: u32,
    ctas_per_sm: u32,
    num_warps: u32,
    stage_bytes: u32,
    mma_per_warp: u32,
    attention: bool,
    dep: String,
    dram: bool,
}

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name}"))
}

/// Fixed split of load bytes into resident, merged and DRAM fractions; stores go to DRAM.
struct FixedSplit(f64, f64);

impl L2 for FixedSplit {
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
        if is_write {
            return Traffic {
                resident: 0.0,
                merged: 0.0,
                dram: bytes,
            };
        }
        Traffic {
            resident: bytes * (1.0 - self.0 - self.1),
            merged: bytes * self.0,
            dram: bytes * self.1,
        }
    }
}

/// Per-step model time for one point, with launch costs removed.
fn per_step_ns(p: &Profile, q: &Point) -> f64 {
    let hit = if q.dram { 0.0 } else { 1.0 };
    per_step_split_ns(p, q, &mut ConstantHitRatio(hit))
}

fn per_step_split_ns(p: &Profile, q: &Point, l2: &mut impl L2) -> f64 {
    let mut m = Machine::nominal(p, MMA_KEY).unwrap();
    m.launch_overhead_ns = 0.0;
    m.cta_launch_ns = 0.0;
    let blocks = if q.attention { 2.0 } else { 1.0 };
    let compute = q.dep != "nocompute";
    let bytes = f64::from(q.stage_bytes);
    let work = Work {
        mma_flops: if compute {
            f64::from(q.num_warps * q.mma_per_warp) * 4096.0 * blocks
        } else {
            0.0
        },
        sfu_insts_per_warp: if q.attention && compute { 32.0 } else { 0.0 },
        shfl_insts_per_warp: if q.attention && compute { 4.0 } else { 0.0 },
        alu_insts_per_warp: if q.attention && compute { 45.0 } else { 8.0 },
        smem_bytes: bytes,
        loads: if q.dep == "noload" {
            vec![]
        } else {
            vec![Access {
                op_id: "ld".into(),
                nominal_bytes: bytes,
                width_bytes: 16.0,
            }]
        },
        ..Work::default()
    };
    let kernel = Kernel {
        grid: q.grid,
        num_warps: q.num_warps,
        registers_per_thread: 64,
        smem_per_cta_bytes: (99 * 1024 / u64::from(q.ctas_per_sm) - 1024)
            .max(u64::from(q.stage_bytes)),
        ub_bytes: 0,
        l0c_bytes: 0,
        cv_workspace: vgpu_core::kernel::CvWorkspace::default(),
        tiles_per_program: TripCounts::Uniform(1),
        mma_key: MMA_KEY.to_string(),
        spill: Spill::default(),
        async_loads: true,
        tile_iterations: None,
        tile: Tile {
            prologue: Work::default(),
            iterations: STEPS,
            per_iteration: work,
            prefetch_distance: 0,
            epilogue: Work::default(),
        },
    };
    simulate(&kernel, p, &m, l2).unwrap().makespan_ns / f64::from(STEPS)
}

fn median(q: &Value) -> f64 {
    let mut s: Vec<f64> = q["turnaround_ns_samples"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

fn run(list: &str, point: impl Fn(&Value) -> (Point, String)) {
    let p = Profile::from_json(&std::fs::read_to_string(env("PROFILE")).unwrap()).unwrap();
    let pts: Value =
        serde_json::from_str(&std::fs::read_to_string(env("POINTS")).unwrap()).unwrap();
    let mut out = String::new();
    for q in pts[list].as_array().unwrap() {
        let (pt, label) = point(q);
        out += &format!("{label} {:.1} {:.1}\n", per_step_ns(&p, &pt), median(q));
    }
    std::fs::write(env("OUT"), out).unwrap();
}

fn u(q: &Value, key: &str) -> u32 {
    u32::try_from(q[key].as_u64().unwrap()).unwrap()
}

/// Lines: pattern dep mem_level ctas_per_sm num_warps stage_bytes mma model_ns measured_ns.
#[test]
#[ignore = "post-hoc analysis, needs PROFILE, POINTS and OUT"]
fn sync_loop_overlap() {
    run("sync_loop_overlap", |q| {
        let pt = Point {
            grid: u(q, "active_ctas"),
            ctas_per_sm: u(q, "ctas_per_sm"),
            num_warps: u(q, "num_warps"),
            stage_bytes: u(q, "stage_bytes"),
            mma_per_warp: u(q, "mma_per_warp_per_block"),
            attention: q["pattern"] == "attention",
            dep: q["dep"].as_str().unwrap().to_string(),
            dram: q["mem_level"] == "dram",
        };
        let label = format!(
            "{} {} {} {} {} {} {}",
            q["pattern"].as_str().unwrap(),
            pt.dep,
            q["mem_level"].as_str().unwrap(),
            pt.ctas_per_sm,
            pt.num_warps,
            pt.stage_bytes,
            pt.mma_per_warp
        );
        (pt, label)
    });
}

/// Lines: dep grid num_warps stage_bytes model_ns measured_ns (gemm_s2 shape, 32 MMAs per warp, L2).
#[test]
#[ignore = "post-hoc analysis, needs PROFILE, POINTS and OUT"]
fn sync_loop_bytes() {
    run("sync_loop_bytes", |q| {
        let pt = Point {
            grid: u(q, "grid"),
            ctas_per_sm: 1,
            num_warps: u(q, "num_warps"),
            stage_bytes: u(q, "stage_bytes"),
            mma_per_warp: 32,
            attention: false,
            dep: q["dep"].as_str().unwrap().to_string(),
            dram: false,
        };
        let label = format!("{} {} {} {}", pt.dep, pt.grid, pt.num_warps, pt.stage_bytes);
        (pt, label)
    });
}

/// Lines: dep grid group stage_bytes cold_fraction_den model_ns measured_ns (gemm_s2 shape, 8 warps,
/// 32 MMAs per warp). The model gets the benchmark's split: the cold fraction f misses, and with
/// groups of G CTAs reading the same cold addresses all but 1/G of it merges.
#[test]
#[ignore = "post-hoc analysis, needs PROFILE, POINTS and OUT"]
fn sync_loop_miss() {
    let p = Profile::from_json(&std::fs::read_to_string(env("PROFILE")).unwrap()).unwrap();
    let pts: Value =
        serde_json::from_str(&std::fs::read_to_string(env("POINTS")).unwrap()).unwrap();
    let mut out = String::new();
    for q in pts["sync_loop_miss"].as_array().unwrap() {
        let (grid, group, fd) = (u(q, "grid"), u(q, "group"), u(q, "cold_fraction_den"));
        let pt = Point {
            grid,
            ctas_per_sm: 1,
            num_warps: u(q, "num_warps"),
            stage_bytes: u(q, "stage_bytes"),
            mma_per_warp: 32,
            attention: false,
            dep: q["dep"].as_str().unwrap().to_string(),
            dram: false,
        };
        let f = if fd == 0 { 0.0 } else { 1.0 / f64::from(fd) };
        let first = if grid == 1 {
            1.0
        } else {
            1.0 / f64::from(group)
        };
        let mut split = FixedSplit(f * (1.0 - first), f * first);
        let model = per_step_split_ns(&p, &pt, &mut split);
        out += &format!(
            "{} {grid} {group} {} {fd} {model:.1} {:.1}\n",
            pt.dep,
            pt.stage_bytes,
            median(q)
        );
    }
    std::fs::write(env("OUT"), out).unwrap();
}

/// Lines: ir_path variant merged dram time_ns. For each KernelIR in IRS (comma separated, warm L2):
/// the model as is (`ir`, with its merged and DRAM bytes over load bytes; DRAM includes stores), without the loop's loads (`noload`), and with a fixed split of every load
/// into resident, merged and DRAM fractions (`split`).
#[test]
#[ignore = "post-hoc analysis, needs PROFILE, IRS and OUT"]
fn attention_split() {
    use vgpu_core::from_ir::{kernel_from_ir, IrL2};
    use vgpu_core::ir::KernelIR;
    use vgpu_core::l2::{L2InitState, L2Options};
    let p = Profile::from_json(&std::fs::read_to_string(env("PROFILE")).unwrap()).unwrap();
    let mut out = String::new();
    for path in env("IRS").split(',') {
        let ir = KernelIR::from_json_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let k = kernel_from_ir(&ir).unwrap();
        let m = Machine::nominal(&p, &k.mma_key).unwrap();
        let mut l2 = IrL2::new(&ir, &p, L2InitState::Warm, L2Options::default());
        let base = simulate(&k, &p, &m, &mut l2).unwrap();
        out += &format!(
            "{path} ir {:.4} {:.4} {:.1}\n",
            base.merged_bytes / base.load_bytes,
            base.dram_bytes / base.load_bytes,
            base.time_ns
        );
        let mut nl = k.clone();
        nl.tile.per_iteration.loads.clear();
        let mut l2 = IrL2::new(&ir, &p, L2InitState::Warm, L2Options::default());
        out += &format!(
            "{path} noload 0 0 {:.1}\n",
            simulate(&nl, &p, &m, &mut l2).unwrap().time_ns
        );
        for merged in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0] {
            for dram in [0.0, 1.0 / 16.0] {
                if merged + dram > 1.0 {
                    continue;
                }
                let t = simulate(&k, &p, &m, &mut FixedSplit(merged, dram))
                    .unwrap()
                    .time_ns;
                out += &format!("{path} split {merged} {dram} {t:.1}\n");
            }
        }
    }
    std::fs::write(env("OUT"), out).unwrap();
}

/// Lines: dep start grid stage_bytes model_ns measured_ns (gemm_s2 shape, 8 warps, 32 MMAs per warp,
/// warm L2). The model does not see the start offsets, so lock and stag points get the same time.
#[test]
#[ignore = "post-hoc analysis, needs PROFILE, POINTS and OUT"]
fn sync_loop_burst() {
    run("sync_loop_burst", |q| {
        let pt = Point {
            grid: u(q, "grid"),
            ctas_per_sm: 1,
            num_warps: u(q, "num_warps"),
            stage_bytes: u(q, "stage_bytes"),
            mma_per_warp: 32,
            attention: false,
            dep: q["dep"].as_str().unwrap().to_string(),
            dram: false,
        };
        let label = format!(
            "{} {} {} {}",
            pt.dep,
            q["start"].as_str().unwrap(),
            pt.grid,
            pt.stage_bytes
        );
        (pt, label)
    });
}
