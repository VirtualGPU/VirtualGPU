//! Online, phase-segment-level L2 cache reuse model (v2: empirical rescan curve + O(1) reuse distance).

use crate::ir::{AccessEntry, CachePolicy, KernelIR, Op, Strided2dEntry};
use crate::profile::{interpolate, Profile};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2InitState {
    Cold,
    Warm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionMode {
    SoftLru,
    StrictLru,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2Options {
    pub eviction_mode: EvictionMode,
    pub cross_wave_reuse: bool,
}

impl Default for L2Options {
    fn default() -> Self {
        Self {
            eviction_mode: EvictionMode::SoftLru,
            cross_wave_reuse: true,
        }
    }
}

/// Perturbable hardware L2/L1 parameters extracted from a GPU profile sample.
#[derive(Debug, Clone, PartialEq)]
pub struct L2Profile {
    pub l2_size_bytes: u64,
    pub l2_effective_capacity_bytes: u64,
    /// Unperturbed nominal effective L2 capacity; perturbation shifts the rescan cliff by
    /// `l2_effective_capacity_bytes / nominal_l2_effective_capacity_bytes`.
    pub nominal_l2_effective_capacity_bytes: u64,
    pub l2_line_bytes: u64,
    pub l2_sector_bytes: u64,
    pub l1_size_bytes: u64,
    pub l1_to_l2_spill_threshold_bytes_per_sm: u64,
    /// Monotone non-increasing `(working_set_bytes, hit_rate)` curve derived from
    /// `measured.memory.l2_rescan_hit_rate_by_working_set`. Empty falls back to a step at 1.0x.
    pub rescan_hit_curve: Vec<(f64, f64)>,
}

/// Builds a sorted, monotone non-increasing `(working_set_bytes, hit_rate)` curve from measured
/// rescan points. Enforcing a suffix-max envelope removes small-buffer ramp-up artifacts while
/// preserving the measured eviction cliff.
fn build_monotone_rescan_curve(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut curve: Vec<(f64, f64)> = points
        .iter()
        .filter(|&&(b, h)| b.is_finite() && b > 0.0 && h.is_finite())
        .map(|&(b, h)| (b, h.clamp(0.0, 1.0)))
        .collect();
    curve.sort_by(|a, b| a.0.total_cmp(&b.0));
    for i in (1..curve.len()).rev() {
        if curve[i - 1].1 < curve[i].1 {
            curve[i - 1].1 = curve[i].1;
        }
    }
    curve
}

impl From<&Profile> for L2Profile {
    fn from(p: &Profile) -> Self {
        let s = &p.specs;
        let spill_threshold = p
            .measured
            .spill
            .get("l1_to_l2_spill_threshold_bytes_per_sm")
            .and_then(Option::as_ref)
            .map_or(s.l1_size_bytes, |v| v.value.max(0.0).round() as u64);
        let eff_cap = p
            .measured
            .memory
            .l2_effective_capacity_bytes
            .value
            .max(0.0)
            .round() as u64;
        let raw_curve: Vec<(f64, f64)> = p
            .measured
            .memory
            .l2_rescan_hit_rate_by_working_set
            .iter()
            .map(|pt| (pt.working_set_bytes as f64, pt.estimated_hit_rate.value))
            .collect();
        Self {
            l2_size_bytes: s.l2_size_bytes,
            l2_effective_capacity_bytes: eff_cap,
            nominal_l2_effective_capacity_bytes: eff_cap,
            l2_line_bytes: s.l2_line_bytes,
            l2_sector_bytes: s.l2_sector_bytes,
            l1_size_bytes: s.l1_size_bytes,
            l1_to_l2_spill_threshold_bytes_per_sm: spill_threshold,
            rescan_hit_curve: build_monotone_rescan_curve(&raw_curve),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct L2SplitResult {
    /// Bytes already resident in L2 at request time (warm L2 or surviving cross-wave data):
    /// served at L2 bandwidth and L2 latency.
    pub resident_hit_bytes: f64,
    /// Bytes coalesced into an in-flight DRAM fill from concurrent resident CTAs (MSHR miss merge):
    /// served at L2 bandwidth (no extra DRAM bandwidth), but waits on DRAM fill latency.
    pub merged_bytes: f64,
    /// Total bytes served by L2 (`resident_hit_bytes + merged_bytes`).
    pub l2_bytes: f64,
    /// Bytes fetched from DRAM (DRAM bandwidth and DRAM latency).
    pub dram_bytes: f64,
    pub hit_ratio: f64,
    pub sector_aligned_bytes: u64,
    pub near_cliff: bool,
}

impl L2SplitResult {
    fn zero() -> Self {
        Self {
            resident_hit_bytes: 0.0,
            merged_bytes: 0.0,
            l2_bytes: 0.0,
            dram_bytes: 0.0,
            hit_ratio: 0.0,
            sector_aligned_bytes: 0,
            near_cliff: false,
        }
    }

    fn all_dram(sector_aligned_bytes: u64) -> Self {
        Self {
            resident_hit_bytes: 0.0,
            merged_bytes: 0.0,
            l2_bytes: 0.0,
            dram_bytes: sector_aligned_bytes as f64,
            hit_ratio: 0.0,
            sector_aligned_bytes,
            near_cliff: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpillTraffic {
    pub l1_bytes_per_iter: u64,
    pub l2_bytes_per_iter: u64,
    pub dram_bytes_per_iter: u64,
    pub overflow_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ChunkKey {
    tensor_idx: usize,
    base_byte_offset: u64,
    rows: u64,
    row_bytes: u64,
    row_stride_bytes: u64,
    inner_step_byte_stride: i64,
}

#[derive(Debug, Clone)]
struct OpBinding {
    tensor_idx: usize,
    access_ref: String,
    is_write: bool,
}

#[derive(Debug, Clone)]
struct ResidentChunk {
    key: ChunkKey,
    step_aligned_bytes: u64,
    effective_steps: u64,
    full_stream_bytes: u64,
}

#[derive(Debug, Clone)]
struct ResidentTile {
    chunks: Vec<ResidentChunk>,
}

#[derive(Debug, Clone)]
struct ActiveChunk {
    /// Effective inner-step trip counts of each currently resident CTA sharing this chunk.
    trips: Vec<u64>,
    max_streamed_steps: u64,
    max_streamed_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct LastSeen {
    stream_clock_bytes: u64,
    covered_steps: u64,
}

/// Stateful, instance-local L2 reuse model driven by the event scheduler in `sim.rs`.
pub struct L2Model<'a> {
    ir: &'a KernelIR,
    profile: L2Profile,
    initial_state: L2InitState,
    options: L2Options,
    op_bindings: HashMap<String, OpBinding>,
    load_bindings: Vec<OpBinding>,
    total_read_footprint_bytes: u64,
    resident_tiles: HashMap<(u32, u32), ResidentTile>,
    active_chunks: HashMap<ChunkKey, ActiveChunk>,
    active_working_set_bytes: u64,
    /// Monotonic counter of unique full-K-stream bytes that have flowed through L2.
    stream_clock_bytes: u64,
    /// O(1) map from ChunkKey to its last stream clock and prefix step coverage.
    last_seen_stream_byte: HashMap<ChunkKey, LastSeen>,
    spill_l2_pressure_bytes: u64,
}

impl<'a> L2Model<'a> {
    pub fn new(
        ir: &'a KernelIR,
        profile: L2Profile,
        initial_state: L2InitState,
        options: L2Options,
    ) -> Self {
        let tensor_index: HashMap<&str, usize> = ir
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.id.as_str(), i))
            .collect();

        let mut op_bindings = HashMap::new();
        let mut load_bindings = Vec::new();
        let mut seen_load_refs = HashMap::new();

        let all_ops = ir
            .body
            .prologue
            .ops
            .iter()
            .chain(ir.body.pipeline_loop.ops.iter())
            .chain(ir.body.epilogue.ops.iter());

        for op in all_ops {
            let (g, is_write) = match op {
                Op::GmemLoad(g) => (g, false),
                Op::GmemStore(g) => (g, true),
                _ => continue,
            };
            if let Some(&tensor_idx) = tensor_index.get(g.tensor_id.as_str()) {
                let binding = OpBinding {
                    tensor_idx,
                    access_ref: g.access_ref.clone(),
                    is_write,
                };
                if !is_write
                    && seen_load_refs
                        .insert((tensor_idx, g.access_ref.clone()), ())
                        .is_none()
                {
                    load_bindings.push(binding.clone());
                }
                op_bindings.insert(g.op_id.clone(), binding);
            }
        }

        let mut model = Self {
            ir,
            profile,
            initial_state,
            options,
            op_bindings,
            load_bindings,
            total_read_footprint_bytes: 0,
            resident_tiles: HashMap::new(),
            active_chunks: HashMap::new(),
            active_working_set_bytes: 0,
            stream_clock_bytes: 0,
            last_seen_stream_byte: HashMap::new(),
            spill_l2_pressure_bytes: 0,
        };
        model.total_read_footprint_bytes = model.compute_total_read_footprint();
        model
    }

    pub fn begin_tile(&mut self, _sm_id: u32, cta_id: u32, outer_idx: u32, _t_ns: f64) {
        let tile_key = (cta_id, outer_idx);
        if self.resident_tiles.contains_key(&tile_key) {
            return;
        }
        let inner_trips = self.ir.inner_trip_count(cta_id, outer_idx).unwrap_or(1);
        let mut chunks = Vec::with_capacity(self.load_bindings.len());
        for b in &self.load_bindings {
            if let Some(entry) = self.lookup_strided(&b.access_ref, cta_id, outer_idx) {
                let ckey = make_chunk_key(b.tensor_idx, entry);
                let step_aligned = self.sector_aligned_step_bytes(entry);
                if step_aligned == 0 {
                    continue;
                }
                let effective_steps = effective_chunk_steps(entry, inner_trips);
                let full_stream = step_aligned.saturating_mul(effective_steps);
                let slot = self
                    .active_chunks
                    .entry(ckey)
                    .or_insert_with(|| ActiveChunk {
                        trips: Vec::new(),
                        max_streamed_steps: 0,
                        max_streamed_bytes: 0,
                    });
                if slot.trips.is_empty() {
                    self.active_working_set_bytes =
                        self.active_working_set_bytes.saturating_add(step_aligned);
                }
                slot.trips.push(effective_steps);
                slot.max_streamed_steps = slot.max_streamed_steps.max(effective_steps);
                slot.max_streamed_bytes = slot.max_streamed_bytes.max(full_stream);
                chunks.push(ResidentChunk {
                    key: ckey,
                    step_aligned_bytes: step_aligned,
                    effective_steps,
                    full_stream_bytes: full_stream,
                });
            }
        }
        self.resident_tiles
            .insert(tile_key, ResidentTile { chunks });
    }

    pub fn end_tile(&mut self, _sm_id: u32, cta_id: u32, outer_idx: u32, _t_ns: f64) {
        let Some(tile) = self.resident_tiles.remove(&(cta_id, outer_idx)) else {
            return;
        };
        for rc in tile.chunks {
            if let Some(slot) = self.active_chunks.get_mut(&rc.key) {
                if let Some(pos) = slot.trips.iter().position(|&t| t == rc.effective_steps) {
                    slot.trips.swap_remove(pos);
                }
                slot.max_streamed_steps = slot.max_streamed_steps.max(rc.effective_steps);
                slot.max_streamed_bytes = slot.max_streamed_bytes.max(rc.full_stream_bytes);
                if slot.trips.is_empty() {
                    let streamed_bytes = slot.max_streamed_bytes;
                    let streamed_steps = slot.max_streamed_steps;
                    self.active_chunks.remove(&rc.key);
                    self.active_working_set_bytes = self
                        .active_working_set_bytes
                        .saturating_sub(rc.step_aligned_bytes);
                    // Advance L2 stream clock by the full K-stream volume in O(1).
                    self.stream_clock_bytes =
                        self.stream_clock_bytes.saturating_add(streamed_bytes);
                    if self.options.cross_wave_reuse {
                        self.last_seen_stream_byte.insert(
                            rc.key,
                            LastSeen {
                                stream_clock_bytes: self.stream_clock_bytes,
                                covered_steps: streamed_steps,
                            },
                        );
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn split(
        &mut self,
        _sm_id: u32,
        cta_id: u32,
        outer_idx: u32,
        op_id: &str,
        k_begin: u64,
        k_end: u64,
        is_write: bool,
        cache_policy: CachePolicy,
        _t_ns: f64,
    ) -> L2SplitResult {
        let steps = k_end.saturating_sub(k_begin);
        if steps == 0 {
            return L2SplitResult::zero();
        }

        let Some(binding) = self.op_bindings.get(op_id) else {
            return L2SplitResult::zero();
        };

        let Some(entry) = self.lookup_strided(&binding.access_ref, cta_id, outer_idx) else {
            return L2SplitResult::zero();
        };

        let step_aligned = self.sector_aligned_step_bytes(entry);
        let total_aligned = step_aligned.saturating_mul(steps);
        let total_f = total_aligned as f64;

        if is_write || binding.is_write {
            // Intentional M1/M2 simplification: treat all global stores as DRAM write-back traffic
            // without modeling write-allocate L2 hit credit.
            return L2SplitResult::all_dram(total_aligned);
        }

        if cache_policy == CachePolicy::Volatile {
            return L2SplitResult::all_dram(total_aligned);
        }

        let ckey = make_chunk_key(binding.tensor_idx, entry);
        // Effective concurrent sharing over [k_begin, k_end): each resident CTA j with trip count
        // T_j overlaps on min(k_end, T_j) - k_begin steps. Reduces to exact integer R when all
        // resident CTAs have T_j >= k_end (e.g., uniform GEMM or non-causal attention).
        let sharing_count = self
            .active_chunks
            .get(&ckey)
            .map(|slot| {
                let overlap_sum: f64 = slot
                    .trips
                    .iter()
                    .map(|&t_j| {
                        let ov = k_end.min(t_j).saturating_sub(k_begin);
                        ov as f64 / steps as f64
                    })
                    .sum();
                overlap_sum.max(1.0)
            })
            .unwrap_or(1.0);

        // 1. Resident L2 hits (data already in L2 at request time -> L2 latency):
        //    from cross-wave reuse (scaled by prefix overlap with covered_steps) or warm L2.
        let mut resident_raw = 0.0f64;
        if self.options.cross_wave_reuse {
            if let Some(&last) = self.last_seen_stream_byte.get(&ckey) {
                let overlap = k_end.min(last.covered_steps).saturating_sub(k_begin);
                if overlap > 0 {
                    let inner_trips = self.ir.inner_trip_count(cta_id, outer_idx).unwrap_or(1);
                    let eff_steps = effective_chunk_steps(entry, inner_trips);
                    let full_stream = step_aligned.saturating_mul(eff_steps);
                    let reuse_dist = self
                        .stream_clock_bytes
                        .saturating_sub(last.stream_clock_bytes)
                        .max(full_stream);
                    let prefix_frac = overlap as f64 / steps as f64;
                    resident_raw =
                        resident_raw.max(self.eval_rescan_hit(reuse_dist as f64) * prefix_frac);
                }
            }
        }
        if self.initial_state == L2InitState::Warm {
            let warm_hit = if self.total_read_footprint_bytes == 0 {
                1.0
            } else {
                self.eval_rescan_hit(self.total_read_footprint_bytes as f64)
            };
            resident_raw = resident_raw.max(warm_hit);
        }

        // 2. Concurrent intra-wave miss merging (MSHR coalescing -> L2 BW, DRAM fill latency):
        //    the non-resident fraction (1 - resident_raw) is fetched from DRAM once across R
        //    concurrent CTAs, so (R - 1) / R of the miss fraction coalesces as merged bytes.
        let miss_raw = (1.0 - resident_raw).clamp(0.0, 1.0);
        let merged_raw = miss_raw * (sharing_count - 1.0) / sharing_count;

        let eff_cap = self.profile.l2_effective_capacity_bytes.max(1) as f64;
        let working_set = (self
            .active_working_set_bytes
            .saturating_add(self.spill_l2_pressure_bytes)) as f64;

        let (cap_factor, near_cliff) = match self.options.eviction_mode {
            EvictionMode::SoftLru => {
                let lo = 0.85 * eff_cap;
                let hi = 1.15 * eff_cap;
                if working_set <= lo {
                    (1.0, false)
                } else if working_set < hi {
                    let t = (working_set - lo) / (hi - lo);
                    let at_hi = (eff_cap / hi).min(1.0);
                    (1.0 - t * (1.0 - at_hi), true)
                } else {
                    ((eff_cap / working_set).min(1.0), false)
                }
            }
            EvictionMode::StrictLru => {
                if working_set <= eff_cap {
                    (1.0, false)
                } else {
                    ((eff_cap / working_set).min(1.0), false)
                }
            }
        };

        let resident_ratio = (resident_raw * cap_factor).clamp(0.0, 1.0);
        let merged_ratio = (merged_raw * cap_factor).clamp(0.0, 1.0 - resident_ratio);
        let hit_ratio = (resident_ratio + merged_ratio).clamp(0.0, 1.0);
        let resident_hit_bytes = total_f * resident_ratio;
        let merged_bytes = total_f * merged_ratio;
        let l2_bytes = resident_hit_bytes + merged_bytes;
        let dram_bytes = (total_f - l2_bytes).max(0.0);

        L2SplitResult {
            resident_hit_bytes,
            merged_bytes,
            l2_bytes,
            dram_bytes,
            hit_ratio,
            sector_aligned_bytes: total_aligned,
            near_cliff,
        }
    }

    pub fn account_spill(
        &mut self,
        _sm_id: u32,
        active_threads_in_sm: u32,
        stack_frame_bytes: u64,
        spill_loads_bytes: u64,
        spill_stores_bytes: u64,
    ) -> SpillTraffic {
        let per_thread_traffic = spill_loads_bytes.saturating_add(spill_stores_bytes);
        let total_traffic = per_thread_traffic.saturating_mul(active_threads_in_sm as u64);
        if total_traffic == 0 {
            return SpillTraffic {
                l1_bytes_per_iter: 0,
                l2_bytes_per_iter: 0,
                dram_bytes_per_iter: 0,
                overflow_ratio: 0.0,
            };
        }

        let total_local_footprint = stack_frame_bytes.saturating_mul(active_threads_in_sm as u64);
        let thresh = if self.profile.l1_to_l2_spill_threshold_bytes_per_sm > 0 {
            self.profile.l1_to_l2_spill_threshold_bytes_per_sm
        } else {
            self.profile.l1_size_bytes.max(1)
        };

        let overflow_ratio = if total_local_footprint <= thresh {
            0.0
        } else {
            ((total_local_footprint - thresh) as f64 / total_local_footprint as f64).clamp(0.0, 1.0)
        };

        let overflow_footprint = (total_local_footprint as f64 * overflow_ratio) as u64;
        self.spill_l2_pressure_bytes = self.spill_l2_pressure_bytes.max(overflow_footprint);

        let overflow_bytes = (total_traffic as f64 * overflow_ratio) as u64;
        let l1_bytes = total_traffic.saturating_sub(overflow_bytes);

        let eff_cap = self.profile.l2_effective_capacity_bytes.max(1) as f64;
        let working_set = (self
            .active_working_set_bytes
            .saturating_add(self.spill_l2_pressure_bytes)) as f64;
        let l2_fit = if working_set <= eff_cap {
            1.0
        } else {
            (eff_cap / working_set).min(1.0)
        };
        let l2_bytes = (overflow_bytes as f64 * l2_fit) as u64;
        let dram_bytes = overflow_bytes.saturating_sub(l2_bytes);

        SpillTraffic {
            l1_bytes_per_iter: l1_bytes,
            l2_bytes_per_iter: l2_bytes,
            dram_bytes_per_iter: dram_bytes,
            overflow_ratio,
        }
    }

    pub fn active_working_set_bytes(&self) -> u64 {
        self.active_working_set_bytes
    }

    pub fn total_read_footprint_bytes(&self) -> u64 {
        self.total_read_footprint_bytes
    }

    /// Evaluates L2 survival probability for a sequential scan of `reuse_distance_bytes`.
    /// The rescan curve's x-axis is absolute `working_set_bytes`; when `l2_effective_capacity_bytes`
    /// is perturbed relative to `nominal_l2_effective_capacity_bytes`, the curve x-axis scales
    /// proportionally so the cliff shifts with the perturbed capacity.
    fn eval_rescan_hit(&self, reuse_distance_bytes: f64) -> f64 {
        let eff_cap = self.profile.l2_effective_capacity_bytes.max(1) as f64;
        let query_bytes = reuse_distance_bytes.max(0.0);
        if !self.profile.rescan_hit_curve.is_empty() {
            let nom_cap = self.profile.nominal_l2_effective_capacity_bytes.max(1) as f64;
            let scale = eff_cap / nom_cap;
            let scaled_query = query_bytes / scale;
            interpolate(&self.profile.rescan_hit_curve, scaled_query).clamp(0.0, 1.0)
        } else {
            // Fallback when profile lacks an empirical rescan curve: strict capacity cliff.
            if query_bytes <= eff_cap {
                1.0
            } else {
                0.0
            }
        }
    }

    fn compute_total_read_footprint(&self) -> u64 {
        let mut max_per_chunk: HashMap<ChunkKey, u64> = HashMap::new();
        let total_ctas = self.ir.total_ctas() as u32;
        for cta_id in 0..total_ctas {
            let outer_trips = self.ir.outer_trip_count(cta_id).unwrap_or(1) as u32;
            for outer_idx in 0..outer_trips {
                let inner_trips = self.ir.inner_trip_count(cta_id, outer_idx).unwrap_or(1);
                for b in &self.load_bindings {
                    if let Some(entry) = self.lookup_strided(&b.access_ref, cta_id, outer_idx) {
                        let ckey = make_chunk_key(b.tensor_idx, entry);
                        let step_aligned = self.sector_aligned_step_bytes(entry);
                        let eff_steps = effective_chunk_steps(entry, inner_trips);
                        let full_stream = step_aligned.saturating_mul(eff_steps);
                        let slot = max_per_chunk.entry(ckey).or_insert(0);
                        *slot = (*slot).max(full_stream);
                    }
                }
            }
        }
        max_per_chunk
            .values()
            .fold(0u64, |acc, &v| acc.saturating_add(v))
    }

    fn lookup_strided(
        &self,
        access_ref: &str,
        cta_id: u32,
        outer_idx: u32,
    ) -> Option<Strided2dEntry> {
        let idx = self.ir.dense_tile_index(cta_id, outer_idx)?;
        match self.ir.access_map.get(access_ref)? {
            AccessEntry::Strided2dPerOuterTile { entries, .. } => entries.get(idx).copied(),
            AccessEntry::ExplicitBlocks { .. } => None,
        }
    }

    fn sector_aligned_step_bytes(&self, entry: Strided2dEntry) -> u64 {
        let sector = self.profile.l2_sector_bytes.max(1);
        let row_bytes = entry.row_bytes();
        if row_bytes == 0 || entry.rows() == 0 {
            return 0;
        }
        let aligned_row = row_bytes.div_ceil(sector).saturating_mul(sector);
        entry.rows().saturating_mul(aligned_row)
    }
}

fn effective_chunk_steps(entry: Strided2dEntry, inner_trips: u64) -> u64 {
    if entry.inner_step_byte_stride() == 0 {
        1
    } else {
        inner_trips
    }
}

fn make_chunk_key(tensor_idx: usize, e: Strided2dEntry) -> ChunkKey {
    ChunkKey {
        tensor_idx,
        base_byte_offset: e.base_byte_offset(),
        rows: e.rows(),
        row_bytes: e.row_bytes(),
        row_stride_bytes: e.row_stride_bytes(),
        inner_step_byte_stride: e.inner_step_byte_stride(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::TripCounts;

    /// Deterministic synthetic L2 profile for mechanism tests (never depends on re-measured files).
    fn synthetic_l2_profile() -> L2Profile {
        let mib = 1024.0 * 1024.0;
        L2Profile {
            l2_size_bytes: 36 * 1024 * 1024,
            l2_effective_capacity_bytes: 36 * 1024 * 1024,
            nominal_l2_effective_capacity_bytes: 36 * 1024 * 1024,
            l2_line_bytes: 128,
            l2_sector_bytes: 32,
            l1_size_bytes: 128 * 1024,
            l1_to_l2_spill_threshold_bytes_per_sm: 64 * 1024,
            rescan_hit_curve: vec![
                (18.0 * mib, 1.0),
                (36.0 * mib, 1.0),
                (39.6 * mib, 0.5),
                (46.8 * mib, 0.0),
            ],
        }
    }

    #[test]
    fn swizzled_resident_ctas_share_a_and_b_in_l2() {
        let json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir = KernelIR::from_json_str(json).unwrap();
        let mut l2 = L2Model::new(
            &ir,
            synthetic_l2_profile(),
            L2InitState::Cold,
            L2Options::default(),
        );

        // First 16 CTAs in GROUP_SIZE_M=8 cover pid_m in 0..8, pid_n in 0..2:
        // each A row-slice is shared by 2 CTAs; each B col-slice is shared by 8 CTAs.
        for cta in 0..16u32 {
            l2.begin_tile(cta % 46, cta, 0, 0.0);
        }

        let res_a = l2.split(0, 0, 0, "load_A", 0, 128, false, CachePolicy::Default, 0.0);
        let res_b = l2.split(0, 0, 0, "load_B", 0, 128, false, CachePolicy::Default, 0.0);

        assert!(
            (res_a.hit_ratio - 0.5).abs() < 1e-6,
            "A shared by 2 CTAs -> 50% hit"
        );
        assert!(
            (res_b.hit_ratio - 0.875).abs() < 1e-6,
            "B shared by 8 CTAs -> 87.5% hit"
        );
        assert_eq!(res_a.sector_aligned_bytes, 128 * 64 * 128);
        assert!(
            (res_a.l2_bytes + res_a.dram_bytes - res_a.sector_aligned_bytes as f64).abs() < 1e-3
        );
        assert!(!res_a.near_cliff);

        // End wave 1 (8 A stripes + 2 B stripes = 10 MiB <= 36 MiB L2) -> rescan curve gives 1.0.
        for cta in 0..16u32 {
            l2.end_tile(cta % 46, cta, 0, 1000.0);
        }
        l2.begin_tile(0, 16, 0, 1000.0);
        let res_a_next = l2.split(
            0,
            16,
            0,
            "load_A",
            0,
            128,
            false,
            CachePolicy::Default,
            1000.0,
        );
        assert!(
            (res_a_next.hit_ratio - 1.0).abs() < 1e-6,
            "10 MiB reuse distance fits in 36 MiB L2"
        );
    }

    #[test]
    fn cross_wave_history_accounts_for_full_k_stream_and_evicts_on_4096_gemm() {
        let json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir = KernelIR::from_json_str(json).unwrap();
        let prof = synthetic_l2_profile();
        let mut l2 = L2Model::new(&ir, prof, L2InitState::Cold, L2Options::default());

        // 4096^3 f16 GEMM has 32 MiB A + 32 MiB B = 64 MiB total read footprint.
        assert_eq!(l2.total_read_footprint_bytes(), 64 * 1024 * 1024);

        // Run all 1024 CTAs in 92-CTA waves.
        let wave_size = 92u32;
        let mut total_b_hit_wave2_plus = 0.0;
        let mut count_b_wave2_plus = 0u32;

        for wave_start in (0..1024u32).step_by(wave_size as usize) {
            let wave_end = (wave_start + wave_size).min(1024);
            for cta in wave_start..wave_end {
                l2.begin_tile(cta % 46, cta, 0, 0.0);
            }
            if wave_start >= 256 {
                let res_b = l2.split(
                    0,
                    wave_start,
                    0,
                    "load_B",
                    0,
                    128,
                    false,
                    CachePolicy::Default,
                    0.0,
                );
                total_b_hit_wave2_plus += res_b.hit_ratio;
                count_b_wave2_plus += 1;
            }
            for cta in wave_start..wave_end {
                l2.end_tile(cta % 46, cta, 0, 1000.0);
            }
        }

        // After streaming 64 MiB (> 45 MiB cliff on RTX 4070), a solo CTA reusing an early B column
        // must see near-zero cross-wave hit (not 1.0 and not C_eff/F).
        l2.begin_tile(0, 256, 0, 2000.0);
        let solo_b = l2.split(
            0,
            256,
            0,
            "load_B",
            0,
            128,
            false,
            CachePolicy::Default,
            2000.0,
        );
        assert!(
            solo_b.hit_ratio < 0.05,
            "evicted B stripe after 64 MiB stream must miss, got {}",
            solo_b.hit_ratio
        );
        let avg_b_hit = total_b_hit_wave2_plus / count_b_wave2_plus as f64;
        assert!(
            avg_b_hit < 0.95,
            "cross-wave B hit on 64 MiB GEMM in 36 MiB L2 must be well below 1.0, got {}",
            avg_b_hit
        );
    }

    #[test]
    fn warm_state_follows_empirical_rescan_curve() {
        let json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir_large = KernelIR::from_json_str(json).unwrap();
        let prof = synthetic_l2_profile();

        // 1. Large 4096^3 GEMM (64 MiB > 46.8 MiB cliff): warm hit must be 0.0.
        let mut l2_large = L2Model::new(
            &ir_large,
            prof.clone(),
            L2InitState::Warm,
            L2Options::default(),
        );
        l2_large.begin_tile(0, 0, 0, 0.0);
        let large_res = l2_large.split(0, 0, 0, "load_A", 0, 128, false, CachePolicy::Default, 0.0);
        assert!(
            large_res.hit_ratio.abs() < 1e-6,
            "64 MiB warm scan exceeds 46.8 MiB cliff; expected 0.0, got {}",
            large_res.hit_ratio
        );

        // 2. Small footprint (K=8 -> 4 MiB <= 36 MiB L2): warm hit must be 1.0.
        let mut ir_small = ir_large.clone();
        ir_small.body.pipeline_loop.trip_counts = TripCounts::Uniform(8);
        let mut l2_small = L2Model::new(
            &ir_small,
            prof.clone(),
            L2InitState::Warm,
            L2Options::default(),
        );
        assert_eq!(l2_small.total_read_footprint_bytes(), 4 * 1024 * 1024);
        l2_small.begin_tile(0, 0, 0, 0.0);
        let small_res = l2_small.split(0, 0, 0, "load_A", 0, 8, false, CachePolicy::Default, 0.0);
        assert!(
            (small_res.hit_ratio - 1.0).abs() < 1e-6,
            "4 MiB footprint in 36 MiB warm L2 should hit 1.0, got {}",
            small_res.hit_ratio
        );

        // 3. Transition knee (K=78 -> 39 MiB, between 36 MiB -> 1.0 and 39.6 MiB -> 0.5):
        //    linear interpolation gives 1.0 - (3.0 / 3.6) * 0.5 = 7/12 ~= 0.583333.
        let mut ir_knee = ir_large.clone();
        ir_knee.body.pipeline_loop.trip_counts = TripCounts::Uniform(78);
        let mut l2_knee = L2Model::new(&ir_knee, prof, L2InitState::Warm, L2Options::default());
        l2_knee.begin_tile(0, 0, 0, 0.0);
        let knee_res = l2_knee.split(0, 0, 0, "load_A", 0, 78, false, CachePolicy::Default, 0.0);
        let expected_knee = 1.0 - (3.0 / 3.6) * 0.5;
        assert!(
            (knee_res.hit_ratio - expected_knee).abs() < 1e-6,
            "39 MiB warm scan should interpolate to {}, got {}",
            expected_knee,
            knee_res.hit_ratio
        );
    }

    #[test]
    fn soft_lru_flags_near_cliff_and_degrades_above_capacity() {
        let json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir = KernelIR::from_json_str(json).unwrap();

        // 16 CTAs hold 8 unique A tiles (8*8KB = 64KB) + 2 unique B tiles (2*8KB = 16KB) = 80KB.
        let mut prof = synthetic_l2_profile();
        prof.l2_effective_capacity_bytes = 80 * 1024; // right at the cliff (1.0 * C_eff)

        let mut l2 = L2Model::new(&ir, prof, L2InitState::Cold, L2Options::default());
        for cta in 0..16u32 {
            l2.begin_tile(cta, cta, 0, 0.0);
        }
        assert_eq!(l2.active_working_set_bytes(), 80 * 1024);

        let res = l2.split(0, 0, 0, "load_B", 0, 16, false, CachePolicy::Default, 0.0);
        assert!(
            res.near_cliff,
            "working set == C_eff should flag near_cliff"
        );
        assert!(res.hit_ratio < 0.875 && res.hit_ratio > 0.7);
    }

    #[test]
    fn spill_overflow_transitions_from_l1_to_l2_dram() {
        let json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir = KernelIR::from_json_str(json).unwrap();
        let mut l2 = L2Model::new(
            &ir,
            synthetic_l2_profile(),
            L2InitState::Cold,
            L2Options::default(),
        );

        // 256 active threads * 128B stack = 32KB <= 64KB threshold -> fits in L1.
        let small = l2.account_spill(0, 256, 128, 32, 32);
        assert_eq!(small.overflow_ratio, 0.0);
        assert_eq!(small.l2_bytes_per_iter, 0);
        assert_eq!(small.dram_bytes_per_iter, 0);

        // 256 active threads * 512B stack = 128KB > 64KB -> 50% overflows to L2.
        let large = l2.account_spill(0, 256, 512, 64, 64);
        assert!((large.overflow_ratio - 0.5).abs() < 1e-6);
        assert!(large.l2_bytes_per_iter > 0);
    }

    #[test]
    fn construct_l2_profile_from_rtx4070_json() {
        let profile_json = include_str!("../../../../profiles/rtx4070.json");
        let gpu_profile = Profile::from_json(profile_json)
            .expect("profiles/rtx4070.json should deserialize into Profile");
        let prof = L2Profile::from(&gpu_profile);

        let raw: serde_json::Value = serde_json::from_str(profile_json).unwrap();
        let measured = |path: &str| raw.pointer(path).and_then(|v| v.as_f64()).unwrap() as u64;
        let capacity = measured("/measured/memory/l2_effective_capacity_bytes/value");

        assert_eq!(prof.l2_size_bytes, 37_748_736);
        assert_eq!(prof.l2_effective_capacity_bytes, capacity);
        assert_eq!(prof.l2_line_bytes, 128);
        assert_eq!(prof.l2_sector_bytes, 32);
        assert_eq!(prof.l1_size_bytes, 131_072);
        assert_eq!(
            prof.l1_to_l2_spill_threshold_bytes_per_sm,
            measured("/measured/spill/l1_to_l2_spill_threshold_bytes_per_sm/value")
        );
        assert!(!prof.rescan_hit_curve.is_empty());
        for w in prof.rescan_hit_curve.windows(2) {
            assert!(w[0].0 <= w[1].0, "rescan curve x-axis must be sorted");
            assert!(
                w[0].1 >= w[1].1,
                "rescan curve must be monotone non-increasing: {} < {}",
                w[0].1,
                w[1].1
            );
        }
        for &(_, h) in &prof.rescan_hit_curve {
            assert!((0.0..=1.0).contains(&h), "hit rate {} out of [0, 1]", h);
        }

        // Verify L2Model consumes the profile's empirical rescan curve in absolute bytes.
        let ir_json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir = KernelIR::from_json_str(ir_json).unwrap();
        let mut l2 = L2Model::new(&ir, prof.clone(), L2InitState::Warm, L2Options::default());
        l2.begin_tile(0, 0, 0, 0.0);
        let res = l2.split(0, 0, 0, "load_A", 0, 128, false, CachePolicy::Default, 0.0);
        assert!((0.0..=1.0).contains(&res.hit_ratio));
        let expected = interpolate(&prof.rescan_hit_curve, 64.0 * 1024.0 * 1024.0);
        assert!(
            (res.hit_ratio - expected).abs() < 1e-6,
            "expected warm hit from rtx4070.json rescan curve ({}), got {}",
            expected,
            res.hit_ratio
        );
    }

    #[test]
    fn rescan_curve_uses_absolute_bytes_when_spec_and_effective_l2_differ() {
        let ir_json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let mut ir_39m = KernelIR::from_json_str(ir_json).unwrap();
        // K=78 -> 39 MiB total read footprint.
        ir_39m.body.pipeline_loop.trip_counts = TripCounts::Uniform(78);

        let mib = 1024.0 * 1024.0;
        // Spec L2 is 64 MiB, while effective L2 capacity is only 36 MiB (distinct from spec).
        let base_prof = L2Profile {
            l2_size_bytes: 64 * 1024 * 1024,
            l2_effective_capacity_bytes: 36 * 1024 * 1024,
            nominal_l2_effective_capacity_bytes: 36 * 1024 * 1024,
            l2_line_bytes: 128,
            l2_sector_bytes: 32,
            l1_size_bytes: 128 * 1024,
            l1_to_l2_spill_threshold_bytes_per_sm: 64 * 1024,
            rescan_hit_curve: vec![(36.0 * mib, 1.0), (39.0 * mib, 0.326), (45.0 * mib, 0.0)],
        };

        // 1. Unperturbed: 39 MiB query must hit the 39 MiB cliff point (0.326), NOT shifted by 64 MiB spec.
        let mut l2_nom = L2Model::new(
            &ir_39m,
            base_prof.clone(),
            L2InitState::Warm,
            L2Options::default(),
        );
        l2_nom.begin_tile(0, 0, 0, 0.0);
        let r_nom = l2_nom.split(0, 0, 0, "load_A", 0, 78, false, CachePolicy::Default, 0.0);
        assert!(
            (r_nom.hit_ratio - 0.326).abs() < 1e-6,
            "unperturbed 39 MiB scan should hit 0.326 regardless of 64 MiB spec L2, got {}",
            r_nom.hit_ratio
        );

        // 2. Perturbed: halve effective L2 capacity (36 MiB -> 18 MiB); cliff shifts 39 MiB -> 19.5 MiB.
        let mut ir_19_5m = KernelIR::from_json_str(ir_json).unwrap();
        ir_19_5m.body.pipeline_loop.trip_counts = TripCounts::Uniform(39); // 19.5 MiB footprint
        let mut perturbed = base_prof;
        perturbed.l2_effective_capacity_bytes = 18 * 1024 * 1024;
        let mut l2_pert = L2Model::new(
            &ir_19_5m,
            perturbed,
            L2InitState::Warm,
            L2Options::default(),
        );
        l2_pert.begin_tile(0, 0, 0, 0.0);
        let r_pert = l2_pert.split(0, 0, 0, "load_A", 0, 39, false, CachePolicy::Default, 0.0);
        assert!(
            (r_pert.hit_ratio - 0.326).abs() < 1e-6,
            "halving effective L2 must shift 39 MiB cliff to 19.5 MiB (0.326), got {}",
            r_pert.hit_ratio
        );
    }

    #[test]
    fn split_distinguishes_resident_hits_from_mshr_merged_bytes() {
        let ir_json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir = KernelIR::from_json_str(ir_json).unwrap();
        let prof = synthetic_l2_profile();

        // 1. Cold first wave (16 CTAs: A shared by 2, B shared by 8):
        //    nothing is resident yet, so all L2 savings are MSHR merged_bytes.
        let mut l2_cold = L2Model::new(&ir, prof.clone(), L2InitState::Cold, L2Options::default());
        for cta in 0..16u32 {
            l2_cold.begin_tile(cta % 46, cta, 0, 0.0);
        }
        let cold_b = l2_cold.split(0, 0, 0, "load_B", 0, 128, false, CachePolicy::Default, 0.0);
        let total = cold_b.sector_aligned_bytes as f64;
        assert!(
            cold_b.resident_hit_bytes.abs() < 1e-6,
            "cold wave 1 has zero resident hits"
        );
        assert!((cold_b.merged_bytes - total * 0.875).abs() < 1e-3);
        assert!((cold_b.dram_bytes - total * 0.125).abs() < 1e-3);
        assert!((cold_b.l2_bytes - (cold_b.resident_hit_bytes + cold_b.merged_bytes)).abs() < 1e-6);

        // 2. Cross-wave reuse after 10 MiB wave (fits in 36 MiB L2):
        //    A row is now resident in L2 -> all bytes are resident_hit_bytes, zero merged/dram.
        for cta in 0..16u32 {
            l2_cold.end_tile(cta % 46, cta, 0, 1000.0);
        }
        l2_cold.begin_tile(0, 16, 0, 1000.0);
        let xwave_a = l2_cold.split(
            0,
            16,
            0,
            "load_A",
            0,
            128,
            false,
            CachePolicy::Default,
            1000.0,
        );
        let total_a = xwave_a.sector_aligned_bytes as f64;
        assert!((xwave_a.resident_hit_bytes - total_a).abs() < 1e-3);
        assert!(xwave_a.merged_bytes.abs() < 1e-6);
        assert!(xwave_a.dram_bytes.abs() < 1e-6);

        // 3. Partial warm residency + concurrent sharing:
        //    synthetic profile gives 0.5 resident hit at 39.6 MiB. With K=79 (39.5 MiB) or
        //    exact 39.6 MiB curve point, test exact 50% resident + 8-way B sharing ->
        //    50% resident, 50% * 7/8 = 43.75% merged, 50% * 1/8 = 6.25% dram.
        let mut prof_half = prof;
        let mib = 1024.0 * 1024.0;
        prof_half.rescan_hit_curve = vec![(32.0 * mib, 0.5), (128.0 * mib, 0.5)];
        let mut l2_partial = L2Model::new(&ir, prof_half, L2InitState::Warm, L2Options::default());
        for cta in 0..16u32 {
            l2_partial.begin_tile(cta % 46, cta, 0, 0.0);
        }
        let pb = l2_partial.split(0, 0, 0, "load_B", 0, 128, false, CachePolicy::Default, 0.0);
        let ptot = pb.sector_aligned_bytes as f64;
        assert!((pb.resident_hit_bytes - ptot * 0.5).abs() < 1e-3);
        assert!((pb.merged_bytes - ptot * 0.4375).abs() < 1e-3);
        assert!((pb.dram_bytes - ptot * 0.0625).abs() < 1e-3);
        assert!((pb.l2_bytes - (pb.resident_hit_bytes + pb.merged_bytes)).abs() < 1e-6);
        assert!((pb.l2_bytes + pb.dram_bytes - ptot).abs() < 1e-3);
    }

    #[test]
    fn causal_gqa_attention_footprint_and_kv_sharing() {
        let ir_json =
            include_str!("../../../../schemas/examples/flash_attention_causal_gqa_sm89.json");
        let ir = KernelIR::from_json_str(ir_json).unwrap();
        let prof = synthetic_l2_profile();
        let mut l2 = L2Model::new(&ir, prof, L2InitState::Cold, L2Options::default());

        // Total read footprint: Q (8 * 128 KiB = 1 MiB) + K (2 * 128 KiB) + V (2 * 128 KiB) = 1.5 MiB.
        // Must take max prefix across causal CTAs rather than cta_id=0's 1-step prefix.
        assert_eq!(l2.total_read_footprint_bytes(), 1536 * 1024);

        // Concurrent CTAs sharing KV head 0:
        // - cta 15: h_q=0, qb=15 (trips=16)
        // - cta 31: h_q=1, qb=15 (trips=16, GQA partner sharing h_kv=0)
        // - cta  7: h_q=0, qb=7  (trips=8, same Q head, shorter causal prefix)
        // Plus cta 79: h_q=4, qb=15 (h_kv=1, different KV head -> must NOT share with h_kv=0).
        l2.begin_tile(0, 15, 0, 0.0);
        l2.begin_tile(1, 31, 0, 0.0);
        l2.begin_tile(2, 7, 0, 0.0);
        l2.begin_tile(3, 79, 0, 0.0);

        // Over [0, 16), cta 15 sees effective_R = 1.0 + 1.0 + 8/16 = 2.5 -> merged = 1.5/2.5 = 0.6.
        let res_long = l2.split(0, 15, 0, "load_K", 0, 16, false, CachePolicy::Default, 0.0);
        assert!(
            (res_long.hit_ratio - 0.6).abs() < 1e-6,
            "expected 0.6, got {}",
            res_long.hit_ratio
        );
        assert!(res_long.resident_hit_bytes.abs() < 1e-6);

        // Over [0, 8), all three h_kv=0 CTAs fully overlap -> R = 3.0 -> merged = 2/3.
        let res_short = l2.split(2, 7, 0, "load_K", 0, 8, false, CachePolicy::Default, 0.0);
        assert!(
            (res_short.hit_ratio - (2.0 / 3.0)).abs() < 1e-6,
            "expected 2/3, got {}",
            res_short.hit_ratio
        );

        // cta 79 belongs to h_kv=1 alone -> zero sharing.
        let res_other_kv = l2.split(3, 79, 0, "load_K", 0, 16, false, CachePolicy::Default, 0.0);
        assert!(res_other_kv.hit_ratio.abs() < 1e-6);
    }

    #[test]
    fn causal_cross_wave_short_cta_only_warms_its_prefix() {
        let ir_json =
            include_str!("../../../../schemas/examples/flash_attention_causal_gqa_sm89.json");
        let ir = KernelIR::from_json_str(ir_json).unwrap();
        let prof = synthetic_l2_profile();
        let mut l2 = L2Model::new(&ir, prof, L2InitState::Cold, L2Options::default());

        // Wave 1: solo short CTA (cta=1, h_q=0, qb=1, trips=2) streams only KV steps 0..2.
        l2.begin_tile(0, 1, 0, 0.0);
        l2.end_tile(0, 1, 0, 100.0);

        // Wave 2: solo long CTA (cta=15, h_q=0, qb=15, trips=16) on same h_kv=0.
        l2.begin_tile(0, 15, 0, 100.0);
        // Full [0, 16) only overlaps on steps 0..2 -> resident hit ratio = 2/16 = 0.125.
        let full = l2.split(
            0,
            15,
            0,
            "load_K",
            0,
            16,
            false,
            CachePolicy::Default,
            100.0,
        );
        assert!(
            (full.hit_ratio - 0.125).abs() < 1e-6,
            "short 2-step CTA must only warm 2/16 of a 16-step CTA, got {}",
            full.hit_ratio
        );
        assert!((full.resident_hit_bytes - full.l2_bytes).abs() < 1e-6);

        // Prefix [0, 2) is 100% resident; tail [2, 16) is 0% resident.
        let prefix = l2.split(0, 15, 0, "load_K", 0, 2, false, CachePolicy::Default, 100.0);
        let tail = l2.split(
            0,
            15,
            0,
            "load_K",
            2,
            16,
            false,
            CachePolicy::Default,
            100.0,
        );
        assert!((prefix.hit_ratio - 1.0).abs() < 1e-6);
        assert!(tail.hit_ratio.abs() < 1e-6);
    }
}
