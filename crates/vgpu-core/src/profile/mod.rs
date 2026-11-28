//! GPU profile, schema 0.1 (see `docs/components/profiles.md`).
//!
//! Only the fields the core reads are typed; unknown fields are ignored so the
//! profile format can grow without breaking the core.

use serde::Deserialize;
use std::collections::BTreeMap;

/// A measured quantity with its uncertainty.
#[derive(Debug, Clone, Deserialize)]
pub struct MeasuredValue {
    pub value: f64,
    #[serde(default)]
    pub ci95_low: Option<f64>,
    #[serde(default)]
    pub ci95_high: Option<f64>,
    pub rel_uncertainty: f64,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub perturb_group: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Profile {
    pub schema_version: String,
    pub identity: Identity,
    #[serde(default)]
    pub provenance: serde_json::Value,
    pub specs: Specs,
    pub measured: Measured,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Identity {
    pub vendor: String,
    pub arch: String,
    pub sku: String,
}

/// Optional Huawei Ascend Da Vinci on-chip SRAM and Cube/Vector core limits.
#[derive(Debug, Clone, Deserialize)]
pub struct AscendSpecs {
    pub cube_cores: u32,
    pub vector_cores: u32,
    pub l1_bytes: u64,
    pub l0a_bytes: u64,
    pub l0b_bytes: u64,
    pub l0c_bytes: u64,
    pub ub_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Specs {
    pub sm_count: u32,
    pub warp_size: u32,
    pub max_threads_per_sm: u32,
    pub max_warps_per_sm: u32,
    pub max_blocks_per_sm: u32,
    pub max_threads_per_block: u32,
    pub registers_per_sm: u32,
    pub max_registers_per_thread: u32,
    pub register_alloc_unit_regs: u32,
    pub shared_mem_per_sm_bytes: u64,
    pub max_shared_mem_per_block_bytes: u64,
    #[serde(default)]
    pub reserved_shared_mem_per_cta_bytes: u64,
    pub shared_mem_alloc_granularity_bytes: u64,
    #[serde(default)]
    pub shared_mem_carveout_options_bytes: Vec<u64>,
    pub l2_size_bytes: u64,
    pub l2_line_bytes: u64,
    pub l2_sector_bytes: u64,
    pub l1_size_bytes: u64,
    #[serde(default)]
    pub capabilities: BTreeMap<String, bool>,
    #[serde(default)]
    pub ascend: Option<AscendSpecs>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Measured {
    pub memory: Memory,
    pub compute: Compute,
    #[serde(default)]
    pub async_pipeline: BTreeMap<String, Option<MeasuredValue>>,
    #[serde(default)]
    pub spill: BTreeMap<String, Option<MeasuredValue>>,
    pub overhead: Overhead,
    pub cta_dispatch: CtaDispatch,
    #[serde(default)]
    pub pipeline: Pipeline,
}

/// Software pipeline costs.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Pipeline {
    /// Per-stage commit, wait, and barrier cost with no data moved.
    #[serde(default)]
    pub fixed_sync_ns: Vec<SyncPoint>,
    /// Per-handoff cross-core Cube/Vector sync latency by active AI Cores.
    #[serde(default)]
    pub cv_sync_hop_ns_by_active_sms: Vec<ActiveSmLatencyPoint>,
    /// Cost of one more barrier per step on an empty step, by warp count.
    #[serde(default)]
    pub barrier_step_ns: Vec<BarrierStepPoint>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BarrierStepPoint {
    pub num_warps: u32,
    pub overhead_ns: MeasuredValue,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ActiveSmLatencyPoint {
    pub active_sms: u32,
    pub latency_ns: MeasuredValue,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SyncPoint {
    pub num_warps: u32,
    /// Stage buffers in the loop.
    pub tokens: u32,
    pub overhead_ns: MeasuredValue,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BwPoint {
    pub active_sms: u32,
    pub bw_gbps: MeasuredValue,
}

/// Delivered bandwidth with a given number of load bytes in flight.
#[derive(Debug, Clone, Deserialize)]
pub struct MlpPoint {
    /// `cp_async` or `ld_global_v4`.
    pub issue_mode: String,
    pub bytes_in_flight: u64,
    pub active_ctas: u32,
    pub bytes_per_cta: u64,
    pub bw_gbps: MeasuredValue,
}

#[derive(Debug, Clone, Deserialize)]
pub struct L2RescanPoint {
    pub working_set_bytes: u64,
    pub working_set_ratio_to_l2: f64,
    pub pass2_bw_gbps: MeasuredValue,
    pub estimated_hit_rate: MeasuredValue,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Memory {
    /// Global load warp instructions per SM with a known number in flight.
    #[serde(default)]
    pub gmem_load_issue_controlled: Vec<GmemIssuePoint>,
    pub dram_bw_by_active_sms: Vec<BwPoint>,
    pub l2_bw_by_active_sms: Vec<BwPoint>,
    #[serde(default)]
    pub l2_bw_by_active_sms_cp_async: Vec<BwPoint>,
    pub l2_effective_capacity_bytes: MeasuredValue,
    #[serde(default)]
    pub l2_rescan_hit_rate_by_working_set: Vec<L2RescanPoint>,
    pub smem_bw_per_sm_gbps: MeasuredValue,
    pub dram_latency_ns: MeasuredValue,
    pub l2_latency_ns: MeasuredValue,
    /// Memory-level parallelism: delivered bandwidth against bytes in flight.
    #[serde(default)]
    pub dram_bw_by_bytes_in_flight: Vec<MlpPoint>,
    #[serde(default)]
    pub l2_bw_by_bytes_in_flight: Vec<MlpPoint>,
    #[serde(default)]
    pub dram_latency_loaded_ns: Option<MeasuredValue>,
    #[serde(default)]
    pub l2_latency_loaded_ns: Option<MeasuredValue>,
    #[serde(default)]
    pub spill_reload_latency_by_count: Vec<SpillReloadPoint>,
    #[serde(default)]
    pub l1_to_l2_spill_threshold_by_carveout: Vec<L1CarveoutPoint>,
}

/// Extra time per loop iteration from `reload_count` local memory reloads.
#[derive(Debug, Clone, Deserialize)]
pub struct SpillReloadPoint {
    /// `l1_hit`, `l1_boundary` or `l2_spill`.
    pub regime: String,
    /// `serial_chain` or `batch_before_dep`.
    pub dependency_mode: String,
    #[serde(default)]
    pub resident_threads_per_sm: Option<u64>,
    #[serde(default)]
    pub frame_bytes_per_thread: Option<u64>,
    pub reload_count: u32,
    pub extra_ns_vs_zero: MeasuredValue,
}

/// L1 miss rate of local memory by working set at one shared memory carveout.
#[derive(Debug, Clone, Deserialize)]
pub struct L1CarveoutPoint {
    pub smem_carveout_bytes: u64,
    #[serde(default)]
    pub miss_rate_by_working_set: Vec<L1MissPoint>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct L1MissPoint {
    pub working_set_bytes: u64,
    pub estimated_miss_rate: MeasuredValue,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MmaPoint {
    pub mma_warps_per_sm: u32,
    pub tflops_per_sm: MeasuredValue,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Compute {
    /// Keyed by `<op_class>:<a_dtype>:<b_dtype>:<acc_dtype>:<M>x<N>x<K>`.
    pub mma_tflops_per_sm_by_warps: BTreeMap<String, Vec<MmaPoint>>,
    pub non_mma_issue_rate_per_sm_gips: MeasuredValue,
    /// Special-function lane operations per second for the whole GPU.
    #[serde(default)]
    pub sfu_tflops: Option<MeasuredValue>,
    /// `ex2.approx` and butterfly `shfl.sync` warp instructions per second
    /// per SM, by the warps issuing them.
    #[serde(default)]
    pub sfu_issue_rate_per_sm_gips: Vec<WarpRate>,
    #[serde(default)]
    pub shfl_issue_rate_per_sm_gips: Vec<WarpRate>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GmemIssuePoint {
    pub load_width_bytes: u32,
    pub issue_rate_gwarp_inst_per_sm: MeasuredValue,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WarpRate {
    pub warps_per_sm: u32,
    pub rate: MeasuredValue,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Overhead {
    pub launch_overhead_us: MeasuredValue,
    pub per_cta_launch_latency_ns: MeasuredValue,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CtaDispatch {
    pub observed_policy: String,
    #[serde(default)]
    pub candidate_policies: Vec<String>,
    #[serde(default)]
    pub confidence: Option<String>,
}

impl Profile {
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
}

/// Piecewise-linear interpolation over sorted `(x, y)` points, clamped at both ends.
pub fn interpolate(points: &[(f64, f64)], x: f64) -> f64 {
    let (first, last) = (points[0], points[points.len() - 1]);
    if x <= first.0 {
        return first.1;
    }
    if x >= last.0 {
        return last.1;
    }
    let i = points.partition_point(|p| p.0 <= x);
    let ((x0, y0), (x1, y1)) = (points[i - 1], points[i]);
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolate_clamps_and_interpolates() {
        let p = [(1.0, 10.0), (3.0, 30.0)];
        assert_eq!(interpolate(&p, 0.0), 10.0);
        assert_eq!(interpolate(&p, 2.0), 20.0);
        assert_eq!(interpolate(&p, 9.0), 30.0);
    }
}
