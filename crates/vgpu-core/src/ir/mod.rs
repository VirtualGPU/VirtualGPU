//! Language- and vendor-agnostic Kernel IR (v0.1).

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fmt;

pub const SCHEMA_VERSION: &str = "0.1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelIR {
    pub schema_version: String,
    pub metadata: Metadata,
    pub launch: LaunchConfig,
    pub resources: ResourceUsage,
    pub tensors: Vec<TensorDesc>,
    pub body: KernelBody,
    pub access_map: BTreeMap<String, AccessEntry>,
    pub diagnostics: Vec<String>,
    pub unsupported_features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub kernel_name: String,
    pub target_vendor: String,
    pub target_arch: String,
    pub frontend: String,
    #[serde(default)]
    pub toolchain: Option<serde_json::Value>,
    #[serde(default)]
    pub config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
    pub cluster_shape: [u32; 3],
    pub num_warps: u32,
    pub warp_size: u32,
    pub warp_groups: Vec<WarpGroup>,
    pub dynamic_smem_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarpGroup {
    pub id: u32,
    pub role: WarpGroupRole,
    pub num_warps: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpGroupRole {
    Unified,
    Producer,
    Consumer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceUsage {
    pub source: ResourceSource,
    pub registers_per_thread: u32,
    pub static_smem_bytes: u64,
    pub stack_frame_bytes: u64,
    pub spill_stores_bytes: u64,
    pub spill_loads_bytes: u64,
    pub spill_scope: SpillScope,
    pub barriers: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceSource {
    AssemblerReport,
    Estimated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpillScope {
    None,
    PrologueEpilogue,
    InnerLoopConfirmed,
    InnerLoopAssumed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorDesc {
    pub id: String,
    pub dtype: String,
    pub element_bits: u32,
    pub shape: Vec<u64>,
    pub strides: Vec<u64>,
    pub base_addr: u64,
    pub cache_policy: CachePolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePolicy {
    Default,
    BypassL1,
    Streaming,
    EvictFirst,
    Volatile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelBody {
    pub outer_loop: OuterLoop,
    pub prologue: OpBlock,
    pub pipeline_loop: PipelineLoop,
    pub epilogue: Epilogue,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuterLoop {
    pub trip_count_mode: OuterTripCountMode,
    pub trip_counts: TripCounts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OuterTripCountMode {
    Uniform,
    PerCta,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpBlock {
    pub ops: Vec<Op>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineLoop {
    pub trip_count_mode: InnerTripCountMode,
    pub trip_counts: TripCounts,
    pub num_stages: u32,
    pub prefetch_distance: u32,
    pub async_copy: AsyncCopyKind,
    pub ops: Vec<Op>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InnerTripCountMode {
    Uniform,
    PerCta,
    PerOuterIter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsyncCopyKind {
    None,
    AsyncElement,
    AsyncTensor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TripCounts {
    Uniform(u64),
    PerItem(Vec<u64>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Epilogue {
    pub overlaps_next_tile: bool,
    pub ops: Vec<Op>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Op {
    GmemLoad(GmemOp),
    GmemStore(GmemOp),
    SmemLoad(SmemOp),
    SmemStore(SmemOp),
    Mma(MmaOp),
    Alu(ScalarOp),
    Sfu(ScalarOp),
    Shfl(ScalarOp),
    Barrier(BarrierOp),
}

impl Op {
    pub fn op_id(&self) -> &str {
        match self {
            Op::GmemLoad(o) | Op::GmemStore(o) => &o.op_id,
            Op::SmemLoad(o) | Op::SmemStore(o) => &o.op_id,
            Op::Mma(o) => &o.op_id,
            Op::Alu(o) | Op::Sfu(o) | Op::Shfl(o) => &o.op_id,
            Op::Barrier(o) => &o.op_id,
        }
    }

    pub fn warp_group_id(&self) -> u32 {
        match self {
            Op::GmemLoad(o) | Op::GmemStore(o) => o.warp_group_id,
            Op::SmemLoad(o) | Op::SmemStore(o) => o.warp_group_id,
            Op::Mma(o) => o.warp_group_id,
            Op::Alu(o) | Op::Sfu(o) | Op::Shfl(o) => o.warp_group_id,
            Op::Barrier(o) => o.warp_group_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GmemOp {
    pub op_id: String,
    pub warp_group_id: u32,
    pub stage_offset: u32,
    pub tensor_id: String,
    pub nominal_bytes_per_cta: u64,
    pub transfer_width_bytes: u32,
    pub access_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmemOp {
    pub op_id: String,
    pub warp_group_id: u32,
    pub stage_offset: u32,
    pub bytes_per_cta: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MmaOp {
    pub op_id: String,
    pub warp_group_id: u32,
    pub stage_offset: u32,
    pub op_class: String,
    pub tile_mnk: [u32; 3],
    pub inst_mnk: [u32; 3],
    pub a_dtype: String,
    pub b_dtype: String,
    pub acc_dtype: String,
    pub count: u64,
}

impl MmaOp {
    /// Returns the throughput lookup key matching GPU profile `measured.compute.tensor_tflops`.
    pub fn profile_key(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}x{}x{}",
            self.op_class,
            self.a_dtype,
            self.b_dtype,
            self.acc_dtype,
            self.inst_mnk[0],
            self.inst_mnk[1],
            self.inst_mnk[2]
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarOp {
    pub op_id: String,
    pub warp_group_id: u32,
    pub stage_offset: u32,
    pub dtype: String,
    /// Instructions per warp per iteration (`insts_per_warp_per_iter`).
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BarrierOp {
    pub op_id: String,
    pub warp_group_id: u32,
    pub stage_offset: u32,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SidecarRef {
    pub file: String,
    pub offset: u64,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "encoding", rename_all = "snake_case")]
pub enum AccessEntry {
    #[serde(rename = "strided_2d_per_outer_tile")]
    Strided2dPerOuterTile {
        #[serde(default)]
        sidecar_ref: Option<SidecarRef>,
        entries: Vec<Strided2dEntry>,
    },
    ExplicitBlocks {
        #[serde(default)]
        sidecar_ref: Option<SidecarRef>,
        entries: Vec<ExplicitBlock>,
    },
}

/// Tuple encoding:
/// `[tile_coord_0, tile_coord_1, base_byte_offset, rows, row_bytes, row_stride_bytes, inner_step_byte_stride, is_masked]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Strided2dEntry(
    pub i64,
    pub i64,
    pub u64,
    pub u64,
    pub u64,
    pub u64,
    pub i64,
    pub bool,
);

impl Strided2dEntry {
    pub fn tile_coord(&self) -> (i64, i64) {
        (self.0, self.1)
    }
    pub fn base_byte_offset(&self) -> u64 {
        self.2
    }
    pub fn rows(&self) -> u64 {
        self.3
    }
    pub fn row_bytes(&self) -> u64 {
        self.4
    }
    pub fn row_stride_bytes(&self) -> u64 {
        self.5
    }
    pub fn inner_step_byte_stride(&self) -> i64 {
        self.6
    }
    pub fn is_masked(&self) -> bool {
        self.7
    }
    /// Post-boundary-mask valid logical bytes for one inner step.
    pub fn valid_bytes(&self) -> u64 {
        self.rows().saturating_mul(self.row_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExplicitBlock {
    pub cta_id: u32,
    pub outer_idx: u32,
    pub inner_step: u64,
    pub byte_offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrValidationError {
    pub message: String,
}

impl IrValidationError {
    fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

impl fmt::Display for IrValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KernelIR validation error: {}", self.message)
    }
}

impl std::error::Error for IrValidationError {}

impl KernelIR {
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    pub fn total_ctas(&self) -> u64 {
        self.launch.grid[0] as u64 * self.launch.grid[1] as u64 * self.launch.grid[2] as u64
    }

    pub fn total_smem_bytes(&self) -> u64 {
        self.resources.static_smem_bytes + self.launch.dynamic_smem_bytes
    }

    pub fn outer_trip_count(&self, cta_id: u32) -> Option<u64> {
        match &self.body.outer_loop.trip_counts {
            TripCounts::Uniform(n) => Some(*n),
            TripCounts::PerItem(v) => v.get(cta_id as usize).copied(),
        }
    }

    pub fn inner_trip_count(&self, cta_id: u32, outer_idx: u32) -> Option<u64> {
        match (
            self.body.pipeline_loop.trip_count_mode,
            &self.body.pipeline_loop.trip_counts,
        ) {
            (InnerTripCountMode::Uniform, TripCounts::Uniform(n)) => Some(*n),
            (InnerTripCountMode::PerCta, TripCounts::PerItem(v)) => v.get(cta_id as usize).copied(),
            (InnerTripCountMode::PerOuterIter, TripCounts::PerItem(v)) => {
                let idx = self.dense_tile_index(cta_id, outer_idx)?;
                v.get(idx).copied()
            }
            _ => None,
        }
    }

    pub fn total_outer_tiles(&self) -> u64 {
        match &self.body.outer_loop.trip_counts {
            TripCounts::Uniform(n) => self.total_ctas().saturating_mul(*n),
            TripCounts::PerItem(v) => v.iter().copied().sum(),
        }
    }

    /// Dense index into `strided_2d_per_outer_tile` entries for `(cta_id, outer_idx)`.
    pub fn dense_tile_index(&self, cta_id: u32, outer_idx: u32) -> Option<usize> {
        match &self.body.outer_loop.trip_counts {
            TripCounts::Uniform(n) => {
                if (cta_id as u64) < self.total_ctas() && (outer_idx as u64) < *n {
                    Some((cta_id as u64 * *n + outer_idx as u64) as usize)
                } else {
                    None
                }
            }
            TripCounts::PerItem(v) => {
                let count = *v.get(cta_id as usize)?;
                if (outer_idx as u64) >= count {
                    return None;
                }
                let prefix: u64 = v[..cta_id as usize].iter().copied().sum();
                Some((prefix + outer_idx as u64) as usize)
            }
        }
    }

    pub fn validate(&self) -> Result<(), IrValidationError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(IrValidationError::new(format!(
                "unsupported schema_version '{}', expected '{}'",
                self.schema_version, SCHEMA_VERSION
            )));
        }
        if self.metadata.kernel_name.is_empty()
            || self.metadata.target_vendor.is_empty()
            || self.metadata.target_arch.is_empty()
            || self.metadata.frontend.is_empty()
        {
            return Err(IrValidationError::new(
                "metadata identifiers must be non-empty",
            ));
        }
        if self.launch.grid.contains(&0) {
            return Err(IrValidationError::new(
                "launch.grid dimensions must be >= 1",
            ));
        }
        if self.launch.cluster_shape.contains(&0) {
            return Err(IrValidationError::new(
                "launch.cluster_shape dimensions must be >= 1",
            ));
        }
        if self.launch.num_warps == 0 {
            return Err(IrValidationError::new("launch.num_warps must be >= 1"));
        }
        if self.launch.warp_size != 32 && self.launch.warp_size != 64 {
            return Err(IrValidationError::new(format!(
                "launch.warp_size must be 32 or 64, got {}",
                self.launch.warp_size
            )));
        }
        if self.launch.warp_groups.is_empty() {
            return Err(IrValidationError::new(
                "launch.warp_groups must be non-empty",
            ));
        }
        let mut wg_ids = HashSet::new();
        let mut wg_warps: u32 = 0;
        for wg in &self.launch.warp_groups {
            if wg.num_warps == 0 {
                return Err(IrValidationError::new("warp_group.num_warps must be >= 1"));
            }
            if !wg_ids.insert(wg.id) {
                return Err(IrValidationError::new(format!(
                    "duplicate warp_group id {}",
                    wg.id
                )));
            }
            wg_warps = wg_warps.saturating_add(wg.num_warps);
        }
        if wg_warps != self.launch.num_warps {
            return Err(IrValidationError::new(format!(
                "sum of warp_groups.num_warps ({}) != launch.num_warps ({})",
                wg_warps, self.launch.num_warps
            )));
        }

        let mut tensor_ids = HashSet::new();
        for t in &self.tensors {
            if t.id.is_empty() || t.dtype.is_empty() || t.element_bits == 0 {
                return Err(IrValidationError::new("invalid tensor descriptor"));
            }
            if t.shape.is_empty() || t.shape.len() != t.strides.len() {
                return Err(IrValidationError::new(format!(
                    "tensor '{}' shape/strides rank mismatch",
                    t.id
                )));
            }
            if t.shape.contains(&0) {
                return Err(IrValidationError::new(format!(
                    "tensor '{}' shape dimensions must be >= 1",
                    t.id
                )));
            }
            if !tensor_ids.insert(t.id.as_str()) {
                return Err(IrValidationError::new(format!(
                    "duplicate tensor id '{}'",
                    t.id
                )));
            }
        }

        match (
            self.body.outer_loop.trip_count_mode,
            &self.body.outer_loop.trip_counts,
        ) {
            (OuterTripCountMode::Uniform, TripCounts::Uniform(n)) if *n >= 1 => {}
            (OuterTripCountMode::PerCta, TripCounts::PerItem(v)) => {
                if v.len() as u64 != self.total_ctas() {
                    return Err(IrValidationError::new(format!(
                        "outer_loop per_cta length ({}) != total_ctas ({})",
                        v.len(),
                        self.total_ctas()
                    )));
                }
                if v.contains(&0) {
                    return Err(IrValidationError::new(
                        "outer_loop per_cta trip counts must be >= 1",
                    ));
                }
            }
            _ => {
                return Err(IrValidationError::new(
                    "outer_loop trip_count_mode does not match trip_counts",
                ));
            }
        }

        let total_outer_tiles = self.total_outer_tiles();
        match (
            self.body.pipeline_loop.trip_count_mode,
            &self.body.pipeline_loop.trip_counts,
        ) {
            (InnerTripCountMode::Uniform, TripCounts::Uniform(_)) => {}
            (InnerTripCountMode::PerCta, TripCounts::PerItem(v)) => {
                if v.len() as u64 != self.total_ctas() {
                    return Err(IrValidationError::new(format!(
                        "pipeline_loop per_cta length ({}) != total_ctas ({})",
                        v.len(),
                        self.total_ctas()
                    )));
                }
            }
            (InnerTripCountMode::PerOuterIter, TripCounts::PerItem(v)) => {
                if v.len() as u64 != total_outer_tiles {
                    return Err(IrValidationError::new(format!(
                        "pipeline_loop per_outer_iter length ({}) != total_outer_tiles ({})",
                        v.len(),
                        total_outer_tiles
                    )));
                }
            }
            _ => {
                return Err(IrValidationError::new(
                    "pipeline_loop trip_count_mode does not match trip_counts",
                ));
            }
        }
        if self.body.pipeline_loop.num_stages == 0 {
            return Err(IrValidationError::new(
                "pipeline_loop.num_stages must be >= 1",
            ));
        }

        let all_ops = self
            .body
            .prologue
            .ops
            .iter()
            .chain(self.body.pipeline_loop.ops.iter())
            .chain(self.body.epilogue.ops.iter());
        for op in all_ops {
            if !wg_ids.contains(&op.warp_group_id()) {
                return Err(IrValidationError::new(format!(
                    "op '{}' references unknown warp_group_id {}",
                    op.op_id(),
                    op.warp_group_id()
                )));
            }
            match op {
                Op::GmemLoad(g) | Op::GmemStore(g) => {
                    if !tensor_ids.contains(g.tensor_id.as_str()) {
                        return Err(IrValidationError::new(format!(
                            "op '{}' references unknown tensor_id '{}'",
                            g.op_id, g.tensor_id
                        )));
                    }
                    if !self.access_map.contains_key(&g.access_ref) {
                        return Err(IrValidationError::new(format!(
                            "op '{}' references unknown access_ref '{}'",
                            g.op_id, g.access_ref
                        )));
                    }
                    if g.transfer_width_bytes == 0 {
                        return Err(IrValidationError::new(format!(
                            "op '{}' transfer_width_bytes must be >= 1",
                            g.op_id
                        )));
                    }
                }
                Op::Mma(m) => {
                    if m.tile_mnk.contains(&0) || m.inst_mnk.contains(&0) || m.count == 0 {
                        return Err(IrValidationError::new(format!(
                            "mma op '{}' has zero dimension or count",
                            m.op_id
                        )));
                    }
                }
                Op::Barrier(b) => {
                    if b.count == 0 {
                        return Err(IrValidationError::new(format!(
                            "barrier op '{}' count must be >= 1",
                            b.op_id
                        )));
                    }
                }
                Op::SmemLoad(_) | Op::SmemStore(_) | Op::Alu(_) | Op::Sfu(_) | Op::Shfl(_) => {}
            }
        }

        for (key, entry) in &self.access_map {
            if let AccessEntry::Strided2dPerOuterTile {
                sidecar_ref: None,
                entries,
            } = entry
            {
                if entries.len() as u64 != total_outer_tiles {
                    return Err(IrValidationError::new(format!(
                        "access_map '{}' has {} entries, expected {} (total_outer_tiles)",
                        key,
                        entries.len(),
                        total_outer_tiles
                    )));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_validate_gemm_sm89_example() {
        let json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let ir = KernelIR::from_json_str(json).expect("gemm_sm89.json should deserialize");
        ir.validate().expect("gemm_sm89.json should validate");
        assert_eq!(ir.total_ctas(), 1024);
        assert_eq!(ir.total_outer_tiles(), 1024);
        assert_eq!(ir.total_smem_bytes(), 32768);

        let mma = ir
            .body
            .pipeline_loop
            .ops
            .iter()
            .find_map(|op| match op {
                Op::Mma(m) => Some(m),
                _ => None,
            })
            .expect("expected mma op");
        assert_eq!(mma.profile_key(), "mma.sync:f16:f16:f32:16x8x16");
        assert_eq!(mma.count, 256);
    }

    #[test]
    fn reject_mismatched_warp_groups() {
        let json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let mut ir = KernelIR::from_json_str(json).unwrap();
        ir.launch.warp_groups[0].num_warps = 2;
        let err = ir.validate().unwrap_err();
        assert!(err.message.contains("warp_groups"));
    }

    #[test]
    fn reject_missing_access_ref() {
        let json = include_str!("../../../../schemas/examples/gemm_sm89.json");
        let mut ir = KernelIR::from_json_str(json).unwrap();
        ir.access_map.remove("acc_A");
        let err = ir.validate().unwrap_err();
        assert!(err.message.contains("acc_A"));
    }

    #[test]
    fn parse_and_validate_flash_attention_causal_gqa_example() {
        let json =
            include_str!("../../../../schemas/examples/flash_attention_causal_gqa_sm89.json");
        let ir = KernelIR::from_json_str(json)
            .expect("flash_attention_causal_gqa_sm89.json should deserialize");
        ir.validate()
            .expect("flash_attention_causal_gqa_sm89.json should validate");
        assert_eq!(ir.total_ctas(), 128);
        assert_eq!(
            ir.body.pipeline_loop.trip_count_mode,
            InnerTripCountMode::PerCta
        );
        assert_eq!(ir.inner_trip_count(0, 0), Some(1));
        assert_eq!(ir.inner_trip_count(15, 0), Some(16));
        assert_eq!(ir.inner_trip_count(16, 0), Some(1));
        assert!(ir
            .body
            .pipeline_loop
            .ops
            .iter()
            .any(|op| matches!(op, Op::Shfl(_))));
        assert!(ir
            .body
            .pipeline_loop
            .ops
            .iter()
            .any(|op| matches!(op, Op::Sfu(_))));
    }
}
