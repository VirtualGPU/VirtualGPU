//! The kernel description the model core simulates.
//!
//! This is the core's own, already-resolved view of one kernel variant. The
//! KernelIR adapter fills it in; tests build it directly.

/// One global memory access stream of a phase, identified by the KernelIR op.
#[derive(Debug, Clone, PartialEq)]
pub struct Access {
    pub op_id: String,
    /// Unmasked bytes per CTA per step (per iteration, or per phase execution).
    pub nominal_bytes: f64,
    /// Bytes each thread moves per memory instruction.
    pub width_bytes: f64,
}

/// Work done by one CTA in one pipeline iteration, or in one execution of the
/// prologue or epilogue.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Work {
    /// Tensor-core floating-point operations.
    pub mma_flops: f64,
    /// Non-MMA instructions, per warp: all of them take issue slots; the
    /// special-function and shuffle ones also take their own pipes.
    pub alu_insts_per_warp: f64,
    pub sfu_insts_per_warp: f64,
    pub shfl_insts_per_warp: f64,
    /// Shared memory bytes moved by the CTA.
    pub smem_bytes: f64,
    /// Barriers (`bar.sync`) per execution, from the IR's barrier ops.
    pub barriers: f64,
    pub loads: Vec<Access>,
    pub stores: Vec<Access>,
}

/// What one CTA does for one output tile.
#[derive(Debug, Clone, PartialEq)]
pub struct Tile {
    /// Work outside the software pipeline, executed once before it.
    pub prologue: Work,
    pub iterations: u32,
    pub per_iteration: Work,
    /// Iterations in flight in the software pipeline. The first
    /// `prefetch_distance` iterations' loads are issued in the prologue. Zero
    /// means synchronous loads: no overlap of loads and compute within the CTA.
    pub prefetch_distance: u32,
    pub epilogue: Work,
}

/// When register spill traffic happens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpillScope {
    #[default]
    None,
    /// Once per tile, outside the pipeline loop.
    Once,
    /// Every pipeline iteration. `assumed` when the compiler did not confirm
    /// that the spill code is inside the loop.
    PerIteration { assumed: bool },
}

/// Register spills as reported by the assembler.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Spill {
    /// Spill load plus store bytes per thread per execution of the scope.
    pub bytes_per_thread: f64,
    /// Spill load bytes per thread per execution of the scope.
    pub load_bytes_per_thread: f64,
    /// Local memory frame per thread.
    pub stack_frame_bytes: u64,
    pub scope: SpillScope,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TripCounts {
    Uniform(u32),
    PerProgram(Vec<u32>),
}

/// Cube/Vector split inter-core workspace traffic and pipeline overlap.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CvWorkspace {
    /// Per-AI-Core GM workspace allocation in bytes (used to check L2 residency).
    pub bytes_per_core: u64,
    /// Total GM workspace bytes moved (store + load) per pipeline iteration.
    pub loop_bytes_per_iter: u64,
    /// True when a multi-buffered workspace ring (>= 2 slots) overlaps Cube and Vector work.
    pub overlap: bool,
    /// Cross-core AIC <-> AIV synchronization handoffs per steady-state iteration.
    pub sync_hops_per_iter: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    /// Number of programs (CTAs) launched.
    pub grid: u32,
    pub num_warps: u32,
    pub registers_per_thread: u32,
    /// Static plus dynamic shared memory per CTA (or L1 `cbuf` bytes on Ascend).
    pub smem_per_cta_bytes: u64,
    /// Optional Ascend Unified Buffer (UB) footprint per AIV core (0 on non-Ascend).
    pub ub_bytes: u64,
    /// Optional Ascend L0C accumulator footprint per AIC core (0 on non-Ascend).
    pub l0c_bytes: u64,
    /// Optional Cube/Vector split workspace traffic and overlap (default zero/false on non-Ascend).
    pub cv_workspace: CvWorkspace,
    /// Tiles processed by each program: one for ordinary kernels, more for
    /// persistent kernels.
    pub tiles_per_program: TripCounts,
    /// Profile key of the MMA instruction, `<op_class>:<a>:<b>:<acc>:<MxNxK>`.
    pub mma_key: String,
    pub tile: Tile,
    pub spill: Spill,
    /// Pipeline loads are asynchronous copies (`cp.async`, TMA) rather than
    /// loads into registers.
    pub async_loads: bool,
    /// Pipeline iterations per `[program][tile]` when they vary, for example
    /// causal attention; `None` means `tile.iterations` everywhere.
    pub tile_iterations: Option<Vec<Vec<u32>>>,
}

impl Kernel {
    pub fn iterations(&self, program: u32, tile: u32) -> u32 {
        self.tile_iterations
            .as_ref()
            .map_or(self.tile.iterations, |v| v[program as usize][tile as usize])
    }

    pub fn tiles_of(&self, program: u32) -> u32 {
        match &self.tiles_per_program {
            TripCounts::Uniform(n) => *n,
            TripCounts::PerProgram(v) => v[program as usize],
        }
    }
}
