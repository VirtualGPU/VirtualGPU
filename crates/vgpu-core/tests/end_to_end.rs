//! The M1 chain: example KernelIR + measured RTX 4070 profile -> prediction.

use vgpu_core::from_ir::{kernel_from_ir, IrL2};
use vgpu_core::ir::KernelIR;
use vgpu_core::l2::{L2InitState, L2Options};
use vgpu_core::profile::Profile;
use vgpu_core::sim::{simulate, Machine};

const IR: &str = include_str!("../../../schemas/examples/gemm_sm89.json");
const PROFILES: [&str; 2] = [
    include_str!("../../../profiles/rtx4070.json"),
    include_str!("../../../schemas/examples/profile_example.json"),
];

#[test]
fn example_gemm_runs_on_every_profile() {
    let ir = KernelIR::from_json_str(IR).unwrap();
    ir.validate().unwrap();
    let k = kernel_from_ir(&ir).unwrap();
    for text in PROFILES {
        let p = Profile::from_json(text).unwrap();
        let m = Machine::nominal(&p, &k.mma_key).unwrap();
        let mut l2 = IrL2::new(&ir, &p, L2InitState::Cold, L2Options::default());
        let r = simulate(&k, &p, &m, &mut l2).unwrap();
        assert!(r.time_ns.is_finite() && r.time_ns > 0.0);
        assert!(r.dram_bytes > 0.0 && r.dram_bytes <= r.load_bytes + 4096.0 * 4096.0 * 2.0);
        eprintln!(
            "{}: {:.1} us, {:?}, waves {:.2}, tail {:.3}, L2 hit {:.2}",
            p.identity.sku,
            r.time_ns / 1e3,
            r.occupancy,
            r.waves,
            r.tail_efficiency,
            r.l2_hit_bytes / r.load_bytes
        );
    }
}

#[test]
fn causal_attention_example_runs_with_per_tile_iterations() {
    let ir = KernelIR::from_json_str(include_str!(
        "../../../schemas/examples/flash_attention_causal_gqa_sm89.json"
    ))
    .unwrap();
    ir.validate().unwrap();
    let k = kernel_from_ir(&ir).unwrap();
    let per_tile = k
        .tile_iterations
        .as_ref()
        .expect("causal attention has per-tile trip counts");
    assert!(per_tile.iter().flatten().any(|&n| n != per_tile[0][0]));
    assert!(
        k.tile.per_iteration.sfu_insts_per_warp > 0.0
            && k.tile.per_iteration.shfl_insts_per_warp > 0.0
    );
    let p = Profile::from_json(PROFILES[0]).unwrap();
    let m = Machine::nominal(&p, &k.mma_key).unwrap();
    let mut l2 = IrL2::new(&ir, &p, L2InitState::Cold, L2Options::default());
    let r = simulate(&k, &p, &m, &mut l2).unwrap();
    assert!(r.time_ns.is_finite() && r.time_ns > 0.0);
}

fn causal_ir_with_spills(loads: u64, stores: u64, loops: Option<u32>) -> KernelIR {
    let mut v: serde_json::Value = serde_json::from_str(include_str!(
        "../../../schemas/examples/flash_attention_causal_gqa_sm89.json"
    ))
    .unwrap();
    let r = &mut v["resources"];
    r["stack_frame_bytes"] = 512.into();
    r["spill_loads_bytes"] = loads.into();
    r["spill_stores_bytes"] = stores.into();
    r["spill_scope"] = "inner_loop_confirmed".into();
    if let Some(n) = loops {
        r["spill_loop_count"] = n.into();
    }
    let ir = KernelIR::from_json_str(&v.to_string()).unwrap();
    ir.validate().unwrap();
    ir
}

#[test]
fn spill_loop_count_splits_per_iteration_spills() {
    assert_eq!(
        causal_ir_with_spills(0, 0, None).resources.spill_loop_count,
        None
    );
    let split = causal_ir_with_spills(512, 256, Some(2));
    let halved = causal_ir_with_spills(256, 128, None);
    let (ks, kh) = (
        kernel_from_ir(&split).unwrap(),
        kernel_from_ir(&halved).unwrap(),
    );
    assert_eq!(ks.spill, kh.spill);
    let p = Profile::from_json(PROFILES[0]).unwrap();
    let m = Machine::nominal(&p, &ks.mma_key).unwrap();
    let time = |ir: &KernelIR, k| {
        let mut l2 = IrL2::new(ir, &p, L2InitState::Cold, L2Options::default());
        simulate(k, &p, &m, &mut l2).unwrap().time_ns
    };
    assert_eq!(time(&split, &ks), time(&halved, &kh));
    let full = causal_ir_with_spills(512, 256, None);
    assert!(time(&full, &kernel_from_ir(&full).unwrap()) > time(&split, &ks));
}

#[test]
fn spill_loop_count_leaves_once_scope_whole() {
    let mut ir = causal_ir_with_spills(512, 256, Some(2));
    ir.resources.spill_scope = vgpu_core::ir::SpillScope::PrologueEpilogue;
    let k = kernel_from_ir(&ir).unwrap();
    assert_eq!(k.spill.bytes_per_thread, 768.0);
}

#[test]
fn ascend910b3_profile_simulates_cube_gemm_and_enforces_sram_limits() {
    use vgpu_core::occupancy::{occupancy, OccupancyError};

    let p = Profile::from_json(include_str!("../../../profiles/ascend910b3.json")).unwrap();
    assert_eq!(p.identity.vendor, "ascend");
    assert_eq!(p.identity.arch, "ascend910b3");
    assert_eq!(p.specs.sm_count, 20);
    let asc = p.specs.ascend.as_ref().expect("ascend specs present");
    assert_eq!(asc.l1_bytes, 524_288);
    assert_eq!(asc.l0c_bytes, 131_072);
    assert_eq!(asc.ub_bytes, 196_608);

    // 2048x2048x2048 FP16 GEMM, 128x128x64 tile, auto-blockified onto 20 AI Cores (12-13 tiles/core).
    let mut ir = KernelIR::from_json_str(IR).unwrap();
    ir.metadata.target_vendor = "ascend".into();
    ir.metadata.target_arch = "ascend910b3".into();
    ir.metadata.frontend = "triton_ascend".into();
    ir.launch.grid = [20, 1, 1];
    ir.launch.num_warps = 1;
    ir.launch.warp_size = 32;
    ir.launch.warp_groups[0].num_warps = 1;
    ir.resources.registers_per_thread = 1;
    ir.resources.static_smem_bytes = 65_536;
    ir.resources.vendor_extensions = Some(vgpu_core::ir::ResourceVendorExtensions {
        ascend: Some(vgpu_core::ir::AscendResourceUsage {
            ub_bytes: Some(0),
            l0c_bytes: Some(65_536),
            workspace_bytes: Some(0),
            workspace_loop_bytes: Some(0),
            workspace_multibuffer: Some(0),
            cv_sync_hops_per_iter: Some(0),
        }),
    });
    ir.body.outer_loop.trip_count_mode = vgpu_core::ir::OuterTripCountMode::Uniform;
    ir.body.outer_loop.trip_counts = vgpu_core::ir::TripCounts::Uniform(12);
    ir.body.pipeline_loop.trip_counts = vgpu_core::ir::TripCounts::Uniform(32);
    ir.body.pipeline_loop.num_stages = 2;
    ir.body.pipeline_loop.prefetch_distance = 1;
    for op in &mut ir.body.pipeline_loop.ops {
        if let vgpu_core::ir::Op::Mma(m) = op {
            m.op_class = "cube.mmad".into();
            m.tile_mnk = [128, 128, 64];
            m.inst_mnk = [16, 16, 16];
            m.count = 8 * 8 * 4;
        }
    }
    for entry in ir.access_map.values_mut() {
        if let vgpu_core::ir::AccessEntry::Strided2dPerOuterTile { entries, .. } = entry {
            entries.truncate(240);
        }
    }
    ir.validate().unwrap();

    let k = kernel_from_ir(&ir).unwrap();
    let m = Machine::nominal(&p, &k.mma_key).unwrap();
    let mut l2 = IrL2::new(&ir, &p, L2InitState::Cold, L2Options::default());
    let r = simulate(&k, &p, &m, &mut l2).unwrap();
    assert!(r.time_ns.is_finite() && r.time_ns > 0.0);
    assert_eq!(r.occupancy.ctas_per_sm, 1);

    // UB overflow (> 192 KiB) must be rejected with a dedicated on-chip buffer error.
    let mut bad_ub = k.clone();
    bad_ub.ub_bytes = asc.ub_bytes + 32;
    assert!(matches!(
        occupancy(&bad_ub, &p.specs),
        Err(OccupancyError::TooMuchOnChipBuffer { buffer: "ub", .. })
    ));

    // L0C overflow (> 128 KiB) must be rejected with a dedicated on-chip buffer error.
    let mut bad_l0c = k;
    bad_l0c.l0c_bytes = asc.l0c_bytes + 32;
    assert!(matches!(
        occupancy(&bad_l0c, &p.specs),
        Err(OccupancyError::TooMuchOnChipBuffer { buffer: "l0c", .. })
    ));
}

#[test]
fn ascend910b3_cube_vector_split_overlaps_and_keeps_workspace_in_l2() {
    use vgpu_core::kernel::{CvWorkspace, Kernel, Spill, Tile, TripCounts, Work};
    use vgpu_core::sim::ConstantHitRatio;

    let p = Profile::from_json(include_str!("../../../profiles/ascend910b3.json")).unwrap();
    let mma_key = "cube.mmad:f16:f16:f32:16x16x16";
    let m = Machine::nominal(&p, mma_key).unwrap();
    assert!(m.cube_vector_split);

    // Balanced Cube MMA (~287 ns/iter) + Vector issue (~271 ns/iter) with L2-resident workspace.
    let base = Kernel {
        grid: 20,
        num_warps: 1,
        registers_per_thread: 1,
        smem_per_cta_bytes: 65_536,
        ub_bytes: 65_536,
        l0c_bytes: 65_536,
        cv_workspace: CvWorkspace {
            bytes_per_core: 163_840,     // 20 * 160 KiB = 3.2 MiB << 192 MiB L2
            loop_bytes_per_iter: 32_768, // 32 KiB/iter AIC<->AIV bounce
            overlap: false,
            sync_hops_per_iter: 0,
        },
        tiles_per_program: TripCounts::Uniform(1),
        mma_key: mma_key.to_string(),
        spill: Spill::default(),
        async_loads: true,
        tile_iterations: None,
        tile: Tile {
            prologue: Work::default(),
            iterations: 64,
            per_iteration: Work {
                mma_flops: 2.0 * 128.0 * 64.0 * 128.0 * 2.0,
                sfu_insts_per_warp: 256.0,
                alu_insts_per_warp: 64.0,
                ..Work::default()
            },
            prefetch_distance: 1,
            epilogue: Work::default(),
        },
    };
    let mut overlapped = base.clone();
    overlapped.cv_workspace.overlap = true;

    let r_serial = simulate(&base, &p, &m, &mut ConstantHitRatio(1.0)).unwrap();
    let r_overlap = simulate(&overlapped, &p, &m, &mut ConstantHitRatio(1.0)).unwrap();

    // Multi-buffered workspace ring overlaps Cube and Vector work (faster than serial sum).
    assert!(
        r_overlap.makespan_ns < r_serial.makespan_ns * 0.75,
        "expected Cube/Vector overlap speedup: overlap={} vs serial={}",
        r_overlap.makespan_ns,
        r_serial.makespan_ns
    );
    // Workspace fits in L2 (3.2 MiB << 192 MiB) -> L2 hit traffic, zero DRAM write-through.
    assert!(r_overlap.l2_hit_bytes > 0.0);
    assert_eq!(r_overlap.dram_bytes, 0.0);

    // Oversized workspace (> 192 MiB total across 20 cores) spills overflow into DRAM.
    let mut spilled_ws = overlapped.clone();
    spilled_ws.cv_workspace.bytes_per_core = 16 * 1024 * 1024; // 320 MiB > 192 MiB L2
    let r_spill = simulate(&spilled_ws, &p, &m, &mut ConstantHitRatio(1.0)).unwrap();
    assert!(r_spill.dram_bytes > 0.0);

    // Cross-core sync hops add measured FFTS+ handoff latency to the sync path.
    let mut synced = overlapped;
    synced.cv_workspace.sync_hops_per_iter = 6;
    let r_synced = simulate(&synced, &p, &m, &mut ConstantHitRatio(1.0)).unwrap();
    assert!(
        r_synced.makespan_ns > r_overlap.makespan_ns * 3.0,
        "expected cross-core sync_block latency: synced={} vs overlap={}",
        r_synced.makespan_ns,
        r_overlap.makespan_ns
    );
}

#[test]
fn ascend910b3_single_mte2_pipeline_and_nd2nz_row_issue() {
    use vgpu_core::kernel::{Access, CvWorkspace, Kernel, Spill, Tile, TripCounts, Work};
    use vgpu_core::sim::ConstantHitRatio;

    let p = Profile::from_json(include_str!("../../../profiles/ascend910b3.json")).unwrap();
    let mma_key = "cube.mmad:f16:f16:f32:16x16x16";
    let m = Machine::nominal(&p, mma_key).unwrap();
    assert!(m.cube_vector_split);
    assert!(
        m.lsu_rate.is_finite() && m.lsu_rate > 0.1 && m.lsu_rate < 2.0,
        "expected measured Nd2Nz row-issue rate in profile, got {}",
        m.lsu_rate
    );

    let make_gemm =
        |prefetch_distance: u32, width_a: f64, width_b: f64, bytes_a: f64, bytes_b: f64| Kernel {
            grid: 20,
            num_warps: 1,
            registers_per_thread: 1,
            smem_per_cta_bytes: 65_536,
            ub_bytes: 0,
            l0c_bytes: 65_536,
            cv_workspace: CvWorkspace::default(),
            tiles_per_program: TripCounts::Uniform(8),
            mma_key: mma_key.to_string(),
            spill: Spill::default(),
            async_loads: prefetch_distance > 0,
            tile_iterations: None,
            tile: Tile {
                prologue: Work::default(),
                iterations: 32,
                per_iteration: Work {
                    loads: vec![
                        Access {
                            op_id: "load_A".into(),
                            nominal_bytes: bytes_a,
                            width_bytes: width_a,
                        },
                        Access {
                            op_id: "load_B".into(),
                            nominal_bytes: bytes_b,
                            width_bytes: width_b,
                        },
                    ],
                    smem_bytes: bytes_a + bytes_b,
                    mma_flops: 2.0 * 128.0 * 64.0 * 64.0,
                    ..Work::default()
                },
                prefetch_distance,
                epilogue: Work::default(),
            },
        };

    // Single-buffered vs double-buffered AIC GEMM: L0A/L0B decouples Cube from L1, so
    // d=0 adds at most MTE1 (smem_load) rather than full Cube MMA time.
    let mb1 = make_gemm(1, 4.0, 8.0, 8192.0, 16384.0); // m64n128k64: 64 + 64 = 128 rows
    let mb0 = make_gemm(0, 4.0, 8.0, 8192.0, 16384.0);
    let r_mb1 = simulate(&mb1, &p, &m, &mut ConstantHitRatio(0.5)).unwrap();
    let r_mb0 = simulate(&mb0, &p, &m, &mut ConstantHitRatio(0.5)).unwrap();
    assert!(
        r_mb0.makespan_ns >= r_mb1.makespan_ns && r_mb0.makespan_ns < r_mb1.makespan_ns * 1.25,
        "expected modest single-buffer MTE1 overhead, got mb0={} vs mb1={}",
        r_mb0.makespan_ns,
        r_mb1.makespan_ns
    );

    // Measured Nd2Nz row-issue rate distinguishes m128n64k64 (192 rows/iter) from
    // m64n128k64 (128 rows/iter) at equal bytes and FLOPs.
    let tall_a = make_gemm(1, 4.0, 4.0, 16384.0, 8192.0); // m128n64k64: 128 + 64 = 192 rows
    let r_tall = simulate(&tall_a, &p, &m, &mut ConstantHitRatio(0.5)).unwrap();
    assert!(
        r_tall.makespan_ns > r_mb1.makespan_ns * 1.25,
        "expected 192-row tile to run slower than 128-row tile: tall={} vs wide={}",
        r_tall.makespan_ns,
        r_mb1.makespan_ns
    );
}
