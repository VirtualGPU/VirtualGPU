//! Performance budget from `docs/components/model-core.md` 3.1: a 4096-tile
//! GEMM should simulate in about 1 ms.
//!
//! Run with `cargo bench -p vgpu-core --bench gemm`.

use std::time::Instant;
use vgpu_core::from_ir::{kernel_from_ir, IrL2};
use vgpu_core::ir::KernelIR;
use vgpu_core::kernel::{Access, Kernel, Spill, Tile, TripCounts, Work};
use vgpu_core::l2::{L2InitState, L2Options};
use vgpu_core::profile::Profile;
use vgpu_core::sim::{simulate, ConstantHitRatio, Machine};

fn access(op: &str, bytes: f64) -> Access {
    Access {
        op_id: op.to_string(),
        nominal_bytes: bytes,
        width_bytes: 16.0,
    }
}

fn main() {
    let p = Profile::from_json(include_str!("../tests/data/test_fixture_sm89.json")).unwrap();
    // M = N = 8192 with 128x128 tiles, K = 4096 with BK = 32, num_stages = 3.
    let k = Kernel {
        grid: 64 * 64,
        num_warps: 4,
        registers_per_thread: 128,
        smem_per_cta_bytes: 3 * 16384,
        tiles_per_program: TripCounts::Uniform(1),
        mma_key: "mma.sync:f16:f16:f32:16x8x16".to_string(),
        spill: Spill::default(),
        async_loads: true,
        tile_iterations: None,
        tile: Tile {
            prologue: Work::default(),
            iterations: 128,
            per_iteration: Work {
                mma_flops: 2.0 * 128.0 * 128.0 * 32.0,
                alu_insts_per_warp: 16.0,
                smem_bytes: 16384.0,
                loads: vec![access("load_A", 8192.0), access("load_B", 8192.0)],
                ..Work::default()
            },
            prefetch_distance: 2,
            epilogue: Work {
                stores: vec![access("store_C", 32768.0)],
                alu_insts_per_warp: 64.0,
                ..Work::default()
            },
        },
    };
    let m = Machine::nominal(&p, &k.mma_key).unwrap();

    let runs = 200;
    let mut last = None;
    let start = Instant::now();
    for _ in 0..runs {
        last = Some(simulate(&k, &p, &m, &mut ConstantHitRatio(0.5)).unwrap());
    }
    let per_run = start.elapsed().as_secs_f64() / f64::from(runs);
    let r = last.take().unwrap();
    println!(
        "tiles: {}, events: {}, predicted: {:.1} us",
        k.grid,
        r.events,
        r.time_ns / 1e3
    );
    println!("per simulation: {:.3} ms", per_run * 1e3);

    // The same budget with the KernelIR adapter and the real L2 model.
    let ir =
        KernelIR::from_json_str(include_str!("../../../schemas/examples/gemm_sm89.json")).unwrap();
    let k = kernel_from_ir(&ir).unwrap();
    let m = Machine::nominal(&p, &k.mma_key).unwrap();
    let start = Instant::now();
    for _ in 0..runs {
        let mut l2 = IrL2::new(&ir, &p, L2InitState::Cold, L2Options::default());
        last = Some(simulate(&k, &p, &m, &mut l2).unwrap());
    }
    let per_run = start.elapsed().as_secs_f64() / f64::from(runs);
    let start = Instant::now();
    for _ in 0..runs {
        std::hint::black_box(IrL2::new(&ir, &p, L2InitState::Cold, L2Options::default()));
    }
    let per_new = start.elapsed().as_secs_f64() / f64::from(runs);
    println!("L2 model construction: {:.3} ms", per_new * 1e3);
    let start = Instant::now();
    for _ in 0..runs {
        std::hint::black_box(simulate(&k, &p, &m, &mut ConstantHitRatio(0.5)).unwrap());
    }
    let per_const = start.elapsed().as_secs_f64() / f64::from(runs);
    println!(
        "example IR with a constant hit ratio: {:.3} ms",
        per_const * 1e3
    );
    let r = last.unwrap();
    println!(
        "example IR with L2 model: tiles: {}, events: {}, predicted: {:.1} us",
        k.grid,
        r.events,
        r.time_ns / 1e3
    );
    println!("per simulation: {:.3} ms", per_run * 1e3);
}
