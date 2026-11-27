//! PyO3 bindings, exposed to Python as `virtualgpu._core`.

use crate::draws::{draws as draw_profiles, Draw};
use crate::from_ir::{kernel_from_ir, IrL2};
use crate::ir::{KernelIR, Op};
use crate::kernel::Kernel;
use crate::l2::L2InitState;
use crate::sim::{simulate, Machine, ModelError, Prediction, RESOURCES};
use crate::verdict::{verdict, Verdict};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn err(e: impl std::fmt::Debug) -> PyErr {
    PyValueError::new_err(format!("{e:?}"))
}

/// Map `f` over `items` on all available cores, keeping their order.
fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = items.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|part| scope.spawn(|| part.iter().map(&f).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("prediction thread panicked"))
            .collect()
    })
}

fn initial_state(l2_state: &str) -> PyResult<L2InitState> {
    match l2_state {
        "cold" => Ok(L2InitState::Cold),
        "warm" => Ok(L2InitState::Warm),
        other => Err(PyValueError::new_err(format!(
            "l2_state must be cold or warm, got {other}"
        ))),
    }
}

fn parse_ir(json: &str) -> PyResult<KernelIR> {
    let ir = KernelIR::from_json_str(json).map_err(err)?;
    ir.validate().map_err(err)?;
    Ok(ir)
}

/// Draw 0's prediction and the time under every draw, in draw order.
type Evaluation = Result<(Prediction, Vec<f64>), ModelError>;

fn simulate_draw(
    ir: &KernelIR,
    k: &Kernel,
    draw: &Draw,
    initial: L2InitState,
) -> Result<Prediction, ModelError> {
    let mut m = Machine::nominal(&draw.profile, &k.mma_key)?;
    m.assumed_spill_scale = draw.assumed_spill_scale;
    let mut l2 = IrL2::new(ir, &draw.profile, initial, draw.l2);
    simulate(k, &draw.profile, &m, &mut l2)
}

/// Simulate one variant under all draws, spreading the draws over cores if
/// `parallel`.
fn evaluate(ir: &KernelIR, all: &[Draw], initial: L2InitState, parallel: bool) -> Evaluation {
    let k = kernel_from_ir(ir)?;
    let r = simulate_draw(ir, &k, &all[0], initial)?;
    let one = |draw: &Draw| simulate_draw(ir, &k, draw, initial).map(|p| p.time_ns);
    let rest: Result<Vec<f64>, ModelError> = if parallel {
        parallel_map(&all[1..], one).into_iter().collect()
    } else {
        all[1..].iter().map(one).collect()
    };
    let mut times = vec![r.time_ns];
    times.extend(rest?);
    Ok((r, times))
}

fn record<'py>(
    py: Python<'py>,
    ir: &KernelIR,
    all: &[Draw],
    l2_state: &str,
    result: Evaluation,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("build", build_info(py)?)?;
    d.set_item("l2_state", l2_state)?;
    d.set_item("dispatch_policy", "round_robin")?;
    let (r, times) = match result {
        Ok(x) => x,
        Err(ModelError::EmptyCurve(name)) => {
            return Err(PyValueError::new_err(format!(
                "profile curve {name} is empty"
            )))
        }
        Err(e) => {
            d.set_item("status", "unsupported")?;
            d.set_item("reason", format!("{e:?}"))?;
            return Ok(d);
        }
    };
    d.set_item("status", "ok")?;
    d.set_item("time_ns", r.time_ns)?;
    let by_draw = PyDict::new(py);
    for (draw, t) in all.iter().zip(times) {
        by_draw.set_item(draw.id.to_string(), t)?;
    }
    d.set_item("draws", by_draw)?;
    d.set_item("makespan_ns", r.makespan_ns)?;
    d.set_item("ctas_per_sm", r.occupancy.ctas_per_sm)?;
    d.set_item("occupancy_limiter", format!("{:?}", r.occupancy.limiter))?;
    d.set_item("waves", r.waves)?;
    d.set_item("tail_efficiency", r.tail_efficiency)?;
    d.set_item("load_bytes", r.load_bytes)?;
    d.set_item("l2_hit_bytes", r.l2_hit_bytes)?;
    d.set_item("merged_bytes", r.merged_bytes)?;
    d.set_item("dram_bytes", r.dram_bytes)?;
    let bound = PyDict::new(py);
    for (name, f) in RESOURCES.iter().zip(r.bound_fraction) {
        bound.set_item(name, f)?;
    }
    d.set_item("bound_fraction", bound)?;
    let mut flags: Vec<String> = Vec::new();
    if ir.resources.spill_loads_bytes + ir.resources.spill_stores_bytes > 0 {
        flags.push("spill".into());
    }
    // Special-function and shuffle instructions are validated only where the
    // profile measured their throughput.
    let compute = &all[0].profile.measured.compute;
    let uses_unmeasured_pipe = [
        &ir.body.prologue.ops,
        &ir.body.pipeline_loop.ops,
        &ir.body.epilogue.ops,
    ]
    .iter()
    .flat_map(|ops| ops.iter())
    .any(|op| match op {
        Op::Sfu(_) => compute.sfu_issue_rate_per_sm_gips.is_empty(),
        Op::Shfl(_) => compute.shfl_issue_rate_per_sm_gips.is_empty(),
        _ => false,
    });
    let mut diagnostics = ir.diagnostics.clone();
    let is_unvalidated_vendor = (all[0].profile.identity.vendor == "ascend"
        && all[0].profile.identity.arch != "ascend910b3")
        || (ir.metadata.target_vendor == "ascend" && ir.metadata.target_arch != "ascend910b3");
    if is_unvalidated_vendor && !diagnostics.iter().any(|d| d == "unvalidated_target_ascend") {
        diagnostics.push("unvalidated_target_ascend".into());
    }
    if !diagnostics.is_empty() || uses_unmeasured_pipe {
        flags.push("unvalidated_feature".into());
    }
    let dispatch = &all[0].profile.measured.cta_dispatch;
    if dispatch.candidate_policies.len() > 1 || dispatch.confidence.as_deref() != Some("high") {
        flags.push("dispatch_ambiguous".into());
    }
    d.set_item("flags", flags)?;
    d.set_item("diagnostics", diagnostics)?;
    Ok(d)
}

/// Predict one kernel variant on one GPU profile under perturbation draws.
///
/// Takes the KernelIR and profile as JSON strings, `l2_state` as "cold" or
/// "warm", the number of perturbed draws and their seed. Returns a dict with
/// `status`: "ok" with the draw 0 (unperturbed) prediction and diagnostics
/// plus `draws`, the time per draw ID including "0"; or "unsupported" with a
/// `reason` (for example a CTA that does not fit on an SM). Draws depend only
/// on the profile, `draws` and `seed`, so variants predicted with the same
/// arguments share them. Malformed input raises `ValueError`.
#[pyfunction]
#[pyo3(signature = (kernel_ir_json, profile_json, l2_state = "cold", draws = 32, seed = 17))]
fn predict<'py>(
    py: Python<'py>,
    kernel_ir_json: &str,
    profile_json: &str,
    l2_state: &str,
    draws: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let initial = initial_state(l2_state)?;
    let ir = parse_ir(kernel_ir_json)?;
    let all = draw_profiles(profile_json, draws, seed).map_err(err)?;
    let result = py.detach(|| evaluate(&ir, &all, initial, true));
    record(py, &ir, &all, l2_state, result)
}

/// Predict many kernel variants on one GPU profile, as `predict` does for
/// each, but parsing the profile and sampling the draws once and spreading
/// the variants over cores. Returns one dict per variant, in order, each
/// identical to what `predict` returns for it. A malformed variant raises
/// `ValueError` naming its index.
#[pyfunction]
#[pyo3(signature = (kernel_ir_jsons, profile_json, l2_state = "cold", draws = 32, seed = 17))]
fn predict_batch<'py>(
    py: Python<'py>,
    kernel_ir_jsons: Vec<String>,
    profile_json: &str,
    l2_state: &str,
    draws: usize,
    seed: u64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let initial = initial_state(l2_state)?;
    let irs = kernel_ir_jsons
        .iter()
        .enumerate()
        .map(|(i, j)| parse_ir(j).map_err(|e| PyValueError::new_err(format!("variant {i}: {e}"))))
        .collect::<PyResult<Vec<_>>>()?;
    let all = draw_profiles(profile_json, draws, seed).map_err(err)?;
    let results = py.detach(|| parallel_map(&irs, |ir| evaluate(ir, &all, initial, false)));
    irs.iter()
        .zip(results)
        .map(|(ir, r)| record(py, ir, &all, l2_state, r))
        .collect()
}

/// Pairwise verdict from two variants' times under the same draws, in draw
/// order: "a_faster", "b_faster" or "indistinguishable". A is faster only if
/// it is at least 5% faster in every draw.
#[pyfunction]
fn pair_verdict(a: Vec<f64>, b: Vec<f64>) -> PyResult<&'static str> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("both variants need the same draws"));
    }
    Ok(match verdict(&a, &b) {
        Verdict::AFaster => "a_faster",
        Verdict::BFaster => "b_faster",
        Verdict::Indistinguishable => "indistinguishable",
    })
}

/// The build that produces predictions: `{"commit": sha, "dirty": bool,
/// "result_class": "formal" | "diagnostic"}`. A build with uncommitted
/// changes to compiled code only produces diagnostic results.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("commit", crate::BUILD_COMMIT)?;
    d.set_item("dirty", crate::BUILD_DIRTY)?;
    d.set_item(
        "result_class",
        if crate::BUILD_DIRTY {
            "diagnostic"
        } else {
            "formal"
        },
    )?;
    Ok(d)
}

/// The JSON Schema contracts, embedded at build time so installed wheels have them.
#[pyfunction]
fn schema(name: &str) -> PyResult<&'static str> {
    match name {
        "kernel_ir" => Ok(include_str!("../../../schemas/kernel_ir.schema.json")),
        "gpu_profile" => Ok(include_str!("../../../schemas/gpu_profile.schema.json")),
        other => Err(PyValueError::new_err(format!(
            "unknown schema {other}; expected kernel_ir or gpu_profile"
        ))),
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(predict, m)?)?;
    m.add_function(wrap_pyfunction!(predict_batch, m)?)?;
    m.add_function(wrap_pyfunction!(schema, m)?)?;
    m.add_function(wrap_pyfunction!(pair_verdict, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    Ok(())
}
