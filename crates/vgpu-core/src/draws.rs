//! Profile perturbation draws (`docs/components/model-core.md`, section 4).
//!
//! Draw 0 is the unperturbed profile. Draws 1 to N are a Latin hypercube over
//! every perturbation dimension:
//!
//! - each measured value is scaled uniformly within `+/- rel_uncertainty`;
//!   values sharing a `perturb_group`, and all points of one curve
//!   (`*_by_active_sms`, `*_by_warps`, ...), share one dimension and one
//!   multiplier, so a curve keeps its shape. The multiplier's range is the
//!   largest `rel_uncertainty` among the values sharing it;
//! - the L2 model's eviction mode and cross-wave reuse;
//! - the scale of spill traffic whose loop placement the compiler did not
//!   confirm, from none to every iteration.
//!
//! Specification fields are never perturbed. The draws depend only on the
//! profile, `n` and the seed, never on the kernel, so all variants are
//! compared under the same draws (common random numbers).

use crate::l2::{EvictionMode, L2Options};
use crate::profile::Profile;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct Draw {
    pub id: u32,
    pub profile: Profile,
    pub l2: L2Options,
    /// Multiplier on per-iteration spill traffic with an assumed loop placement.
    pub assumed_spill_scale: f64,
}

/// SplitMix64: small, deterministic, and good enough for sampling.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// `n` samples of `dims` dimensions, each stratified into `n` intervals.
fn latin_hypercube(n: usize, dims: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut samples = vec![vec![0.0; dims]; n];
    for d in 0..dims {
        let mut strata: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            strata.swap(i, rng.below(i + 1));
        }
        for (sample, s) in samples.iter_mut().zip(strata) {
            sample[d] = (s as f64 + rng.unit()) / n as f64;
        }
    }
    samples
}

fn is_measured(v: &Value) -> bool {
    v.get("value").is_some_and(Value::is_number)
        && v.get("rel_uncertainty").is_some_and(Value::is_number)
}

/// Visit every measured value with its perturbation dimension key.
fn visit(v: &mut Value, path: &str, curve: Option<&str>, f: &mut impl FnMut(&str, &mut Value)) {
    if is_measured(v) {
        let key = match v.get("perturb_group").and_then(Value::as_str) {
            Some(g) => format!("group:{g}"),
            None => curve.unwrap_or(path).to_string(),
        };
        f(&key, v);
        return;
    }
    match v {
        Value::Object(map) => {
            for (k, child) in map.iter_mut() {
                visit(child, &format!("{path}.{k}"), curve, f);
            }
        }
        Value::Array(items) => {
            let curve = curve.unwrap_or(path).to_string();
            for (i, child) in items.iter_mut().enumerate() {
                visit(child, &format!("{path}[{i}]"), Some(&curve), f);
            }
        }
        _ => {}
    }
}

/// Draw 0 plus `n` perturbed draws of the profile.
pub fn draws(profile_json: &str, n: usize, seed: u64) -> Result<Vec<Draw>, serde_json::Error> {
    let nominal: Value = serde_json::from_str(profile_json)?;
    let mut measured = nominal.get("measured").cloned().unwrap_or(Value::Null);

    // Per key: dimension index, in a stable order, and the largest
    // relative uncertainty of the values sharing it.
    let mut dims: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    visit(&mut measured, "measured", None, &mut |key, mv| {
        let u = mv["rel_uncertainty"].as_f64().unwrap_or(0.0);
        let next = dims.len();
        let d = dims.entry(key.to_string()).or_insert((next, 0.0));
        d.1 = d.1.max(u);
    });
    let discrete = dims.len();
    let total = discrete + 3; // eviction mode, cross-wave reuse, assumed spill scale

    let mut rng = Rng(seed);
    let samples = latin_hypercube(n, total, &mut rng);
    let mut out = vec![Draw {
        id: 0,
        profile: serde_json::from_value(nominal.clone())?,
        l2: L2Options::default(),
        assumed_spill_scale: 1.0,
    }];
    for (i, x) in samples.iter().enumerate() {
        let mut v = nominal.clone();
        visit(&mut v["measured"], "measured", None, &mut |key, mv| {
            let (dim, u) = dims[key];
            let scale = 1.0 + u * (2.0 * x[dim] - 1.0);
            let value = mv["value"].as_f64().unwrap_or(0.0) * scale;
            mv["value"] = Value::from(value);
        });
        out.push(Draw {
            id: i as u32 + 1,
            profile: serde_json::from_value(v)?,
            l2: L2Options {
                eviction_mode: if x[discrete] < 0.5 {
                    EvictionMode::SoftLru
                } else {
                    EvictionMode::StrictLru
                },
                cross_wave_reuse: x[discrete + 1] < 0.5,
            },
            assumed_spill_scale: x[discrete + 2],
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE: &str = include_str!("../../../profiles/rtx4070.json");

    fn dram_curve(p: &Profile) -> Vec<f64> {
        p.measured
            .memory
            .dram_bw_by_active_sms
            .iter()
            .map(|q| q.bw_gbps.value)
            .collect()
    }

    #[test]
    fn draw_zero_is_nominal_and_draws_are_reproducible() {
        let a = draws(PROFILE, 8, 17).unwrap();
        let b = draws(PROFILE, 8, 17).unwrap();
        let c = draws(PROFILE, 8, 18).unwrap();
        assert_eq!(a.len(), 9);
        let nominal = Profile::from_json(PROFILE).unwrap();
        assert_eq!(dram_curve(&a[0].profile), dram_curve(&nominal));
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(dram_curve(&x.profile), dram_curve(&y.profile));
        }
        assert_ne!(dram_curve(&a[1].profile), dram_curve(&c[1].profile));
    }

    #[test]
    fn a_curve_with_mixed_uncertainties_keeps_its_shape() {
        // Synthetic profile: one curve whose points carry different
        // uncertainties, and a group whose members do too.
        let mut v: Value = serde_json::from_str(PROFILE).unwrap();
        let mem = &mut v["measured"]["memory"];
        for (i, q) in mem["dram_bw_by_active_sms"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            q["bw_gbps"]["rel_uncertainty"] = Value::from(0.05 + 0.05 * i as f64);
            q["bw_gbps"]["perturb_group"] = Value::Null;
        }
        mem["dram_latency_ns"]["perturb_group"] = Value::from("shared");
        mem["dram_latency_ns"]["rel_uncertainty"] = Value::from(0.1);
        mem["l2_latency_ns"]["perturb_group"] = Value::from("shared");
        mem["l2_latency_ns"]["rel_uncertainty"] = Value::from(0.3);
        let text = v.to_string();
        let nominal = Profile::from_json(&text).unwrap();
        let widest = 0.05 + 0.05 * (nominal.measured.memory.dram_bw_by_active_sms.len() - 1) as f64;
        let mut largest: f64 = 0.0;
        for d in draws(&text, 32, 17).unwrap() {
            let ratios: Vec<f64> = dram_curve(&d.profile)
                .iter()
                .zip(dram_curve(&nominal))
                .map(|(a, b)| a / b)
                .collect();
            assert!(
                ratios.iter().all(|r| (r - ratios[0]).abs() < 1e-12),
                "{ratios:?}"
            );
            assert!((ratios[0] - 1.0).abs() <= widest + 1e-12);
            largest = largest.max((ratios[0] - 1.0).abs());
            let m = &d.profile.measured.memory;
            let a = m.dram_latency_ns.value / nominal.measured.memory.dram_latency_ns.value;
            let b = m.l2_latency_ns.value / nominal.measured.memory.l2_latency_ns.value;
            assert!((a - b).abs() < 1e-12 && (a - 1.0).abs() <= 0.3 + 1e-12);
        }
        // The range is the widest point's, not the first point's.
        assert!(largest > 0.05 + 0.05, "{largest}");
    }

    #[test]
    fn latin_hypercube_covers_every_stratum_once() {
        let n = 32;
        let s = latin_hypercube(n, 3, &mut Rng(5));
        for d in 0..3 {
            let mut hit = vec![false; n];
            for x in &s {
                hit[(x[d] * n as f64) as usize] = true;
            }
            assert!(hit.iter().all(|&h| h));
        }
    }

    #[test]
    fn specs_are_not_perturbed() {
        for d in draws(PROFILE, 4, 17).unwrap() {
            assert_eq!(d.profile.specs.sm_count, 46);
            assert_eq!(d.profile.specs.l2_size_bytes, 37748736);
        }
    }

    #[test]
    fn cp_async_l2_curve_shares_l2_bw_group_without_shifting_draws() {
        let base_draws = draws(PROFILE, 8, 17).unwrap();
        assert!(base_draws[0]
            .profile
            .measured
            .memory
            .l2_bw_by_active_sms_cp_async
            .is_empty());

        let mut v: Value = serde_json::from_str(PROFILE).unwrap();
        let cp_curve = v["measured"]["memory"]["l2_bw_by_active_sms"].clone();
        v["measured"]["memory"]["l2_bw_by_active_sms_cp_async"] = cp_curve;
        let with_cp = draws(&v.to_string(), 8, 17).unwrap();

        for (a, b) in base_draws.iter().zip(&with_cp) {
            assert_eq!(dram_curve(&a.profile), dram_curve(&b.profile));
            assert_eq!(a.assumed_spill_scale, b.assumed_spill_scale);
            let l2_ld = &b.profile.measured.memory.l2_bw_by_active_sms;
            let l2_cp = &b.profile.measured.memory.l2_bw_by_active_sms_cp_async;
            assert_eq!(l2_ld.len(), l2_cp.len());
            for (p, q) in l2_ld.iter().zip(l2_cp) {
                assert_eq!(p.active_sms, q.active_sms);
                assert!((p.bw_gbps.value - q.bw_gbps.value).abs() < 1e-12);
            }
        }
    }
}
