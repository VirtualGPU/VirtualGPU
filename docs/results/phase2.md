# Phase 2 summary: RTX 4070 to DGX Spark

Phase 2 tested whether the frozen VirtualGPU model could rank Triton GEMM
configurations on DGX Spark (GB10) using a hardware profile measured on that
device. The primary cohort reused nine RTX 4070 development shapes. A separate
cohort reused nine workload-holdout shapes, testing transfer across both
devices and workloads.

## Protocol

- The core, frontend, and Triton 3.9.0 stayed frozen at `phase1-final`
  (`129a6f5`). VirtualGPU used a GB10 profile built from isolated hardware
  microbenchmarks. The NeuSight-style baseline reused its frozen RTX 4070
  development fit. Roofline + wave used the GB10 profile.
- The primary cohort had 255 configurations. The secondary cohort had 324.
  The GB10-only tensor memory accelerator (TMA) diagnostic had three. Each
  case used three sessions and ten randomized blocks per session, with warm
  and clean-cold L2 conditions and a 128 MiB read-only eviction buffer.
- The prediction archive was published before GB10 GEMM timing. Before
  scoring, all 57 sealed report digests and the frozen prediction archive
  were verified. All configurations passed AOT/JIT matching. The primary
  cohort alone determines the decision under [design section 8](../design.md#8-validation).
  [Preregistration](../../validation/phase2-preregistration.md),
  [formal report](../../validation/phase2-gb10-report.md).

### Thermal stability amendments

1. The original gate required at least 60 seconds of untimed warm-up and a
   temperature range of at most 2 C over the previous 30 seconds. Ten
   case-sessions passed under that implementation: the nine primary cases
   and secondary `prefill-256`, all in session 1. Secondary `prefill-1024`
   hit the five-minute implementation timeout twice. An in-flight
   `prefill-2048` attempt also hit it. All three stopped before timing.
2. Commit `42689c8` raised the implementation timeout to 20 minutes and
   kept the gate. The next `prefill-1024` attempt still stopped before
   timing. Its saved 1 Hz trace had a minimum sampled 30-second range of
   3 C after warm-up, even though the temperature later fluctuated around
   a steady level. Its first pair of adjacent 30-second windows within
   0.5 C of one another ended at second 94. The trace contained temperature
   only, so it cannot establish clock stability or the absence of clock
   events. [Timeout audit](../../validation/phase2-thermal-audit.json),
   [stop record](../../validation/phase2-thermal-stop.md),
   [temperature trace](../../validation/phase2-thermal-prefill-1024.json).
3. Commit `fd5c947` kept the minimum warm-up and 20-minute timeout. It
   required the means of adjacent 30-second temperature windows to differ
   by at most 0.5 C, their SM clock medians to differ by at most 1%, and
   both windows to be clear of thermal or power slowdown. The remaining
   47 case-sessions used this gate. The first ten retained their original
   eligibility. [Trend-gate audit](../../validation/phase2-thermal-trend-audit.json).

Both amendments were published before resumed measurements. Their timed
labels remained sealed throughout the amendments. The changes governed
eligibility to start timing. The frozen profile, model, toolchain,
predictions, scoring rules, and formal thresholds stayed fixed.

## Result

| Cohort and metric | VirtualGPU | NeuSight-style | Roofline + wave |
|---|---:|---:|---:|
| Primary Kendall tau-b, warm / clean-cold | **0.446 / 0.597** | 0.311 / 0.501 | 0.213 / 0.247 |
| Primary mean top-5 regret, warm / clean-cold | **1.095% / 0.156%** | 1.331% / 1.593% | 1.946% / 1.022% |
| Secondary Kendall tau-b, warm / clean-cold | **0.762 / 0.749** | 0.432 / -0.194 | 0.523 / 0.519 |
| Secondary mean top-5 regret, warm / clean-cold | **0.136% / 0.129%** | 36.838% / 45.199% | 4.124% / 3.891% |

**Formal decision: Fail.** Both cache regimes cleared the ranking and
shortlist non-inferiority criteria. Clean-cold passed the absolute shortlist
criterion. Warm exceeded its 1% ceiling by 0.095 percentage points. The
secondary and TMA results are descriptive and have no effect on this decision.

The three TMA configurations passed AOT/JIT checks in all sessions. For
requested stages 2, 3, and 4, the hardware medians were 20.29, 20.35, and
22.26 us warm and 24.42, 22.34, and 22.34 us clean-cold.
[Per-case scores and TMA predictions](../../validation/phase2-gb10-report.md).

## Do pairwise verdicts agree with the hardware?

This descriptive metric divides matching directions by every ordered call.
The denominator includes calls on pairs that hardware judged equivalent or
unresolved. VirtualGPU orders a pair when all 33 draws clear the 5% band in
the same direction. The baselines apply the band to nominal predictions.

| Cohort | Method | Ordered calls, warm / clean-cold | Agree, warm / clean-cold | Reversed, warm / clean-cold | Hardware-ordered pairs correctly ordered, warm / clean-cold |
|---|---|---:|---:|---:|---:|
| Primary | VirtualGPU | 1,236 / 1,592 | **95.1% / 95.4%** | 0.2% / 0.1% | 39.0% / 54.3% |
| Primary | NeuSight-style | 3,854 / 3,777 | 51.1% / 50.9% | 20.1% / 15.1% | 65.3% / 68.6% |
| Primary | Roofline + wave | 3,853 / 3,853 | 46.2% / 43.2% | 23.4% / 21.2% | 59.1% / 59.4% |
| Secondary | VirtualGPU | 3,025 / 3,098 | **96.9% / 97.6%** | 0.8% / 0.2% | 64.4% / 66.2% |
| Secondary | NeuSight-style | 5,009 / 4,943 | 53.1% / 41.3% | 28.6% / 40.6% | 58.5% / 44.7% |
| Secondary | Roofline + wave | 4,899 / 4,899 | 69.6% / 70.4% | 13.7% / 13.2% | 74.9% / 75.5% |

## What Phase 2 shows

- The frozen model's high-confidence directions transferred from RTX 4070 to
  GB10, including workloads held out in Phase 1. On both cohorts, hardware
  agreed with more than 95% of its ordered calls.
- VirtualGPU's median ranking exceeded both baselines by the prespecified
  margin in both primary cache regimes. Its directional agreement came
  with fewer ordered calls. On the primary warm regime, it correctly
  ordered 39.0% of the pairs that hardware ordered.
- The frozen RTX 4070 learned fit's transfer error depended on the workload.
  The largest shortlist losses on the secondary cohort occurred on the
  three resident-wave cases.

## What Phase 2 does not show

- A pass on the full primary gate, or a guarantee of a near-optimal warm
  shortlist on GB10.
- Transfer to other accelerators or kernel families, including attention.
  The three TMA cases support only a narrow GEMM diagnostic.
- The cause of residual model errors. Any post-hoc analysis is separate and
  leaves the frozen Phase 1 and Phase 2 decisions unchanged.
