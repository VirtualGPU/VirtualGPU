# v0.3 holdout amendment for near-L2 reporting

This amendment follows the sealing of both GB10 cohorts and applies
before either RTX 4070 cohort is collected. It does not change the
published manifests or capture policies. The registered criteria,
eligibility rules, scoring and four cohort definitions stay in force.
The amendment makes no retroactive change to the criteria for the
sealed GB10 data.

## Descriptive reporting

Flag these cases by geometry before opening labels. The figures are
the nominal FP16 input footprint for A+B in GEMM and K+V in attention.
They describe shape size. Actual L2 residency depends on the
configuration.

| Device | Family | Case ID | Footprint (MiB) |
| --- | --- | --- | ---: |
| RTX 4070 | GEMM | `v03-rtx4070-mask-mnk-small` | 32.3 |
| RTX 4070 | GEMM | `v03-rtx4070-decode-small` | 38.1 |
| RTX 4070 | Attention | `b2-q16-kv16-s2688-d128-full` | 42.0 |
| RTX 4070 | Attention | `b1-q24-kv24-s3840-d128-causal` | 45.0 |
| GB10 | Attention | `b2-q12-kv12-s2176-d128-full` | 25.5 |
| GB10 | Attention | `b2-q32-kv8-s2563-d128-causal` | 20.0 |

Report the flagged cases separately for descriptive purposes. The flags
and the separate results enter no verdict. Each case retains its
original eligibility and treatment in the registered full-cohort
scoring.

Cross-wave L2 reuse near capacity remains an open question in
[design section 9.3](../docs/design.md#93-cross-wave-l2-reuse-near-capacity)
regardless of the GEMM or attention verdict. The family verdicts still
use the [registered decision criteria](v03-preregistration.md#decisions).
