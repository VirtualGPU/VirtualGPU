# v0.3 RTX 4070 attention pre-timing interruption

This amendment requests permission to run one published slot:
RTX 4070 attention `b1-q16-kv16-s640-d64-full`, session 1.

On 2026-11-27 the local batch driver logged the attention dispatch at
23:36:08 UTC, after a GEMM audit had reported 1 complete and 26 pending.
The capture process started at approximately 23:36:18 UTC. The
stop snapshot records the process at 26 seconds elapsed at
23:36:44.541008984 UTC. The operator stopped it at that point.

The released RTX 4070 entry starts a fresh desktop gate for every
invocation and requires at least 60 continuous seconds below its
utilization threshold before entering the timed session. The process's
whole lifetime was shorter than that gate. The interrupted invocation
did not persist a desktop-reading log. The gate's readings stay in
memory until it returns, so there is no per-reading log digest for
this interruption.

The external batch loop let child processes share the input stream
containing the slot list. It stopped the GEMM traversal after one
slot, and its driver used the successful partial audit exit code to
enter attention. The local replacement isolates child input, checks
the next unfinished slot against the manifest and requires the
preceding cohort's complete audit before advancing.

| Retained evidence | SHA-256 |
| --- | --- |
| Campaign log, including the partial GEMM audit and attention dispatch | `f90f88994fe7d39740bfbac724e02fcd32d32038756cfbdcdbea148528848a45` |
| Process stop snapshot | `f2daea7ef4b8ce26d7cc809da4600c3497e597614c4017f4417e10ed3c559d05` |
| Local attention start marker | `5e56deee2709d0ae10a2f88deb471577ed9dc64d6a7eea31c0495b505849a5d5` |

The attention slot has the preserved local start marker and no
receipt, report, partial report, host sidecar or thermal timeout
record. This was established from filenames and metadata. The
released entry at `4ecde308dc086a073f8242c6f35fbb91d96a1161`
checks the capture artifact paths and does not consult the local
batch driver's marker. A CPU-only test with the marker in a scratch
directory reached the desktop gate. The local driver blocks this
slot and preserves the original marker pending authorization.

Authorize one run of this slot as attempt 1 after the release lead's
separate approval, with session seed 101 and the same released
capture code, policy and manifests. Preserve the original marker,
campaign log and stop snapshot with the audit. The successful first
GEMM receipt remains part of the existing capture.
