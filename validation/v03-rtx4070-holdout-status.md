# RTX 4070 v0.3 holdout capture status

As of 2026-11-28 UTC, GEMM is complete and attention is partial.

| Cohort | Complete slots | Failed attempts | Pending slots | Audit |
| --- | ---: | ---: | ---: | --- |
| GEMM | 27 of 27 | 0 | 0 | [Complete audit](v03-rtx4070-gemm-seal.json) |
| Attention | 17 of 24 | 1 | 7 | [Partial audit](v03-rtx4070-attention-partial-seal.json) |

The release used capture commit
`4ecde308dc086a073f8242c6f35fbb91d96a1161` and the device approval
SHA-256 `65c5ecfbc4a6c49842e035537cb51352fc4daf41e28ff44f41de900d87f3bd10`.
The RTX 4070 capture policy SHA-256 is
`3bb13b7307eabb803465a05cbcd939b78a6252bb2c54f3dc761c069ac2a8ed91`.

The [GEMM receipts](v03-receipts/rtx4070-gemm/) cover all 27 slots,
from 2026-11-27 23:35:58 to 2026-11-28 00:25:10 UTC. The
[attention receipts](v03-receipts/rtx4070-attention/) include 17
completed slots, from 00:31:41 to 00:59:09 UTC, and one failed
attempt. All 44 completed attempts have unopened local reports
listed in the [mode-zero inventory](v03-receipts/rtx4070-seal-modes.txt).
The audits used receipt contents and file metadata.

Attention stopped after the desktop gate timed out for
[`b2-q20-kv20-s1217-d64-causal`, session 3, attempt 1](v03-receipts/rtx4070-attention/b2-q20-kv20-s1217-d64-causal-session-3.json).
Its receipt SHA-256 is
`9406dcf4b7e2a7488bfe416b6fa9765439c38a87319859798d9556cfedcd335b`.
The attempt left a `protocol_failed` receipt and host sidecar.
The timed session had not started, so the attempt has no sealed
report. A separate retry approval is pending for that slot.

## Pending attention slots

| Case | Session | Next attempt | Status |
| --- | ---: | ---: | --- |
| `b1-q16-kv16-s640-d64-full` | 1 | 1 | Original pre-timing stop recorded in [the amendment](v03-rtx4070-attention-pre-timing-amendment.md). [Separate written approval](v03-approvals/pre-timing/rtx4070/b1-q16-kv16-s640-d64-full-session-1-attempt-1.json) is on main. Awaiting a new desktop window. |
| `b2-q20-kv20-s1217-d64-causal` | 3 | 2 | Desktop-gate timeout. Awaiting a new desktop window and separate retry approval. |
| `b1-q24-kv24-s3840-d128-causal` | 3 | 1 | Awaiting a new desktop window. |
| `b2-q24-kv6-s1347-d64-full` | 3 | 1 | Awaiting a new desktop window. |
| `b1-q32-kv4-s3328-d128-full` | 3 | 1 | Awaiting a new desktop window. |
| `b1-q32-kv8-s1795-d64-causal` | 3 | 1 | Awaiting a new desktop window. |
| `b2-q24-kv6-s2305-d128-causal` | 3 | 1 | Awaiting a new desktop window. |
