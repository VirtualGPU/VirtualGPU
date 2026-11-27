# GB10 spill-latency at 4 KiB frames: does the level move with the local memory allocation? (plan, diagnostic)

Status: registered before any run; waits for the result of the L2 buffer experiment (`v0.3-gb10-first-capture-l2-diagnosis.md`, section 3) and the lead's approval. Diagnostic only; no profile or rule changes. Code on the branch `s8/diag-spill-alloc` (`vgpu_bench --mode spill-alloc-diag`, `microbench/spill_alloc.py`).

## Question

In the first GB10 0.23.0 capture (main `42e039d`), the eight 4 KiB `l2_spill` points with reloads (serial and batched, 8 to 128 reloads) failed rule 3: their round medians spread 24% to 31%, every point in the same order of rounds (round 6 highest, rounds 5 and 7 lowest), while the same frames' points without reloads and the smaller frames did not move. The level is set per process. The L2 buffer experiment tests, for the capacity sweep, whether such a level follows a buffer's physical placement, the session, or compression. This is the matching test for the spill frames: the local memory a 4 KiB frame lives in (512 KiB per SM) is allocated by the driver per context, so a process gets one placement for all its spill launches.

## Design

- Points: the 10 spill points at 4 KiB frames (serial and batched, 0, 8, 32, 68 and 128 reloads) and, as a control in L1, the batched 512 B points at 0 and 128 reloads, each at its frozen steps from the `42e039d` freeze record, 15 samples, 3 warm-ups, the SM clock before and after every point.
- Each process reads the 12 points in six local memory allocations, each twice in a row: the first launch's allocation (0); two allocations after raising the stack limit by 1 KiB each time, which makes the driver allocate local memory anew (1, 2); a new context after `cudaDeviceReset` (3); two more raises (4, 5). The stack limit of each allocation is recorded.
- Three processes in one session on Spark 1, under the GPU lock, after the 10 s quiet-host gate, each process traced and host-checked.

## Quantities

For a process, the ratio of a point's median in one allocation and visit to that point's median over the process's 12 readings. An allocation's level is the median of these ratios over the eight 4 KiB points with reloads, averaged over its two visits; its control level the same over the two 4 KiB points without reloads. An allocation is stable when its two visits' levels differ by at most 0.03. The between-process spread of a point is the span of its three process medians over their median.

## Frozen reading

- **Placement:** in at least 2 of the 3 processes the six levels span at least 0.12 and the control levels at most 0.03, and at least 15 of the 18 allocations are stable.
- **Context:** in at least 2 of the 3 processes the levels within each context (allocations 0 to 2 and 3 to 5) span at most 0.03 and the two contexts' mean levels differ by at least 0.08. This reading cannot tell a per-context state from stack-limit raises that do not move the local memory.
- **Session:** in every process the six levels span at most 0.03, and at least 5 of the 8 points spread more than 10% between processes.
- **Not reproduced:** every process's levels span at most 0.05 and no point spreads more than 5% between processes; nothing is concluded.
- Otherwise **neither supported**, with the spans reported. A failed host check voids the run. Readings are checked in the order not reproduced, placement, context, session.

The spill kernel fills its frame with a short regular pattern, so a compression effect cannot be separated here. If the L2 buffer experiment supports compression, or rejects placement, this plan is revised before any run.
