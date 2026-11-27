# GB10 spill-latency at 4 KiB frames: does the level move with the local memory allocation? (plan, diagnostic)

Status: registered before any run (`68223cd`), revised before any run after the L2 buffer experiment (`v0.3-gb10-first-capture-l2-diagnosis.md`, section 3, result `15ec340`): neither placement, session state nor compression was supported there, and buffers did not keep their level when reread seconds later. As this plan required, it now separates a level that moves with the allocation from one that moves with time. Diagnostic only; no profile or rule changes. The code (`vgpu_bench --mode spill-alloc-diag`, `microbench/spill_alloc.py`) stays on the branch `s8/diag-spill-alloc` (HEAD `e264c84`, with the registration `68223cd`, the revision `060cfc7` and the result `f8e0ff8`) and is not part of the suite.

## Question

In the first GB10 0.23.0 capture (main `42e039d`), the eight 4 KiB `l2_spill` points with reloads (serial and batched, 8 to 128 reloads) failed rule 3: their round medians spread 24% to 31%, every point in the same order of rounds (round 6 highest, rounds 5 and 7 lowest), while the same frames' points without reloads and the smaller frames did not move. The level is set per process. The L2 buffer experiment tests, for the capacity sweep, whether such a level follows a buffer's physical placement, the session, or compression. This is the matching test for the spill frames: the local memory a 4 KiB frame lives in (512 KiB per SM) is allocated by the driver per context, so a process gets one placement for all its spill launches.

## Design

- Points: the 10 spill points at 4 KiB frames (serial and batched, 0, 8, 32, 68 and 128 reloads) and, as a control in L1, the batched 512 B points at 0 and 128 reloads, each at its frozen steps from the `42e039d` freeze record, 15 samples, 3 warm-ups, the SM clock before and after every point.
- Each process reads the 12 points in twelve blocks, each block reading every point twice in a row. Six blocks start a new local memory allocation: the first launch's allocation (0); two after raising the stack limit by 1 KiB each time, which makes the driver allocate local memory anew (1, 2); a new context after `cudaDeviceReset` (3); two more raises (4, 5). Each is followed by a block in the same allocation with nothing changed, one block's time later, the same interval that separates two allocations. The stack limit and start time of every block are recorded.
- Three processes in one session on Spark 1, under the GPU lock, after the 10 s quiet-host gate, each process traced and host-checked.

## Quantities

For a process, the ratio of a point's median in one block and visit to that point's median over the process's 24 readings. A block's level is the median of these ratios over the eight 4 KiB points with reloads, averaged over its two visits; its control level the same over the two 4 KiB points without reloads. An allocation's level is that of its first block; it is stable when that block's two visits differ by at most 0.03. Its drift is the difference between the levels of its second block and its first. The between-process spread of a point is the span of its three process medians over their median.

## Frozen reading

- **Placement:** in at least 2 of the 3 processes the six allocation levels span at least 0.12 and the control levels at most 0.03; at least 15 of the 18 allocations are stable; and at least 15 of the 18 drifts are at most 0.03 (the level stays put while the allocation does).
- **Drift:** placement is not met and, in at least 2 of the 3 processes, at least 3 of the 6 drifts exceed 0.03: the level moves within one allocation over time.
- **Context:** in at least 2 of the 3 processes the levels within each context (allocations 0 to 2 and 3 to 5) span at most 0.03 and the two contexts' mean levels differ by at least 0.08. This reading cannot tell a per-context state from stack-limit raises that do not move the local memory.
- **Session:** in every process the six levels span at most 0.03, and at least 5 of the 8 points spread more than 10% between processes.
- **Not reproduced:** every process's levels span at most 0.05 and no point spreads more than 5% between processes; nothing is concluded.
- Otherwise **neither supported**, with the spans reported. A failed host check voids the run. Readings are checked in the order not reproduced, placement, drift, context, session.

The spill kernel fills its frame with a short regular pattern, so a compression effect cannot be separated here. The L2 buffer experiment did not support compression, so the fill is not varied here.

## Result

Captured 2026-11-27 05:36:18 to 05:38:36 UTC on Spark 1 under the GPU lock, after the other sessions released it; binary `13f7b1d1...` built from `060cfc7`, freeze record `59984726...` (the `42e039d` freeze); `spill-alloc-1.json` `1ef1e909...`. All three processes passed the host check (busiest second 5.6% to 5.9%); each took about 11 s.

Verdict by the frozen reading: **placement: the level moves with the local memory allocation.**

| Process | Allocation levels, 4 KiB with reloads (allocations 0 to 5) | Span | Control span (4 KiB, no reloads) | Largest same-allocation drift |
|---|---|---|---|---|
| 1 | 1.12, 0.91, 0.98, 0.98, 1.01, 1.26 | 0.35 | 0.004 | 0.007 |
| 2 | 0.90, 0.91, 1.02, 0.98, 1.14, 1.05 | 0.23 | 0.004 | 0.007 |
| 3 | 1.16, 0.97, 0.98, 1.20, 0.97, 1.02 | 0.23 | 0.006 | 0.004 |

- All 18 allocations are stable (their two visits within 0.03) and all 18 same-allocation rereads, one block (about 0.9 s) later, stay within 0.008 of the first reading: an allocation keeps its level, and a new allocation, by a stack-limit raise or a new context, draws a new one. The eight points with reloads move together; the points without reloads do not move (spans at most 0.006).
- The levels do not follow the context: within a context they span 0.12 to 0.28, and the two contexts' means differ by 0.02 to 0.11.
- Between processes the eight points spread 6.7% to 9.1%, less than the 24% to 31% of the capture's seven processes; three processes sample the distribution less widely.
- The stack limit read before the first block was the default 1 KiB; the runtime raised it to 4 KiB for the first 4 KiB frame launch, so allocation 0 is the one the first launch made. The raises then went to 5 and 6 KiB (allocations 1, 2) and, after the reset, again from the default (3 to 5).

This differs from the capacity sweep of the L2 buffer experiment, where a buffer did not keep its level. For the spill frames the level is fixed for the life of the local memory allocation, which a process normally keeps for all its launches: that is why the capture's rounds each had one level. A measurement design that reads the spill points in several allocations per process and takes their median would average over this; any such change is registered before a recapture.

### Post-hoc note: the stack limit itself changes some points

Read after the result, not part of the frozen reading. The batched 512 B control at 128 reloads is not random across allocations: in all three processes it takes 1.74 to 1.77 ms at stack limits of 4 and 5 KiB and 1.28 to 1.29 ms at 6 KiB (allocations 2 and 5), the value the capture measured at the default 1 KiB limit (1.29 to 1.31 ms). The stack limit sets the size of each thread's local memory and so the stride between frames, which changes this L1-resident point by about 35%. A stack-limit raise is therefore not a neutral way to get a new allocation.

The 4 KiB result stands on allocations with the same stack limit: allocations 0 and 3 (4 KiB), 1 and 4 (5 KiB) and 2 and 5 (6 KiB) differ by 0.00 to 0.28 in level across the three processes, so a new allocation at the same stride still draws a new level. A measurement design built on this should take new allocations by a new context at the default stack limit, not by raising the limit.
