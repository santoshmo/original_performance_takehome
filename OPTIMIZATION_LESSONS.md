# Optimization Lessons From `original_performance_takehome`

This document is a technical handoff for the optimization work done across the
`original_performance_takehome` branches and worktrees. It is written for a
future agent or engineer who needs to understand what was tried, what actually
worked, why several attractive ideas failed, and where the next serious attempt
should begin.

The most important constraint never changed:

```text
Do not modify tests/.
```

All meaningful cycle measurements should be validated with:

```bash
./.venv/bin/python tests/submission_tests.py
```

## Executive Summary

The best verified result after the latest continuation is:

```text
cycles: 1126
branch: final-level4-cache-prototype
local worktree: original_performance_takehome
validation: ./.tools/python312/bin/python3 tests/submission_tests.py
```

This improved the prior `1153` selector-ring baseline by changing final node
ownership, not by adding more context scratch. The final round's level-4 node is
prefetched after round 14 into the per-block `idx_vec` path storage. That path
state is dead because final index updates are dropped, and unlike `ctx["node"]`
it is not reused by the next active context group.

The latest `1126` result adds setup-only improvements. The shallow-node cache
is initialized with two streaming address registers over contiguous tree nodes,
and many small constants are synthesized with ALU/flow operations instead of
loaded. The body schedule is unchanged at 1113 cycles; setup drops from 26 to
13 cycles.

Winning continuation defaults:

```text
simd_prefetch_final_node_into_idx = True
simd_pair_stream_node_setup = True
simd_predrain_group_rotation = 6
round 5 hash scalarization:
  h1 = []
  h2 = [1]
  combine = []
pre-drain hash scalarization:
  h1 = [1]
  h2 = [5]
  combine = [1]
tail_reschedule_window = 25
tail_rounds = 1
final_store_bonus = 40
final_hash_bonus = 0
```

The previous best verified result was:

```text
cycles: 1153
branch: selector-ring-1130-1150
private fork branch: origin/selector-ring-1130-1150
local worktree: original_performance_takehome_selector1130
```

The exploratory branch, `final-level4-cache-prototype`, previously also
defaulted to `1153` after reverting failed level-4 cache and prefetch
experiments. It now contains the `1126` path-vector final handoff and setup
constant-synthesis improvements plus the extra reports and allocator/lifetime
tools.

The most important lesson is:

```text
The 1153-cycle kernel wins by preserving broad active-context overlap.
Most attempts to attack the final-round load tail failed because they extended
context lifetimes and destroyed that overlap.
```

At `1153`, the obvious bottleneck signals were tempting:

```text
load_active ~= 1112
VALU_active ~= 1146
final gather ~= 1115
final hash ~= 1125
final store ~= 1126
```

But final loads are not independently reducible in the current state layout.
Any attempt to move or cache final node values must store them somewhere. In the
current kernel, every plausible full-vector storage location is either live or
will be reused by a later block before the earlier block reaches final hash.

The path below `1100` likely requires a non-local representation or ownership
change, not another final-tail tweak.

The `1139` improvement supports that thesis: the successful change was not a
final cache or a context-local prefetch, but a small ownership change that moved
final node state into storage owned by the block rather than by the reusable
context. The later `1126` improvements are separate: they reduce setup cost,
not the body tail.

## Repository And Branch Map

The private fork is:

```text
git@github.com:santoshmo/original_performance_takehome.git
```

The local grouped worktree directory is:

```text
/Users/santoshm/Documents/original_performance_takehome_repos
```

Important branches:

```text
main
  Mirror of upstream public repo baseline.

blog_clean_rewrite
  Earlier blog-inspired rewrite and shallow-cache exploration.

scheduler_rewrite
  Active-context scheduling checkpoint.

fresh-rewrite
  Early fresh rewrite branch.

fresh_start2
  Another fresh rewrite checkpoint.

fresh-bitstream-selectors
  First carried-selector / bitstream experiment.
  Verified around 1178 cycles.

selector-ring-1130-1150
  Clean best implementation.
  Verified 1153 cycles.

fresh-under-1100
  Under-1100 planning and region-aware compiler experiments.

final-level4-cache-prototype
  Current exploratory branch.
  Defaults to 1126, includes later diagnostic tools and reverted failed cache
  lessons.
```

## Current Best Kernel Metrics

For the current `1126` continuation default:

```text
cycles: 1126
scratch: 1522 / 1536
free scratch: 14 words
load active: 1098
VALU active: 1114
ALU active: 1116
flow active: 708
last_final_gather: 1095
last_final_hash: 1105
last_final_store: 1113
```

Important clarification:

```text
last_final_hash=1105 is not the benchmark score.
The simulator-reported score is still 1126 cycles.
```

`last_final_hash` and `last_final_store` are internal schedule milestones used
to reason about the tail.

## Architecture And Workload

The simulator is a VLIW-like machine with separate execution engines. The exact
engine balance matters more than any one operation count:

```text
ALU:   scalar arithmetic and bit operations
VALU:  vector operations, VLEN=8
Load:  scalar/vector memory loads
Store: stores
Flow:  select/control-style operations
```

The workload is a repeated tree traversal:

```text
for each input:
  for each round:
    choose/load tree node for current path
    value = hash(value xor node)
    update path/index from hash parity
  store final value
```

The hard parts are:

```text
random lane gathers for deeper tree levels
hash pipeline pressure
scratch pressure
active-context overlap
engine slot packing
final value readiness
```

## Performance Timeline

Approximate major milestones:

```text
147734 cycles
  Original slow scalar baseline.

~1240 cycles
  SIMD active-context kernel family.

1218 cycles
  Split final round into its own value-only final drain.

1199 cycles
  Better pre-drain scalar hash policy and tile shape.

1178 cycles
  One-bit selector ring plus pre-drain group rotation.

1170 -> 1163 -> 1160 -> 1157 cycles
  Deeper selector rings and targeted hash/XOR retuning.

1153 cycles
  Three-bit selector ring, selector aliases, scalar XOR level 0, and round-5
  h2 scalarization.

1139 cycles
  Final node prefetch into dead per-block path storage plus pre-drain and tail
  scheduling retuning.

1136 cycles
  Two-stream shallow-node setup that removes per-node offset constants from
  initialization.

1135 -> 1130 cycles
  Removed duplicate setup zero/vlen loads and synthesized small scalar constants
  such as 2, 3, 4, 5, 9, 12, 16, 19, 33, and 4097.

1128 -> 1126 cycles
  Synthesized level-base constants 22..1030 from a short ALU recurrence,
  reducing setup to 13 cycles while leaving the 1113-cycle body unchanged.
```

The improvements became progressively less about "remove obvious work" and more
about preserving several constraints simultaneously:

```text
do not exceed scratch
do not reduce active-context overlap
do not trade VALU bottleneck for ALU tail
do not trade load bottleneck for flow tail
do not delay final hash/store readiness
```

## The Winning Design

The best kernel is a SIMD active-context IR kernel.

Core features:

```text
SIMD batch state in scratch
active context pool
group_size = 16
offset-state traversal
cached shallow tree nodes 0..14
depth-3 vselect selection tree
hash multiply_add fusion
selective scalarization
split final value-only drain
dropped final index update
explicit IR read/write sets
zero-latency WAR modeling
ASAP-style scheduler
three-bit selector ring
```

Important default policies:

```text
simd_body_tile_boundaries = [0, 11, 15]
simd_predrain_order_rounds = [11]
simd_predrain_group_order = rotate
simd_predrain_group_rotation = 6
simd_selector_ring_depth = 3
simd_prefetch_final_node_into_idx = True
simd_pair_stream_node_setup = True
scalar_xor_levels = {0,1,2,3,4,5,6,7,8,10}
```

Hash scalarization defaults include:

```text
round 5:
  h1 = []
  h2 = [1]
  combine = []

pre-drain:
  h1 = [1]
  h2 = [5]
  combine = [1]
```

## IR And Scheduling Lessons

The explicit IR was essential. It made it possible to reason about scratch,
aliases, dependencies, and tail milestones.

Each `KernelOp` carries:

```text
engine
slot
tag
round
level
group
region
contributes_value
contributes_index
droppable_final
reads
writes
read_roles
write_roles
```

Dependencies include:

```text
RAW: read after write, latency 1
WAW: write after write, latency 1
WAR: write after read, latency 0
MEM: optional memory-order edge
```

The zero-latency WAR detail matters because this machine permits same-cycle
read-before-write behavior. Treating WAR as latency 1 artificially serializes
valid bundles.

Generic critical-path scheduling was not a reliable improvement. A simple ASAP
packer with accurate read/write sets often performed better because it preserved
the beneficial greedy packing behavior.

The scheduler lesson:

```text
Correct dependencies are mandatory, but "more sophisticated" scheduling is not
automatically better. Tail shape, engine packing, and generation order interact.
```

## Offset-State Traversal

Offset-state traversal replaced heap-index-heavy logic with offset-within-level
state.

This made deep gather address generation simpler:

```text
forest_values_p + level_base + offset
```

It also eliminated stale subtract/base-fixup work. This mattered early, but by
the `1153` result the remaining gap was not dominated by simple address math.

## Shallow Node Cache

Nodes `0..14` are cached as vector broadcasts:

```text
v_node_0
...
v_node_14
```

They cover levels:

```text
level 0: 1 node
level 1: 2 nodes
level 2: 4 nodes
level 3: 8 nodes
```

For these shallow levels, a vselect tree is a good trade:

```text
few possible nodes
no scalar lane loads
flow cost is bounded
```

For level 4, the same idea becomes much less attractive because there are 16
possible nodes and a full binary tree costs 15 vselects per vector block.

### Setup Compression

The shallow-node cache setup originally loaded one scalar offset constant per
cached node and computed each node address independently. The first setup win
uses two address registers:

```text
tmp_init  = forest_values_p + 0
tmp_init2 = forest_values_p + 1
```

It then streams through nodes `0,2,4,...` and `1,3,5,...`, incrementing each
address by two after every pair. This removes several tiny setup constants,
reduces scratch from `1531` to `1525`, and cuts setup from `26` to `23` cycles.

The later `1126` default keeps that stream and removes more setup loads:

```text
offset = forest_values_p - 7
inp_indices_p = forest_values_p + 2047
inp_values_p = forest_values_p + 2303
idx reset uses idx_vec ^ idx_vec instead of v_zero
small constants are synthesized from 1, 2, 4, 16, and 32
level bases 22..1030 are synthesized by recurrence
```

This cuts setup to `13` cycles. The body remains `1113` cycles, so the setup
work is no longer the main bottleneck.

## Depth-3 Selection

Depth-3 selection among nodes `7..14` uses a binary vselect tree:

```text
4 pair selects
2 quad selects
1 final select
```

This was important, but it introduced scratch hazards. The tree needs temporary
vectors, and some of those temps are aliased with selector-ring storage in the
winning implementation.

The main correctness rule:

```text
Never overwrite a selector vector before its final consumer in that tree.
```

Several incorrect variants failed because they reused `tmp2` or `tmp3` while
those locations still held carried selector bits.

The working solution makes reusable selector-backed temporaries explicit and
only writes into a selector-backed temp after the corresponding bit is dead.

## Hash Pipeline

The hash stages are a major source of VALU pressure. Some stage patterns can be
fused as vector `multiply_add`, which is a large win.

Scalarization is useful only when carefully targeted:

```text
vector op:
  consumes VALU
  preserves ALU capacity

scalarized per-lane op:
  reduces VALU
  consumes ALU
  may create ALU tail
```

Broad scalarization frequently reduced VALU slots while making cycles worse.
The best policies were round-specific and region-specific.

The successful late scalarization was small:

```text
round 5 h2 stage 3 scalarized
scalar XOR includes level 0
pre-drain hash policy adjusted
```

## Split Final Drain

Splitting round 15 into a separate value-only final region was one of the
largest post-1240 improvements.

Final round needs only:

```text
gather/select final node
xor into value
hash final value
store final value
```

It does not need future path/index updates. Dropping final index work and
separating final generation broke an old local optimum around 1240 cycles.

After this win, however, final-only tuning produced diminishing returns.

## Selector Ring

The selector ring carries recent hash parity bits so shallow levels can consume
branch decisions directly instead of recomputing masks from `idx_vec`.

Conceptually:

```text
after hash:
  parity_vec = value & 1
  selector[round % depth] = parity_vec

at shallow levels:
  selector_bit(age) supplies branch condition
```

Depth 1:

```text
removes path & 1 at levels 1..3
helped reach ~1178 after retuning
```

Depth 2:

```text
also removes path & 2 at levels 2..3
fit by aliasing selector1 with tmp3
```

Depth 3:

```text
also removes path & 4 at level 3
fit by aliasing selector2 with tmp2
requires hash h2 temporary to move to node
```

Winning aliases:

```text
selector1 -> select_tmp1 / tmp3
selector2 -> select_tmp0 / tmp2
hash h2 tmp -> node
```

The selector ring is the last major successful representation change in the
current line of work.

## Scratch Pressure

Scratch is nearly exhausted:

```text
SCRATCH_SIZE = 1536
current scratch = 1531
free = 5 words
one full context vector array = 16 contexts * 8 lanes = 128 words
```

Current context vector arrays:

```text
node_tmp
hash_tmp
select_tmp0 / selector2
select_tmp1 / selector1
selector_latest / selector0
```

Adding a new full vector array at `group_size=16` requires freeing about 128
words. There is no trivial way to do that.

## Why Final-Only Load Attacks Failed

At `1153`, final gather/load looked like the next bottleneck:

```text
load_active ~= 1112
final_drain:gather ~= 1115
final_drain:hash ~= 1125
```

This led to several local final-round experiments. They all failed for the same
structural reason: moving or replacing final loads extended lifetimes or added
too much work in another engine.

### Full Level-4 Cache

Hypothesis:

```text
Final round is level 4.
Level 4 has 16 nodes.
Cache nodes 15..30 and select with vselects.
```

Result:

```text
load pressure improved
flow/select work exploded
final select became the tail
cycles regressed badly
```

The 16-way tree costs 15 vselects per vector block. Flow had slack, but not
that much slack.

### Partial Level-4 Cache

Hypothesis:

```text
Cache only 8 of 16 level-4 nodes.
Fallback gather for the other half.
```

Result:

```text
correct but slower
fallback gathers still required
added select work without removing enough loads
```

Because loads are not conditionally skipped in a useful way, the partial cache
did not attack the real load count.

### Final Gather Prefetch

Hypothesis:

```text
issue final gather at end of round 14
store final node in ctx["node"]
skip final gather in round 15
```

Result:

```text
correct but slower
cycles: 1153 -> 1247
final hash/store moved about 94 cycles later
```

Why:

```text
ctx["node"] had to remain live until final hash
the context could not be reused by the next block group
overlap collapsed
```

Explicit liveness guards confirmed this was a real lifetime/ownership problem,
not a correctness accident.

## Region Liveness Findings

`tools/region_liveness_report.py` asks:

```text
Is any full context vector array free between round-14 prefetch point and
round-15 hash?
```

Answer:

```text
No.
```

Blocked arrays:

```text
hash_tmp:        blocked 32/32 windows
node_tmp:        blocked 32/32 windows
select_tmp0:     blocked 16/32 windows
select_tmp1:     blocked 16/32 windows
selector_latest: blocked 16/32 windows
```

It also showed:

```text
all 16 contexts are reused by blocks 16..31
before blocks 0..15 reach final hash
```

This is the central reason final prefetch into existing context storage fails.

## Region-Local Scratch Allocator

We implemented:

```text
region_local_compact_scratch=True
```

It:

```text
1. analyzes IR allocation intervals
2. colors non-overlapping vector allocations
3. rewrites IR addresses
4. reschedules with alias dependencies visible
```

Result:

```text
eligible context vector allocations: 64
colors needed: 64
remapped allocations: 0
cycles: 1153
scratch: 1531
```

This proves that whole-vector allocation lifetimes do not expose reusable
scratch in the current design.

## Logical Lifetime Findings

`tools/logical_lifetime_report.py` asks whether arrays are blocked by one
long-lived logical value or by many short values.

The result is subtle:

```text
individual logical values are mostly short-lived
but prefetch windows are blocked by same-context reuse
```

First blockers:

```text
hash_tmp:
  blocked 32/32
  first blockers all same_context_reuse

node_tmp:
  blocked 32/32
  first blockers all same_context_reuse

select_tmp0:
  blocked 16/32
  first blockers all same_context_reuse

select_tmp1:
  blocked 16/32
  first blockers all same_context_reuse

selector_latest:
  blocked 16/32
  first blockers all same_context_reuse
```

Representative max logical generation durations:

```text
hash_tmp:        17 ops
node_tmp:        16 ops
select_tmp0:     99 ops
select_tmp1:     47 ops
selector_latest: 78 ops
```

Interpretation:

```text
Finer-grained splitting may help local temp pressure, but it does not create a
safe final prefetch slot unless context ownership changes.
```

The issue is not one stubborn long-lived value. It is that the same context is
reused by another block before the earlier block reaches final hash.

## What Actually Worked

Successful changes shared a pattern:

```text
remove work without extending lifetimes
shift work to ALU only where ALU has slack
avoid recomputing state by changing representation
fit scratch by aliasing values proven dead
preserve active-context overlap
```

Examples:

```text
SIMD active contexts
offset-state traversal
cached shallow nodes
depth-3 select tree
multiply_add hash fusion
split final drain
drop final index update
selector ring
selector aliases
hash h2 temp via node
targeted hash scalarization
scalar XOR level 0
final node prefetch into dead path storage
two-stream shallow-node setup
setup constant synthesis
```

## What Did Not Work

Repeatedly bad patterns:

```text
generic critical-path scheduler
broad tail rescheduling
final-only wavefront generation
interleaved final stores
final-only hash scalarization
full level-4 cache
partial level-4 cache
final gather prefetch into ctx["node"]
offset increments moved from ALU to flow add-immediate
deferred big hash constants into body scheduling
post-1126 pre-drain rotation and tail-weight retuning
lowering group_size to fit scratch
scalarizing depth-3 bit extraction
shared scratch pools that serialize groups
```

Common failure modes:

```text
reduced VALU slots but created ALU tail
reduced load activity but created flow tail
freed scratch but serialized useful overlap
moved final work earlier but delayed final hash readiness
```

## Why 1100 Was Not Reached

At `1126`, the kernel is close to several independent lower bounds:

```text
load active: 1098
VALU active: 1114
ALU active: 1116
```

To get below `1100`, reducing one engine is not enough if another engine becomes
the tail. The failed experiments show that the current layout is already near a
local optimum:

```text
final load attacks -> flow or overlap failure
VALU scalarization -> ALU tail
scratch reuse -> context reuse collision
```

The remaining gap probably needs a representation change that alters ownership,
not just scheduling.

## Most Promising Future Directions

### 1. Change Context Ownership

The current active context owns:

```text
path
value
node temp
hash temp
select temps
selector history
```

The failure mode is that the same context is reused before final hash. A future
design may need separate ownership:

```text
body context:
  owns traversal state for rounds 0..14

final context / handoff pool:
  owns value/path/final-node state needed for final hash/store
```

A possible experiment:

```text
final_handoff_pool_size = k

after round 14:
  hand off final-needed state into a smaller final pool
  release body context for reuse
  finish final hash/store from final pool
```

The hard part is scratch. The handoff pool must be much smaller than a full
extra set of context arrays or must alias storage that is truly globally dead.

### 2. Reduce Deep Gathers Across Multiple Rounds

Final-only level-4 cache failed, but load pressure exists across levels 4..10.
A more global memory-layout transform may be needed:

```text
tile tree nodes into scratch
group lanes/blocks by shared prefixes
cache windows of deep nodes
convert some gathers into vector-friendly access
```

This is risky because path distribution may be random and setup cost may erase
the win.

### 3. More Radical Traversal State

The selector ring was successful because it stopped using `idx_vec` as the only
source of truth for shallow branch bits.

A more radical representation might separate:

```text
deep offset state
shallow selector history
pending branch bit stream
final-only path state
```

The goal would be to remove more VALU work without pushing pressure to ALU or
scratch.

### 4. Search With Critical-Chain Awareness

Future search should keep candidates on a Pareto frontier across:

```text
cycles
VALU active
ALU active
load active
flow active
last_final_gather
last_final_hash
last_final_store
scratch
context reuse hazards
```

Do not rank only by total cycles. Some candidates reduce a real bottleneck but
need another complementary change before they win.

## Recommended Next Experiment

The most informative next experiment is not another final cache.

Instead, quantify context ownership alternatives:

```text
variant: finalize_group_before_reuse

for each group_start:
  run rounds 0..14
  immediately run final round
  then reuse context for next group_start
```

This likely regresses, but it measures the overlap cost of avoiding context
reuse. If the cost is enormous, then any same-context final handoff is dead.

Then prototype:

```text
variant: final_handoff_pool_size = k

move only final-needed data into a small final pool
release body context
finish final hash/store from handoff pool
```

This is the first direction that directly addresses the real blocker exposed by
the liveness reports.

## Useful Tools

Created or used during the exploration:

```text
tools/phase4_variant_search.py
  Variant evaluator and metric collector.

tools/round14_order_search.py
  Focused pre-drain group-order search.

tools/scratch_liveness_report.py
  Scratch allocation report by role/pool.

tools/tail_causality_trace.py
  Critical-chain trace into final hash/store.

tools/pareto_policy_search.py
  Pareto-preserving policy search.

tools/load_pressure_report.py
  Load pressure and late-load analysis.

tools/region_liveness_report.py
  Full-vector liveness across prefetch/final windows.

tools/logical_lifetime_report.py
  Logical generation lifetimes and blocker classification.
```

Most useful current commands:

```bash
./.venv/bin/python tests/submission_tests.py
./.venv/bin/python tools/region_liveness_report.py
./.venv/bin/python tools/logical_lifetime_report.py
```

## Correctness Hazards

### Selector Storage

Do not overwrite `selector0`, `selector1`, or `selector2` before their last
local consumer.

### Hash H2 Temp

When `selector2` aliases `tmp2`, hash h2 cannot freely use `tmp2`. The winning
depth-3 selector design routes hash h2 through `node`.

### Shared Pools

A shared scratch pool can be correct but still bad if it serializes overlap.
Scratch saved at the cost of context overlap is usually a loss.

### Non-Divisor Group Sizes

Some `group_size` values that do not divide the number of vector blocks caused
incorrectness or misleading measurements. Treat them carefully.

### Final Prefetch

Prefetching into `ctx["node"]` is safe only if the context is not reused before
final hash. In the 1153 schedule, contexts are reused before final hash.

## Final Takeaway

The current `1126` result is not stuck because we failed to tune the last few
cycles of the final tail. The `1139` ownership change proved one narrow final
handoff was still available, and the `1126` setup changes removed startup work.
The remaining body schedule is still constrained because:

```text
load pressure
VALU pressure
ALU pressure
scratch pressure
context overlap
```

are all tightly coupled.

The successful path below `1100` probably needs to answer:

```text
How can final/deep node state be owned independently from body contexts,
without adding a full context-sized scratch pool?
```

Until that ownership problem is solved, most local changes will simply move the
bottleneck from one engine or phase to another.
