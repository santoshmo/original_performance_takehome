# Final-Only Level-4 Cache Prototype Plan

This plan targets the current `1153` selector-ring design family. The goal is
to prototype whether replacing final-round level-4 scalar gathers with cached
vector selection can reduce load pressure and move the final hash/store earlier.

Do not modify anything under `tests/`.

## Current Motivation

The current best verified result is:

```text
cycles: 1153
load active: ~1112
VALU active: ~1122
final_drain:gather: ~1115
final_drain:hash: ~1125
store_tail:store: ~1126
```

Load-pressure analysis shows the final gather-to-hash window is load-saturated.
The final round is level `4`, which has only `16` possible tree nodes. That
makes final-only level-4 caching the most direct load-pressure experiment.

## Core Hypothesis

Replace final round scalar lane gathers:

```text
8 address ALU ops per vector block
8 scalar loads per vector block
```

with:

```text
16 cached level-4 node vectors
15 vselects per vector block
```

This trades load/ALU pressure for flow pressure. Flow has much more slack than
load near the current tail.

Expected upside:

- fewer final-round scalar loads
- earlier final hash readiness
- lower load active cycles
- possible cycle reduction from `1153`

Main risk:

- 16 cached vectors cost `128` scratch words
- 15 `vselect`s per block may create flow pressure
- selection tree may delay hash more than loads did

## Phase 1: Scratch-Ignoring Prototype

First prove whether the idea has cycle value before solving scratch elegantly.

Add a variant:

```python
cache_final_level4 = True
```

For this first prototype, allow one of:

- temporarily reduce `group_size` enough to fit the 16 vectors
- temporarily disable another optional feature to free scratch
- temporarily run a standalone simulation with larger scratch if easy

The point is not to produce the final implementation. The point is to answer:

```text
Does replacing final level-4 loads with cached select reduce cycles?
```

Validation:

```bash
./.venv/bin/python tests/submission_tests.py
```

Measure:

- cycles
- scratch
- load active
- VALU active
- flow active
- final gather/hash/store cycles

## Phase 2: Level-4 Cache Construction

Level 4 nodes are heap indices:

```text
15..30
```

Build vector broadcasts:

```python
v_l4_0  = broadcast(tree[15])
v_l4_1  = broadcast(tree[16])
...
v_l4_15 = broadcast(tree[30])
```

These should be emitted during setup or immediately before the final region.

Initial simple version:

- allocate 16 persistent vectors
- load scalar node value
- `vbroadcast` to vector

Later optimized version:

- allocate these vectors only in final-region scratch
- alias them over dead context/select scratch

## Phase 3: 16-Way Select Tree

The final round level is `4`, so node selection depends on four path bits:

```text
b0, b1, b2, b3
```

Current selector-ring state carries three bits:

```text
b0, b1, b2
```

The fourth bit can initially be materialized from path:

```text
b3 = path & 8
```

Then select:

```text
pairs:  8 selects using b0
quads:  4 selects using b1
octets: 2 selects using b2
final:  1 select  using b3
```

Total:

```text
15 flow vselects
1 remaining VALU mask for b3
0 final scalar loads
0 final gather address ALU ops
```

This should only apply to final round level 4. Body rounds should stay unchanged
at first.

## Phase 4: Compare Against Baseline

Run baseline vs prototype:

```text
baseline: current 1153 selector-ring kernel
prototype: cache_final_level4=True
```

Report:

- total cycles
- final gather/hash/store timing
- load active cycles
- load slots
- flow active cycles
- VALU slots/active
- ALU slots/active

Success criteria for continuing:

- total cycles improve, or
- final hash/store move earlier without major flow tail, or
- load active drops enough that later scheduling work could plausibly win

Rejection criteria:

- final hash moves later
- flow becomes tail
- scratch workaround destroys overlap
- total cycles regress badly despite fewer loads

## Phase 5: Fit Scratch At Group Size 16

If the scratch-ignoring prototype wins, implement a real scratch plan.

Current context arrays include:

```text
node_tmp
hash_tmp
select_tmp0 / selector2
select_tmp1 / selector1
selector0
```

Final-only level-4 cache needs one full vector array:

```text
16 vectors = 128 words
```

Candidate alias strategies:

### Alias Over Selector Ring

Selector bits are no longer needed after final node selection starts.

Possible plan:

```text
selector0/1/2 scratch -> l4 cache vectors
```

This requires careful ordering:

1. Use selector bits to choose final node.
2. Do not overwrite selector storage before selection finishes.

This is easiest if level-4 cache is built before final region, but selector bits
must still survive into final selection. So full aliasing over selectors may not
work unless cache construction happens after selector use, which defeats the
cache.

### Alias Over Context Temps

After final node selection and xor, some select temps are dead.

But level-4 cache must exist before selection, so it cannot alias temps needed
by the selection tree unless the tree is redesigned.

### Alias Over Top-Node Cache Or Constants

Top-node cache vectors `v_node_0..14` are still needed for levels `0..3`, but
not in final round level `4`.

In a region-local allocator, final level-4 cache could reuse top-node cache
storage after body/pre-drain finishes.

This is the most promising real fit:

```text
top-node cache vectors -> final level-4 cache vectors
```

There are 15 top-node cache vectors (`120` words), almost exactly enough for 15
of the 16 final level-4 vectors. The remaining vector may come from an existing
temp or a compact scalar broadcast.

### Reduce Cache Size

Cache only 8 level-4 vectors and load/select the other half.

This tests whether partial caching gives most of the benefit with half the
scratch:

```python
cache_final_level4_nodes = [15..22]  # or [23..30]
```

Use path high bit to choose cached half vs loaded half.

## Phase 6: Partial Cache Search

If full cache is hard to fit, search partial caches:

```text
cache 4 nodes
cache 8 nodes
cache 12 nodes
cache 16 nodes
```

Candidate node sets:

- first half: `15..22`
- second half: `23..30`
- hot nodes from profiling final path distribution
- nodes selected by group/lane histograms under seed `123`

For each candidate:

- count final load reduction
- measure cycle effect
- check if final hash moves earlier

## Phase 7: Prefetch Alternative

If cached select is too flow-heavy, try final prefetching:

```text
precompute final addresses earlier
issue loads earlier
store loaded nodes in per-context node_tmp
consume in final hash
```

This requires keeping prefetched node values live across a region boundary.

Potential scratch source:

- `node_tmp` itself, if prefetched immediately before final region
- alias select temps after pre-drain selection is done

This is less likely to reduce load slots, but may move final loads earlier.

## Phase 8: Decide Direction

After prototype measurements:

### If full final level-4 cache improves cycles

Invest in region-local allocator:

- top-node cache reused as final level-4 cache
- final-only cache construction
- selector-safe select tree

### If partial cache improves cycles

Search hot node sets and scratch fit.

### If cache reduces loads but does not improve cycles

Use causality trace to see whether flow select tree became critical.

### If cache regresses

Deprioritize level-4 cache and focus instead on:

- deep gather prefetch
- load scheduling
- memory-layout transformation in scratch

## Expected Range

Best-case:

```text
1153 -> 1120-1140
```

Moderate case:

```text
1153 -> 1145-1152
```

Failure case:

```text
flow/select tree offsets the load savings
```

The experiment is still worth doing because it directly tests whether the next
bottleneck can be attacked structurally rather than by more hash scalarization.
