# Plan For 1130-1150 Cycles: Full-Width Selector Ring

This plan assumes a fresh branch based on `origin/main`, but with the lessons
from the `fresh-bitstream-selectors` experiment available as design context. Do
not modify anything under `tests/`.

Current best from the selector-ring branch:

```text
cycles: 1178
group_size: 16
selector ring depth: 1
scratch: 1531 / 1536
VALU slots: 6632
VALU active: 1172
load active: 1112
```

The realistic next target is `1130-1150`. The key constraint is scratch: one
selector vector per active context already almost exhausts scratch. A deeper
selector ring appears useful, but simply reducing `group_size` to fit it loses
too much overlap. The goal is therefore:

```text
fit a 2-bit or 3-bit selector ring at group_size=16
without increasing tail latency or reducing active-context overlap
```

## Core Hypothesis

The one-bit selector ring wins because it avoids rematerializing `path & 1` at
shallow levels.

A two-bit ring should additionally remove:

- the level-2 `path & 2` mask
- the level-3 `path & 2` mask

A three-bit ring should remove all shallow bit extraction for levels `1..3`,
including `path & 4` at level 3.

This should reduce `VALU` slots further. The danger is that storing selector
history consumes scratch and extends lifetimes enough to destroy overlap.

## Target Outcomes

Primary:

- `tests/submission_tests.py` passes
- cycles in `1130-1150`

Intermediate:

- preserve `group_size=16`
- fit scratch under `1536`
- reduce `VALU` slots below the one-bit-ring `6632`
- keep final hash/store no later than the `1178` schedule
- keep `ALU` active below the final tail

## Phase 1: Establish Selector-Ring Baseline

Start from `origin/main`, then recreate the known-good `1178` design:

- SIMD IR kernel
- split final value-only round
- custom body boundaries `[0, 11, 15]`
- pre-drain group rotation `13`
- one-entry selector ring
- round-13 hash scalarization seed

Verify:

```bash
./.venv/bin/python tests/submission_tests.py
# expected: CYCLES: 1178
```

Record:

- scratch usage
- engine slots
- engine active cycles
- last final hash/store

## Phase 2: Scratch Audit

Before adding more selector vectors, produce a scratch role report.

Current per-context scratch roughly includes:

- `node`
- `tmp1` / hash temp
- `tmp2` / select temp
- `tmp3` / select temp
- `selector0`

At `group_size=16`, one extra vector per context costs `16 * VLEN = 128`
scratch words. The current one-bit design has only about `5` words free, so a
two-bit ring requires freeing almost a full vector array.

Audit scratch by role:

```text
batch path/value state
context node temp
context hash temp
context select temp 0
context select temp 1
selector ring
top-node cache
constants
setup/io temps
```

The audit should answer:

- Which vectors are live across region boundaries?
- Which vectors are only needed inside levels `1..3`?
- Which vectors are only needed during hash and can be aliased after hash?
- Which constants or node cache entries can be represented more compactly?

## Phase 3: Free One Full Vector Array

The first concrete milestone is to free `128` scratch words while preserving
`group_size=16`.

Candidate approaches:

### 1. Merge `node` and one select temp where lifetimes allow

In shallow selection, `node`, `tmp1`, `tmp2`, and `tmp3` are used as mux
intermediates. Rework the level-2 and level-3 select trees to use fewer live
temps.

Success condition:

- same correctness
- no additional serialization
- one vector array removed or made available for selector ring

### 2. Use a 2-temp depth-3 tree

Current depth-3 selection wants a node temp plus select temps. Search/select a
tree shape that consumes:

```text
node
tmp1
tmp2
```

instead of:

```text
node
tmp1
tmp2
tmp3
```

The selector ring already supplies low-bit conditions, so the tree may not need
as many condition temporaries. This is the most promising scratch source.

### 3. Make selector storage reuse an existing temp after parity

Instead of adding `selector1`, make it alias an existing context vector that is
dead after the hash/index update.

Candidate aliases:

- `tmp3` outside level-3 selection
- `node` after xor
- `tmp2` after hash combine

This requires exact liveness and may need explicit alias dependency modeling.
Only accept if it does not serialize the useful hash pipeline.

### 4. Compress top-node cache or constants

This is less likely to free enough scratch, but still audit:

- duplicate vector constants
- unused constants after offset-state traversal
- top-node cache representation

Do not trade a large runtime cost for scratch here unless it enables a much
better selector ring.

## Phase 4: Two-Bit Selector Ring

Once one vector array is available, implement:

```python
selector_ring_depth = 2
```

Semantics:

```text
selector[(round - 1) % 2] = newest prior branch bit
selector[(round - 2) % 2] = older prior branch bit
```

Use selectors for:

- level 1: bit 0
- level 2: bits 0 and 1
- level 3: bits 0 and 1, still materialize bit 2 from path

Expected effect:

- removes level-2 `path & 2`
- removes level-3 `path & 2`
- further reduces `VALU` slots

Validation:

```bash
./.venv/bin/python tests/submission_tests.py
```

Measure:

- cycles
- `VALU` slots
- `VALU active`
- final hash/store
- scratch

If cycles regress despite lower `VALU` slots, inspect whether selector lifetime
or reduced overlap caused it.

## Phase 5: Three-Bit Selector Ring

If two-bit selector ring improves or is neutral, try depth `3`.

Use selectors for:

- level 1: bit 0
- level 2: bits 0 and 1
- level 3: bits 0, 1, and 2

This removes all shallow bit extraction. It costs another vector array unless
aliasing can support it.

Only pursue depth `3` if:

- depth `2` improves or maintains schedule quality
- scratch can fit without reducing `group_size`
- final hash/store readiness does not move later

## Phase 6: Retune Region Order

Selector-ring depth changes the schedule shape, so retune:

- pre-drain group rotation
- body boundaries
- final group order
- round-13 hash scalarization
- local tail reschedule windows

Start with:

```python
simd_body_tile_boundaries = [0, 11, 15]
simd_predrain_order_rounds = [11]
simd_predrain_group_order = "rotate"
simd_predrain_group_rotation = 13
```

Then sweep:

```text
rotations: 0..15
boundaries: [0,11,15], [0,11,13,15], [0,10,12,15], [0,10,13,15]
selector depth: 1, 2, 3
hash policy: current seed plus small perturbations
```

## Phase 7: Decide Whether To Keep It

Accept the two- or three-bit selector design only if it shows one of:

- total cycles below `1178`
- `VALU active` meaningfully below `1172` with same or earlier final hash/store
- a lower `VALU` slot count that combines with a boundary/order retune into a
  cycle win

Reject designs that:

- reduce `VALU` slots but move final hash/store later
- require reducing `group_size`
- create scratch alias dependencies that serialize hash or gather
- increase `ALU` enough to create a new tail

## Most Likely Implementation Path

1. Keep the one-bit selector ring exactly as in the `1178` branch.
2. Rework level-3 selection to use one fewer temp.
3. Use the freed vector array as `selector1`.
4. Validate two-bit selector ring at `group_size=16`.
5. Retune pre-drain rotation.
6. If good, try aliasing another temp for `selector2`.

## Why This Could Reach 1130-1150

The current selector-ring kernel has a theoretical `VALU` floor around `1106`,
but actual cycles are `1178`. A deeper selector ring can lower the `VALU` slot
floor. If it also preserves full overlap and final readiness, the practical
schedule may move into the `1130-1150` range.

The important point is that this is not just another scalarization tweak. It is
a representation change: shallow traversal reads carried selector state instead
of reconstructing selector bits from the path vector.

That is the kind of structural change needed for another meaningful step down.
