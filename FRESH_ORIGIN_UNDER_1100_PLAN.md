# Fresh-Origin Plan For <1100 Cycles

This plan assumes a new branch starting from `origin/main` of Anthropic's
`original_performance_takehome`. Do not modify anything under `tests/`.

The target is `<1100` cycles on:

```bash
./.venv/bin/python tests/submission_tests.py
```

The known optimized trajectory reached `1199` cycles, but that result is close
to the `VALU` active-cycle floor. A fresh attempt should not just port the final
kernel and keep tuning tails. It should build a small region-aware compiler and
search engine whose first-class objective is reducing `VALU` pressure without
creating an `ALU` tail.

## Core Thesis

The remaining gap from `1199` to `<1100` is too large to come from final-drain
ordering alone.

The winning `1199` shape had roughly:

- `cycles`: `1199`
- `VALU active`: about `1191`
- final hash/store already pulled much earlier than the old `1218` schedule

That leaves only a few cycles of tail slack. To reach `<1100`, the implementation
must remove or relocate a large amount of `VALU` work from the body and
pre-drain rounds. The search should optimize engine pressure directly, especially
`VALU active cycles`, while ensuring `ALU` does not become the new bottleneck.

## Target Architecture

Build a tiny compiler pipeline rather than a single kernel builder with ad hoc
flags:

```text
Region Policy Search
  -> Region-Aware IR Generator
  -> Dependency-Aware ASAP Scheduler
  -> Metrics/Profiler
  -> Candidate Ranker
  -> Validated Kernel Defaults
```

The existing simulator ISA and scratch constraints still apply. The kernel
should eventually emit the same `KernelBuilder.build_kernel` output expected by
the tests.

## Region Policy Model

Represent code generation as a structured policy:

```python
policy = {
    "regions": [
        {
            "name": "body_a",
            "rounds": range(0, 11),
            "group_order": "forward",
            "hash_scalar_by_round": {},
            "scalar_xor_levels": {1, 2, 3, 4, 5, 6, 7, 8, 10},
            "scalar_index_levels": set(),
        },
        {
            "name": "pre_drain",
            "rounds": range(11, 15),
            "group_order": "forward",
            "hash_scalar_by_round": {
                13: {"h1": [1, 3], "h2": [5], "combine": [3, 5]},
            },
        },
        {
            "name": "final",
            "rounds": [15],
            "value_only": True,
            "group_order": "forward",
        },
    ]
}
```

The key is that hash scalarization, XOR scalarization, index update strategy,
group order, and scratch aliasing are round-local or region-local decisions.
Avoid global rules like "stage 3 is scalar everywhere" unless the metrics prove
they help.

## Known Good Algorithmic Baseline

Recreate these concepts early, because they are structural prerequisites:

- SIMD batch state in scratch.
- Active-context tiling with enough contexts to preserve overlap.
- Offset-state tree traversal.
- Cached shallow nodes `0..14`.
- Depth-3 selection using a 7-`vselect` tree.
- Fused hash `multiply_add` where algebra permits it.
- Split final round as a value-only final region.
- Drop final index/path update at generation time.
- Exact IR read/write tracking, including zero-latency WAR handling.
- ASAP-style scheduler as the base packer.

Do not spend early effort on final-wavefront, final store interleaving, or
generic critical-path scheduling. Those already proved weak or regressive in the
optimized trajectory.

## IR Requirements

Every emitted instruction should become an IR op with:

- engine and slot
- exact scratch reads and writes
- round, level, group, lane when relevant
- region name
- tag such as `gather`, `select`, `hash`, `index`, `store`
- contribution flags for value/index paths
- droppable/dead-final metadata
- scratch role metadata

Dependency modeling must include:

- RAW latency `1`
- WAW latency `1`
- WAR latency `0`
- optional memory ordering only where required

The IR should support region-local generation while still scheduling all useful
ops together when that gives better packing.

## Scheduler Strategy

Use ASAP scheduling as the default packer. It is simple, but it preserved the
best known schedules better than generic list schedulers.

Do not try to solve scheduling first with a global critical-path scheduler.
Instead, use generation order and policy search to shape the ready queues. The
scheduler should produce rich metrics for each candidate:

- total cycles
- total slots by engine
- active cycles by engine
- `VALU` active cycles
- `ALU` active cycles
- last final gather
- last final hash
- last final store
- scratch usage
- dependency edge counts

The ranker should reject candidates that reduce op count but keep the same tail
or create an `ALU` tail.

## Cost Model

Use total cycles as the primary score only after correctness passes. For search,
rank candidates with a pressure-aware score:

```python
score = (
    total_cycles * 10000
    + valu_active * 100
    + alu_tail_penalty
    + max(0, last_final_hash - baseline_last_hash) * 50
    + max(0, last_final_store - baseline_last_store) * 50
    + scratch_penalty
)
```

Also keep a Pareto frontier of candidates by:

- cycles
- `VALU active`
- `ALU active`
- last final hash
- last final store

This prevents discarding a candidate that has worse current cycles but much
better `VALU` pressure and may combine well with another policy.

## Search Plan

### Phase 1: Rebuild a Measurable SIMD IR Baseline

Start from `origin/main` and implement the IR-backed SIMD kernel. The goal is
not immediately `<1100`; it is to get a correct, measurable compiler pipeline.

Expected milestones:

- Correctness on `tests/submission_tests.py`.
- A low-cycle SIMD baseline.
- Metrics emitted for engine slots, engine active cycles, and final hash/store
  positions.

### Phase 2: Recreate the Known Structural Wins

Add these in order:

1. Active-context SIMD batching.
2. Fused hash `multiply_add`.
3. Offset-state traversal.
4. Cached shallow nodes and depth-3 `vselect` tree.
5. Split final value-only round.
6. Dead final index/path elimination.
7. Custom body tile boundaries.

Known useful boundary shape from the optimized trajectory:

```python
simd_body_tile_boundaries = [0, 11, 15]
```

Known useful pre-drain scalar hash seed:

```python
hash_scalar_by_round = {
    13: {"h1": [1, 3], "h2": [5], "combine": [3, 5]},
}
```

Treat these as seeds, not final answers.

### Phase 3: Region-Local Hash Scalarization Search

Search round-local hash policies across body and pre-drain rounds:

- h1 scalar stages by round
- h2 scalar stages by round
- combine scalar stages by round
- combinations across adjacent rounds

Start with rounds `11..14`, then expand to `4..10` if `ALU` slack remains.

Accept a policy only if it does at least one of:

- reduces `VALU active`
- moves final hash/store earlier
- lowers total cycles

Reject or quarantine policies that:

- increase `ALU active` into the tail
- delay final hash/store
- reduce slot count without improving scheduled position

### Phase 4: Traversal and XOR VALU Reduction

Search body-level reductions beyond hash:

- scalar XOR by level and by region
- scalar index update by level and by region
- scalar or mixed bit extraction for shallow levels
- alternate depth-1/depth-2 select codegen
- alternate depth-3 bit computation order only if it reduces `VALU active`

Avoid globally scalarizing depth-3 bit extraction; it was already measured as
bad in prior work. Re-test only in narrow regions if the engine-pressure metrics
show `ALU` slack.

### Phase 5: Tile Boundary and Group Order Co-Search

Do not rely on one global `round_tile`. Search explicit boundaries:

```python
[0, 11, 15]
[0, 10, 12, 15]
[0, 10, 13, 15]
[0, 11, 13, 15]
[0, 9, 11, 13, 15]
[0, 8, 11, 15]
```

For each boundary shape, search:

- group order at each region start
- pre-drain hash scalarization tied to actual region starts
- final group order only as a secondary check

The prior result showed `[0, 11, 15]` beat `[0, 11, 13, 15]`, but this can
change after deeper body scalarization.

### Phase 6: Scratch Role and Region Aliasing

Make scratch allocation explicit by role:

- `path`
- `value`
- `hash_tmp`
- `select_tmp`
- `node_tmp`
- `addr_tmp`
- `const`
- `node_cache`
- `store_addr`

Then allow region-local aliasing only where liveness proves it is safe.

Important opportunities:

- final region does not need future path/index state
- pre-drain may not need all traversal temps after final value readiness
- store address scratch can easily lengthen lifetimes, so keep it controlled

Scratch savings only matter if they enable a codegen shape that reduces cycles
or `VALU active`.

### Phase 7: Candidate Composition

After single-policy sweeps, compose only candidates that improve a metric:

1. Best boundary shapes.
2. Best hash scalar policies.
3. Best XOR/index policies.
4. Best region group orders.
5. Safe scratch aliasing variants.

Use a local search over the Pareto frontier instead of brute-forcing all knobs.

## Tooling To Build

Create a search tool, for example:

```bash
./.venv/bin/python tools/region_policy_search.py
```

It should:

- generate policies
- run correctness quickly
- record JSONL results
- print top candidates by cycles and Pareto frontier
- support resuming from prior result files
- report last final hash/store positions
- report engine active cycles and engine slot counts

Example output:

```json
{
  "ok": true,
  "cycles": 1098,
  "valu_active": 1089,
  "alu_active": 1075,
  "last_final_hash": 1068,
  "last_final_store": 1071,
  "scratch": 1403,
  "policy": {
    "boundaries": [0, 10, 12, 15],
    "hash_scalar_by_round": {
      "12": {"h1": [1], "h2": [], "combine": [5]},
      "13": {"h1": [1, 3], "h2": [5], "combine": [3, 5]}
    }
  }
}
```

## Validation Rules

Always validate with:

```bash
./.venv/bin/python tests/submission_tests.py
```

Never edit `tests/`.

For every accepted candidate, record:

- variant or policy
- cycles
- correctness result
- scratch usage
- engine active cycles
- final hash/store cycles

## Success Criteria

Primary:

- `tests/submission_tests.py` passes
- reported cycles `<1100`

Intermediate useful milestones:

- `VALU active` below `1150`
- `VALU active` below `1125`
- final hash/store remains earlier than the `1199` trajectory
- no `ALU` tail after moving hash/index/XOR work off `VALU`

## Likely Failure Modes

- Reducing op count but preserving the same tail.
- Moving too much work to `ALU` and creating a scalar tail.
- Splitting regions too finely and destroying overlap.
- Optimizing final round after final-round tuning is already exhausted.
- Treating scratch savings as useful when they do not enable a faster schedule.

## Guiding Principle

The fresh attempt should not ask, "Which knob gives fewer cycles?"

It should ask:

```text
Which round-local codegen policy reduces VALU active cycles while preserving
hash readiness, ALU slack, and scratch feasibility?
```

That is the path most likely to produce a result below `1100`.
