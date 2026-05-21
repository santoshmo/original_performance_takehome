# From-Scratch Variant Plan

Goal: rebuild the kernel exploration around schedule shape, scratch lifetimes, and final-drain behavior so we can search past the current 1240-cycle implementation without getting trapped in the old steady-state local minimum.

## Current Baseline

Use the current `scheduler_rewrite` kernel as the reference behavior, not as the architecture to copy blindly.

- Verified default performance: 1240 cycles.
- Core algorithmic choices are already strong: offset-state traversal, depth-3 cached 7-`vselect` tree, scalar XOR on selected levels, scalar parity extraction, and hash `multiply_add` fusion.
- The recent breakthrough came from scheduling shape, not a new arithmetic trick: final tile rotation moved final hash/store work earlier, then pause removal and dead tail pruning removed wasted cycles.

The from-scratch build should preserve the proven algorithm but redesign the code generator around regions and scratch lifetimes.

## Design Principle

Do not emit one flat steady-state program and hope the greedy scheduler drains well.

Instead split the kernel into regions:

1. Setup and initial loads.
2. Steady-state body tiles.
3. Final drain tile.
4. Store tail.

Each region should be allowed to use different group order, scratch pooling, dead-work policy, and scheduler tie-breaks.

## Phase 1: Explicit IR

Build a small operation IR before emitting VLIW slots.

Each operation should track:

- engine and slot payload
- exact scratch reads and writes
- semantic tags: `load`, `select`, `hash`, `index`, `store`, `setup`
- round, level, group, lane metadata
- whether the op contributes to final values, final indices, or only future traversal
- whether it is safe to drop in the final round

Dependency construction should infer:

- RAW latency 1
- WAW latency 1
- WAR latency 0

The scheduler should consume this IR directly. Avoid relying on Python emit order as the hidden scheduling policy.

## Phase 2: Region-Aware Scheduler

Implement a scheduler that can score candidates differently by region.

Steady-state objective:

- keep load and VALU active
- avoid starving flow during depth-3 select chains
- keep ALU parity/address work off the critical path

Final-drain objective:

- prioritize operations on the last few groups whose values have not yet been stored
- deprioritize or drop final index updates
- move final stores as early as value readiness permits
- allow final tile group order to differ from body order

The first useful scheduler modes to implement:

- `fifo`: current-style baseline.
- `critical_path`: longest path to final store.
- `tail_weighted`: add extra priority for rounds 13-15 and groups that determine the final store tail.
- `engine_balanced`: prefer ready ops on underfilled bottleneck engines.

## Phase 3: Scratch Role Model

Replace fixed context fields like `node/tmp1/tmp2/tmp3` with named scratch roles.

Suggested roles:

- `value`: persistent input/output vector.
- `path`: persistent offset vector.
- `node_tmp`: selected node or gathered node.
- `hash_tmp`: temporary used by unfused hash stages.
- `select_tmp0`: depth-2/depth-3 pair or bit temp.
- `select_tmp1`: depth-3 second pair/quad temp.
- `addr_tmp`: scalar lane addresses for gathers.
- `parity_tmp`: scalar lane parity for index update.

Then implement a simple allocator that maps roles to scratch pools based on non-overlapping live ranges.

Important experiments:

- Keep `tmp3` pooled even in the default kernel. We measured `tmp3_pool_size=2` preserving 1245/1240 behavior while freeing scratch.
- Test whether `tmp2` can become select-only under single-temp hash.
- Let final-drain regions reuse scratch more aggressively because final index state is dead.

## Phase 4: Variant Families

### Variant A: Current Shape, Better Tail Search

Keep the current `group_size=16`, `round_tile=13` structure.

Search:

- final tile rotations `0..15`
- final tile even/odd group orders
- final two-tile rotations
- special ordering for rounds 13, 14, and 15 separately

Success metric:

- beat 1240 with no algorithmic change
- inspect whether final `store` moves before cycle 1240

This is the lowest-risk path.

### Variant B: Pooled Select Temps

Keep two-temp hash, but pool depth-3 select temps.

Known-good starting point:

- `group_size=16`
- `round_tile=13`
- `tmp3_pool_size=2`
- `final_tile_rotation=2`

Search:

- `tmp3_pool_size` in `1, 2, 3, 4, 6, 8`
- final tile rotations and final ordering
- whether freed scratch can support extra final-only temps

Purpose:

- preserve current performance while creating scratch headroom for region-specific scheduling.

### Variant C: Single-Temp Hash With Pooled Select Temps

Use single-temp hash so `tmp2` no longer has hash lifetime.

Known close result:

- `single_temp_hash=True`
- `group_size=32`
- `round_tile=11`
- `tmp2_pool_size=6`
- `tmp3_pool_size=2`
- measured around 1246 before tail trimming

Search:

- `group_size` in `20, 24, 28, 32`
- `round_tile` in `6..16`
- `tmp2_pool_size` in `2, 4, 6, 8, 12`
- `tmp3_pool_size` in `1, 2, 4, 8`
- final tile rotation per configuration

Purpose:

- determine whether larger active context count can beat 16-context scheduling once final-drain ordering is co-designed.

### Variant D: Split Body And Final Kernel

Generate body rounds normally, then emit a separately ordered final drain program for rounds 13-15.

Body:

- tuned for steady-state throughput
- no awareness of final stores

Final kernel:

- round-major or group-order searched
- no final index update after round 15
- stores interleaved as soon as each group value is ready
- scratch roles reused without preserving future traversal state

Search:

- body `group_size=16`, final `group_size=16`
- body `group_size=16`, final full-batch order
- body `tmp3_pool_size=2`, final extra private temps
- final group rotations and partial permutations

This is the most promising structural direction if simple final-order search stalls.

### Variant E: Tail-Only Post-Schedule Optimizer

Keep the current generator, but add a post-schedule pass for the last 50-100 cycles.

Passes:

- remove final pause/debug-only bundles
- remove final index update ops that cannot affect output values
- locally reschedule remaining tail ops with exact read/write dependencies
- preserve the already-good steady-state schedule before the tail window

Purpose:

- get most of the benefit of a new scheduler without rewriting the whole kernel.

## Search Harness

For each candidate:

1. Build kernel.
2. Verify correctness against multiple seeds.
3. Record cycle count, scratch use, engine slots, active cycles, and last-op cycles by engine.
4. Save the variant dict and a compact tail trace.

Minimum recorded fields:

- `cycles`
- `scratch`
- `variant`
- `engine_slots`
- `engine_active`
- `last_flow_vselect`
- `last_hash_valu`
- `last_store`
- `tail_trace`

Reject candidates that only win by failing correctness or depending on modified tests.

## Near-Term Implementation Order

1. Add a tail-search script that tries final group rotations and small final group permutations against the current generator.
2. Add a post-schedule tail rescheduler over the final 50 cycles.
3. Split final rounds 13-15 into a separate generation region.
4. Add role-based scratch allocation for final-only region.
5. Re-test single-temp hash and pooled `tmp2/tmp3` only after the final region exists.

## Expected Wins

Likely:

- 1-3 cycles from better final ordering and dead-tail handling.

Possible:

- 3-8 cycles if a final-only region starts the last groups' final hash chains earlier while preserving steady-state throughput.

Harder:

- larger gains require a genuinely better scheduler or a new way to reduce final hash-chain depth, not just more scratch.

## Stop Conditions

Stop a branch if:

- it increases active VALU cycles without moving the final store earlier
- it relies on more scratch but keeps the same tail critical path
- it beats cycles only by dropping value-producing hash/store work
- it makes correctness seed-sensitive

The main signal to watch is not total VALU slot count. It is whether the final value-producing hash and final store move left.
