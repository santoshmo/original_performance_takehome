# Performance Takehome — TTC Exploration Findings

Generated 2026-05-19. Starting baseline cycles: **1486** (current committed best
on `main` before this session). Current best in working tree: **1479**, with two
1477 candidates discovered via random search (not yet committed as defaults).

Target thresholds (from `tests/submission_tests.py`):

- `< 1487` opus 4.5 11h — passing
- `< 1363` opus 4.5 improved harness — **NOT yet passing**

## Current state of `perf_takehome.py`

Default variant produces 1479 cycles. Key changes from the original 1486 commit:

| field | value | notes |
|---|---|---|
| `scalar_hash_start_by_stage` | `{1:26, 3:23, 5:25}` | was `{1:28, 3:25, 5:25}` |
| `scalar_gather_start_group` | `24` | was `20` |
| `scalar_index_start_group` | `26` | was `19` |
| `cache_depth3_start_group` | `14` | unchanged |
| `hash_h2_in_vals` | `True` | new — see below |

### `hash_h2_in_vals` (committed in defaults)

Non-mul_add hash stages (1, 3, 5) used to write the two intermediates `h1` and
`h2` into `tmp1[g]` and `tmp2[g]`, then combine them into `vals`. The new path
writes `h2` directly into `vals` (relying on same-cycle read-before-write to
keep `h1` intact). The op count is the same (3 valu per stage), but `tmp2` is
now free during hash. By itself this is cycle-neutral, but it unlocks
`early_doubled` (which ended up not helping) and any future optimisation that
needs scratch during the hash phase.

### Engine profile at 1479 cycles

```
alu   slots=11808 floor= 984 active=1163 util=68.9%
valu  slots= 7300 floor=1217 active=1427 util=85.1%
load  slots= 2304 floor=1152 active=1152 util=80.2%
flow  slots=  256 floor= 256 active= 256 util=17.9%
```

- **valu is the bottleneck.** floor=1217, active=1427, gap=210 cycles.
- load is fully saturated at its floor.
- flow has huge headroom (only 17.9% util) but vselect chains are serial.
- valu fill histogram has ~62% of valu-active cycles at 6 slots, the rest at
  ≤5 slots. Most of the slack is in a **dip during the load-heavy depth ≥ 4
  rounds**, where valu has nothing to do because hash for round R is waiting
  on round R's 128-cycle load.

## Things tried (variant flags exposed in `KernelBuilder.build_kernel`)

| flag | tried values | result | conclusion |
|---|---|---|---|
| `scalar_hash_start_by_stage` | grid + random search, ranges 18-32 per stage | best `{1:26, 3:23, 5:26}` → **1477** | very flat valley around 1477-1486 |
| `scalar_gather_start_group` | 0, 4, 8, 12, 16, 18, 20, 22, 24, 26, 28 | 23-25 is the sweet spot | 0 and 28+ both regress strongly |
| `scalar_index_start_group` | 14-32 | 25-27 is best | each unit shifts ~1 cycle |
| `cache_depth3_start_group` | 0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 32 | **14-15** | full caching loses, no caching loses |
| `cache_depth3_select` | `muladd` (default), `vselect` | `vselect` regresses to **1530** | flow chain serialises 7 vselects per group |
| `cache_depth3_vselect_count` | 0-8 (trailing vselect substitutions) | all regress (1486 → 1490-1567) | mixed mul_add/vselect chain still serialises on flow |
| `cache_depth3_tree` | full 3-level binary tree mux over shared 4-vec pool, pairs vselect, upper mul_add | regresses to **1608-1631** | shared pool serialises 18 cached groups, kills parallelism |
| `hash_h2_in_vals` | True / False | cycle-neutral by itself | committed; frees `tmp2` during hash |
| `early_doubled` | True / False | regresses (+26 cycles) | the +1 valu per group from splitting `mul_add` outweighs the dip-filling |
| `hybrid_hash_start_by_stage` | thresholds 16-32 | regresses | the 2-valu-per-stage variant trades poorly into 8 alu |
| `scalar_muladd_hash_start_group` | 4-32 | regresses | scalarising the mul_add stages is always net loss |
| `group_order` | forward, reverse, interleave | no effect | dep graph dominates |
| `scheduler` | fifo (default), cp, lp, valu_cp | all custom regress (cp=1699, lp=1619, valu_cp=1771) | naive critical-path priority defeats the carefully arranged FIFO add order |
| `scalar_hash_start_by_depth_stage` | per-depth overrides | regresses or neutral | per-round shifts didn't help the dip |

## What is fundamentally limiting?

The valu floor of 1217 corresponds to the hash work:

```
hash valu slots = 5088, floor = 848
+ select valu = 1048, floor = 175
+ index valu = 676, floor = 113
+ gather valu = 488, floor = 82
≈ 1218 minimum valu cycles
```

The 210-cycle gap above the floor is essentially the **load-bottleneck dip**:
during the ≈ 8 depth ≥ 4 rounds (each ≈ 128 cycles of load + 64 cycles of
hash that can overlap), valu is starved because every ready valu op needs
this round's gathered value. There is no independent valu work to fill the
dip — every group's next-stage hash strictly depends on its previous stage's
output, and every round's gather strictly depends on the previous round's
index update.

## Structural ideas that did **not** work and why

1. **vselect tree mux at depth 3** — saves valu but the shared 4-vec scratch
   pool needed to fit the scratch budget (53 free words → 4 pool + 1 bit2 =
   40) forces serial deps between cached groups; the 7-deep flow chain per
   group at 1 flow/cycle costs more cycles than the valu reduction saves.

2. **Per-group tree mux with private pair scratches** — would need
   18 cached groups × 4 vec pair scratches × 8 words = 576 words extra. The
   only 53 free scratch words mean we cannot afford private per-group pair
   storage.

3. **early-doubled (split `mul_add` of vector index update)** — moves the
   `<<` of `idx<<1` into the load-bottleneck dip but adds +1 valu per group
   per round. The added 228 valu ops pushed valu floor up by 38 while the
   schedule only absorbed ~12, yielding net +26 cycles.

4. **`vselect` cond elimination at depth 1** — semantically equivalent (use
   `idxs` directly as cond) and saves 64 valu, but regressed by 42 cycles.
   The cond op was effectively pacing the vselect issue and removing it
   created a flow burst that desynchronised downstream work.

5. **All-scalar gather (`sg=0`)** — frees vec_const scratch but pushes alu
   active to 1574, becoming the new bottleneck → 1634 cycles.

6. **Critical-path priority scheduler / longest-path priority** — the FIFO
   add order in the current code is already very close to a good list
   schedule for this particular DAG; any naive priority that resorts the
   ready queue tends to either deplete the valu queue (CP) or starve the
   chain (LP).

7. **Per-depth scalar thresholds** — neither making depth ≥ 4 rounds more
   scalar (to fill the dip with alu work) nor making depth 0-3 rounds less
   scalar (to relieve valu in non-load-heavy rounds) helps. The FIFO
   scheduler can't pack the extra alu ops into the dip because they still
   sit in long chains through hash stages.

8. **Absolute-address `idxs` representation** — already tried and rejected
   by the prior author per notes; we did not revisit.

## Search infrastructure used

The new `tools/ttc_runner.py` harness was very effective for parameter
tuning. Run patterns that worked well:

```bash
# Local search (small radius around the current best)
.venv/bin/python tools/ttc_runner.py \
  --local-center-json '{"cache_depth3_start_group":15,...}' \
  --grid scalar_hash_start_by_stage.1:22:30 \
  --grid scalar_hash_start_by_stage.3:18:28 \
  --grid scalar_hash_start_by_stage.5:22:30 \
  --local-radius 2 \
  --output ttc-results/local.jsonl

# Random search over a wider box
.venv/bin/python tools/ttc_runner.py \
  --variant-json '{"hash_h2_in_vals":true,...}' \
  --grid scalar_gather_start_group:18:30 \
  --grid scalar_index_start_group:20:32 \
  ... \
  --random 2000 --random-seed 99 \
  --output ttc-results/random.jsonl
```

JSON keys for `scalar_hash_start_by_stage` need to be strings; the kernel
builder now normalises them with `int(k)` at the top of `build_kernel`. Same
for `hybrid_hash_start_by_stage` and `scalar_hash_start_by_depth_stage`.

## Best discovered variants (not yet committed as defaults)

The randomized search surfaced two variants with **1477 cycles** — 2 cycles
under the committed default:

```jsonc
// Variant A — cycles=1477
{
  "cache_depth3_start_group": 15,
  "scalar_gather_start_group": 24,
  "scalar_hash_start_by_stage": {"1": 26, "3": 23, "5": 26},
  "scalar_index_start_group": 25
}

// Variant B — cycles=1477
{
  "cache_depth3_start_group": 14,
  "scalar_gather_start_group": 23,
  "scalar_hash_start_by_stage": {"1": 27, "3": 23, "5": 24},
  "scalar_index_start_group": 27
}
```

These both already implicitly use `hash_h2_in_vals=true` because the default
in the current source is `True`.

## Where the gains would have to come from

To beat the next threshold (1363) we need ~115 more cycles. The honest
diagnosis is that pure parameter tuning is saturated near 1477 and we need
either:

1. **A structurally different selection / gather method** that genuinely
   reduces valu count *and* keeps parallelism (e.g. caching depth-4 nodes,
   which requires reclaiming ~90 scratch words from somewhere — likely
   eliminating the 8 depth-specific `vaddr_const` vectors and going
   scalar-gather everywhere, which itself regresses today).

2. **A smarter scheduler** that actually outperforms FIFO on this DAG. The
   naive critical-path heuristic loses, but a slack-aware list scheduler
   (or a hand-tuned reordering of `add_node` calls that respects engine
   floors) might recover some of the 210-cycle gap.

3. **Re-shaping the dep graph** so that round R+1's parallelisable work
   (e.g. address arithmetic, broadcast-style ops) can run inside round R's
   load-bottlenecked gather window. Today every such candidate transitively
   depends on round R's hash output via the parity bit.

## Files produced this session

- `tools/ttc_runner.py`, `tools/profile_kernel.py`, `tools/worktree_experiment.py`
  (provided by the user)
- `bench.py` — single-variant wrapper used during exploration
- `sweep.py`, `sweep2.py`, `slot_profile.py`, `dip_profile.py`,
  `profile_schedule.py` — local exploration scripts
- `ttc-results/*.jsonl` — raw experiment outputs from the harness
