#!/usr/bin/env python3
"""Region-policy search for the fresh <1100-cycle attempt.

The search is intentionally metric-driven: it keeps total cycles as the final
objective, but prints enough engine-pressure data to identify candidates that
reduce VALU pressure without creating an ALU tail.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from frozen_problem import (  # noqa: E402
    Input,
    Machine,
    N_CORES,
    Tree,
    build_mem_image,
    reference_kernel2,
)
from perf_takehome import KernelBuilder  # noqa: E402


BASELINE_HASH_POLICY = {"h1": [1, 3], "h2": [5], "combine": [3, 5]}
BOUNDARY_SHAPES = [
    [0, 11, 15],
    [0, 10, 12, 15],
    [0, 10, 13, 15],
    [0, 11, 13, 15],
    [0, 9, 11, 13, 15],
    [0, 8, 11, 15],
]


def evaluate(variant: dict[str, Any], seed: int = 123) -> dict[str, Any]:
    random.seed(seed)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16, variant=variant)
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    for ref_mem in reference_kernel2(mem):
        pass

    inp_values_p = ref_mem[6]
    ok = (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    )

    engine_slots: Counter[str] = Counter()
    engine_active: Counter[str] = Counter()
    for instr in kb.instrs:
        for engine, slots in instr.items():
            if engine == "debug":
                continue
            engine_slots[engine] += len(slots)
            engine_active[engine] += 1

    summary = kb.ir_schedule_summary
    return {
        "ok": ok,
        "cycles": machine.cycle,
        "scratch": kb.scratch_ptr,
        "engine_slots": dict(engine_slots),
        "engine_active": dict(engine_active),
        "last_final_hash_cycle": summary.get("last_final_hash_cycle"),
        "last_final_store_cycle": summary.get("last_final_store_cycle"),
        "variant": variant,
    }


def hash_policy(h1=(), h2=(), combine=()) -> dict[str, list[int]]:
    return {
        "h1": list(h1),
        "h2": list(h2),
        "combine": list(combine),
    }


def named_variant(name: str, variant: dict[str, Any]):
    return name, variant


def candidate_variants(mode: str) -> Iterable[tuple[str, dict[str, Any]]]:
    yield named_variant("baseline", {})

    # Boundary seeds from the plan.
    for boundaries in BOUNDARY_SHAPES:
        yield named_variant(
            f"boundaries_{'_'.join(map(str, boundaries))}",
            {"simd_body_tile_boundaries": boundaries},
        )

    # Single-round hash scalarization around the known good policy.
    h1_sets = [(), (3,), (1, 3), tuple(range(6))]
    h2_sets = [(), (1,), (3,), (5,), (1, 3)]
    combine_sets = [(), (5,), (1, 5), (3, 5), (4, 5)]
    rounds = range(4, 15) if mode == "broad" else range(11, 15)
    for round_i in rounds:
        for h1 in h1_sets:
            yield named_variant(
                f"r{round_i}_h1_{h1}",
                {"simd_hash_scalar_by_round": {round_i: hash_policy(h1=h1)}},
            )
        for h2 in h2_sets:
            yield named_variant(
                f"r{round_i}_h2_{h2}",
                {
                    "simd_hash_scalar_by_round": {
                        round_i: hash_policy(h1=(3,), h2=h2)
                    }
                },
            )
        for combine in combine_sets:
            yield named_variant(
                f"r{round_i}_combine_{combine}",
                {
                    "simd_hash_scalar_by_round": {
                        round_i: hash_policy(h1=(3,), combine=combine)
                    }
                },
            )

    # Pairwise perturbations only where prior work found leverage.
    focus_rounds = [11, 12, 13, 14]
    for r1, r2 in combinations(focus_rounds, 2):
        yield named_variant(
            f"pair_hash_{r1}_{r2}",
            {
                "simd_hash_scalar_by_round": {
                    r1: BASELINE_HASH_POLICY,
                    r2: BASELINE_HASH_POLICY,
                }
            },
        )

    # VALU pressure probes outside hash.
    base_xor_levels = {1, 2, 3, 4, 5, 6, 7, 8, 10}
    for level in range(11):
        yield named_variant(
            f"xor_add_level_{level}",
            {"scalar_xor_levels": sorted(base_xor_levels | {level})},
        )
        if level in base_xor_levels:
            yield named_variant(
                f"xor_remove_level_{level}",
                {"scalar_xor_levels": sorted(base_xor_levels - {level})},
            )

    for round_i in focus_rounds:
        yield named_variant(f"scalar_xor_round_{round_i}", {"scalar_xor_rounds": [round_i]})
        yield named_variant(f"vector_xor_round_{round_i}", {"vector_xor_rounds": [round_i]})
        yield named_variant(
            f"scalar_index_round_{round_i}",
            {"scalar_index_rounds": [round_i]},
        )


def score(row: dict[str, Any], baseline: dict[str, Any]) -> tuple[int, int, int, int]:
    if not row["ok"]:
        return (1, 10**12, 10**12, 10**12)
    active = row["engine_active"]
    last_hash = row["last_final_hash_cycle"] or 10**9
    last_store = row["last_final_store_cycle"] or 10**9
    baseline_hash = baseline["last_final_hash_cycle"] or last_hash
    baseline_store = baseline["last_final_store_cycle"] or last_store
    pressure_score = (
        row["cycles"] * 10000
        + active.get("valu", 0) * 100
        + max(0, last_hash - baseline_hash) * 50
        + max(0, last_store - baseline_store) * 50
    )
    return (0, pressure_score, row["cycles"], active.get("valu", 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "broad"), default="quick")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    baseline: dict[str, Any] | None = None
    out = args.jsonl.open("a") if args.jsonl else None
    try:
        for name, variant in candidate_variants(args.mode):
            key = json.dumps(variant, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            try:
                row = evaluate(variant)
            except Exception as exc:  # noqa: BLE001 - searches should report bad policies.
                row = {
                    "ok": False,
                    "cycles": None,
                    "scratch": None,
                    "engine_active": {},
                    "engine_slots": {},
                    "last_final_hash_cycle": None,
                    "last_final_store_cycle": None,
                    "variant": variant,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            row["name"] = name
            rows.append(row)
            if name == "baseline":
                baseline = row
            if out is not None:
                out.write(json.dumps(row, sort_keys=True) + "\n")
                out.flush()
    finally:
        if out is not None:
            out.close()

    baseline = baseline or rows[0]
    for row in sorted(rows, key=lambda item: score(item, baseline))[: args.top]:
        print(json.dumps(row, sort_keys=True))
    print(json.dumps({"count": len(rows), "mode": args.mode}, sort_keys=True))


if __name__ == "__main__":
    main()
