#!/usr/bin/env python3
"""Pareto-guided policy search for selector-ring kernels."""

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


CRITICAL_CHAIN_TAGS = ("final_drain:gather", "final_drain:hash", "store_tail:store")


def evaluate(variant: dict[str, Any]) -> dict[str, Any]:
    random.seed(123)
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

    return {
        "ok": ok,
        "cycles": machine.cycle,
        "scratch": kb.scratch_ptr,
        "engine_slots": dict(engine_slots),
        "engine_active": dict(engine_active),
        "last_by_region_tag": kb.ir_schedule_summary.get("last_by_region_tag", {}),
        "last_final_hash_cycle": kb.ir_schedule_summary.get("last_final_hash_cycle"),
        "last_final_store_cycle": kb.ir_schedule_summary.get("last_final_store_cycle"),
        "variant": variant,
    }


def policy(h1=(), h2=(), combine=()) -> dict[str, list[int]]:
    return {"h1": list(h1), "h2": list(h2), "combine": list(combine)}


def candidates() -> Iterable[tuple[str, dict[str, Any]]]:
    yield "baseline", {}

    single_policies = {
        "r5_h2_3": {5: policy(h1=(3,), h2=(3,))},
        "r8_h2_5": {8: policy(h1=(3,), h2=(5,))},
        "r6_c_35": {6: policy(h1=(3,), combine=(3, 5))},
        "r9_h1_3": {9: policy(h1=(3,))},
        "r13_h2_5": {13: policy(h1=(3,), h2=(5,))},
        "r12_c_5": {12: policy(h1=(3,), combine=(5,))},
        "r14_h2_5": {14: policy(h1=(3,), h2=(5,))},
    }

    for name, by_round in single_policies.items():
        yield name, {"simd_hash_scalar_by_round": by_round}

    for size in (2, 3):
        for names in combinations(single_policies, size):
            by_round: dict[int, dict[str, list[int]]] = {}
            for name in names:
                by_round.update(single_policies[name])
            yield "+".join(names), {"simd_hash_scalar_by_round": by_round}

    base_xor = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    for level in range(11):
        if level in base_xor:
            reduced = [candidate for candidate in base_xor if candidate != level]
            yield f"xor_without_{level}", {"scalar_xor_levels": reduced}
        else:
            yield f"xor_with_{level}", {"scalar_xor_levels": sorted(base_xor + [level])}

    for rotation in range(16):
        yield f"rotation_{rotation}", {"simd_predrain_group_rotation": rotation}

    for boundaries in (
        [0, 11, 15],
        [0, 12, 15],
        [0, 11, 13, 15],
        [0, 10, 12, 15],
        [0, 10, 13, 15],
        [0, 9, 11, 15],
    ):
        yield f"boundaries_{'_'.join(map(str, boundaries))}", {
            "simd_body_tile_boundaries": boundaries
        }


def metrics(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    active = row["engine_active"]
    slots = row["engine_slots"]
    return (
        row["cycles"],
        active.get("valu", 10**9),
        slots.get("valu", 10**9),
        active.get("alu", 10**9),
        row["last_final_store_cycle"] or 10**9,
    )


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lm = metrics(left)
    rm = metrics(right)
    return all(l <= r for l, r in zip(lm, rm)) and any(l < r for l, r in zip(lm, rm))


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    good = [row for row in rows if row["ok"]]
    frontier = []
    for row in good:
        if any(dominates(other, row) for other in good if other is not row):
            continue
        frontier.append(row)
    return sorted(frontier, key=metrics)


def critical_chain_deltas(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, int]:
    tags = row.get("last_by_region_tag", {})
    base_tags = baseline.get("last_by_region_tag", {})
    deltas = {
        tag: tags.get(tag, -1) - base_tags.get(tag, -1)
        for tag in CRITICAL_CHAIN_TAGS
    }
    deltas["last_final_hash"] = (
        (row.get("last_final_hash_cycle") or 0)
        - (baseline.get("last_final_hash_cycle") or 0)
    )
    deltas["last_final_store"] = (
        (row.get("last_final_store_cycle") or 0)
        - (baseline.get("last_final_store_cycle") or 0)
    )
    return deltas


def annotate_chain_deltas(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    for row in rows:
        if not row["ok"]:
            continue
        deltas = critical_chain_deltas(row, baseline)
        row["critical_chain_delta"] = deltas
        row["critical_chain_max_delay"] = max(deltas.values())
        row["critical_chain_delayed"] = row["critical_chain_max_delay"] > 0


def chain_metrics(row: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    base = metrics(row)
    return (
        row.get("critical_chain_max_delay", 10**9),
        row.get("critical_chain_delta", {}).get("final_drain:gather", 10**9),
        row.get("critical_chain_delta", {}).get("final_drain:hash", 10**9),
        row.get("critical_chain_delta", {}).get("last_final_store", 10**9),
        base[1],
        base[2],
    )


def chain_dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lm = chain_metrics(left)
    rm = chain_metrics(right)
    return all(l <= r for l, r in zip(lm, rm)) and any(l < r for l, r in zip(lm, rm))


def chain_safe_frontier(
    rows: list[dict[str, Any]], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    good = [
        row
        for row in rows
        if row["ok"] and row.get("critical_chain_max_delay", 10**9) <= 0
    ]
    frontier = []
    for row in good:
        if any(chain_dominates(other, row) for other in good if other is not row):
            continue
        frontier.append(row)
    return sorted(frontier, key=lambda row: (row["cycles"], chain_metrics(row)))


def low_valu_rejected_by_chain(
    rows: list[dict[str, Any]], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    base_valu_slots = baseline["engine_slots"].get("valu", 10**9)
    return sorted(
        [
            row
            for row in rows
            if row["ok"]
            and row["engine_slots"].get("valu", 10**9) < base_valu_slots
            and row.get("critical_chain_max_delay", 0) > 0
        ],
        key=lambda row: (
            row["engine_slots"].get("valu", 10**9),
            row.get("critical_chain_max_delay", 10**9),
            row["cycles"],
        ),
    )


def chain_aware_score(row: dict[str, Any], baseline: dict[str, Any]) -> int:
    """Score candidates by cycles, final-chain readiness, and engine pressure."""
    if not row["ok"]:
        return 10**18
    active = row["engine_active"]
    slots = row["engine_slots"]
    base_active = baseline["engine_active"]
    base_slots = baseline["engine_slots"]
    chain = row.get("critical_chain_delta", {})
    chain_delay = max(0, row.get("critical_chain_max_delay", 0))
    final_gather_delay = max(0, chain.get("final_drain:gather", 0))
    final_hash_delay = max(0, chain.get("final_drain:hash", 0))
    final_store_delay = max(0, chain.get("last_final_store", 0))
    valu_active_delta = active.get("valu", 0) - base_active.get("valu", 0)
    valu_slot_delta = slots.get("valu", 0) - base_slots.get("valu", 0)
    alu_active_delta = active.get("alu", 0) - base_active.get("alu", 0)
    return (
        row["cycles"] * 100_000
        + chain_delay * 20_000
        + final_gather_delay * 8_000
        + final_hash_delay * 8_000
        + final_store_delay * 4_000
        + max(0, valu_active_delta) * 800
        + max(0, alu_active_delta) * 200
        + valu_slot_delta
    )


def summarize_for_score(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    active = row["engine_active"]
    slots = row["engine_slots"]
    base_active = baseline["engine_active"]
    base_slots = baseline["engine_slots"]
    return {
        "name": row["name"],
        "score": row.get("chain_aware_score"),
        "cycles": row["cycles"],
        "cycle_delta": row["cycles"] - baseline["cycles"],
        "critical_chain_delta": row.get("critical_chain_delta"),
        "critical_chain_max_delay": row.get("critical_chain_max_delay"),
        "valu_active_delta": active.get("valu", 0) - base_active.get("valu", 0),
        "valu_slot_delta": slots.get("valu", 0) - base_slots.get("valu", 0),
        "alu_active_delta": active.get("alu", 0) - base_active.get("alu", 0),
        "alu_slot_delta": slots.get("alu", 0) - base_slots.get("alu", 0),
        "variant": row["variant"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, variant in candidates():
        key = json.dumps(variant, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        try:
            row = evaluate(variant)
        except Exception as exc:  # noqa: BLE001 - search should report bad candidates.
            row = {
                "ok": False,
                "cycles": None,
                "scratch": None,
                "engine_slots": {},
                "engine_active": {},
                "last_final_hash_cycle": None,
                "last_final_store_cycle": None,
                "variant": variant,
                "error": f"{type(exc).__name__}: {exc}",
            }
        row["name"] = name
        rows.append(row)

    baseline = next(row for row in rows if row["name"] == "baseline" and row["ok"])
    annotate_chain_deltas(rows, baseline)
    for row in rows:
        row["chain_aware_score"] = chain_aware_score(row, baseline)

    print("# top_by_cycles")
    for row in sorted(
        rows,
        key=lambda item: (
            not item["ok"],
            item["cycles"] if item["cycles"] is not None else 10**9,
            item["last_final_store_cycle"] or 10**9,
        ),
    )[: args.top]:
        print(json.dumps(row, sort_keys=True))

    print("# pareto_frontier")
    for row in pareto_frontier(rows)[: args.top]:
        print(json.dumps(row, sort_keys=True))

    print("# chain_safe_frontier")
    for row in chain_safe_frontier(rows, baseline)[: args.top]:
        print(json.dumps(row, sort_keys=True))

    print("# chain_aware_score")
    for row in sorted(rows, key=lambda item: item["chain_aware_score"])[: args.top]:
        print(json.dumps(summarize_for_score(row, baseline), sort_keys=True))

    print("# low_valu_rejected_by_chain")
    for row in low_valu_rejected_by_chain(rows, baseline)[: args.top]:
        print(json.dumps(row, sort_keys=True))

    print(
        json.dumps(
            {
                "count": len(rows),
                "frontier": len(pareto_frontier(rows)),
                "chain_safe_frontier": len(chain_safe_frontier(rows, baseline)),
                "low_valu_rejected_by_chain": len(
                    low_valu_rejected_by_chain(rows, baseline)
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
