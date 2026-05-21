#!/usr/bin/env python3
"""Analyze tail-readiness deltas for Pareto search candidates.

This tool answers why some lower-VALU candidates lose: it compares each
candidate against the current default by final-region milestones, engine
pressure, and slot counts.
"""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any


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
from pareto_policy_search import candidates, dominates, pareto_frontier  # noqa: E402


MILESTONES = (
    "body:select",
    "body:gather",
    "body:hash",
    "body:index",
    "final_drain:select",
    "final_drain:gather",
    "final_drain:hash",
    "store_tail:store",
)


def evaluate_detailed(name: str, variant: dict[str, Any]) -> dict[str, Any]:
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

    summary = kb.ir_schedule_summary
    return {
        "name": name,
        "ok": ok,
        "cycles": machine.cycle,
        "scratch": kb.scratch_ptr,
        "engine_slots": dict(engine_slots),
        "engine_active": dict(engine_active),
        "last_by_region_tag": summary.get("last_by_region_tag", {}),
        "last_final_hash_cycle": summary.get("last_final_hash_cycle"),
        "last_final_store_cycle": summary.get("last_final_store_cycle"),
        "ir_cycles": summary.get("cycles"),
        "variant": variant,
    }


def diff_dict(
    candidate: dict[str, int],
    baseline: dict[str, int],
    keys: tuple[str, ...] | list[str],
) -> dict[str, int]:
    return {
        key: candidate.get(key, 0) - baseline.get(key, 0)
        for key in keys
        if candidate.get(key, 0) != baseline.get(key, 0)
    }


def milestone_diff(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, int]:
    candidate_tags = row.get("last_by_region_tag", {})
    baseline_tags = baseline.get("last_by_region_tag", {})
    return {
        key: candidate_tags.get(key, -1) - baseline_tags.get(key, -1)
        for key in MILESTONES
        if candidate_tags.get(key, -1) != baseline_tags.get(key, -1)
    }


def summarize(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    engine_keys = sorted(set(row["engine_active"]) | set(baseline["engine_active"]))
    slot_keys = sorted(set(row["engine_slots"]) | set(baseline["engine_slots"]))
    return {
        "name": row["name"],
        "cycles": row["cycles"],
        "cycle_delta": row["cycles"] - baseline["cycles"],
        "valu_active": row["engine_active"].get("valu"),
        "valu_active_delta": row["engine_active"].get("valu", 0)
        - baseline["engine_active"].get("valu", 0),
        "valu_slots": row["engine_slots"].get("valu"),
        "valu_slot_delta": row["engine_slots"].get("valu", 0)
        - baseline["engine_slots"].get("valu", 0),
        "last_final_hash_delta": (row["last_final_hash_cycle"] or 0)
        - (baseline["last_final_hash_cycle"] or 0),
        "last_final_store_delta": (row["last_final_store_cycle"] or 0)
        - (baseline["last_final_store_cycle"] or 0),
        "engine_active_delta": diff_dict(row["engine_active"], baseline["engine_active"], engine_keys),
        "engine_slot_delta": diff_dict(row["engine_slots"], baseline["engine_slots"], slot_keys),
        "milestone_delta": milestone_diff(row, baseline),
        "variant": row["variant"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, variant in candidates():
        key = json.dumps(variant, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        try:
            row = evaluate_detailed(name, variant)
        except Exception as exc:  # noqa: BLE001 - analyzer should report bad variants.
            row = {
                "name": name,
                "ok": False,
                "cycles": None,
                "scratch": None,
                "engine_slots": {},
                "engine_active": {},
                "last_by_region_tag": {},
                "last_final_hash_cycle": None,
                "last_final_store_cycle": None,
                "variant": variant,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)

    baseline = next(row for row in rows if row["name"] == "baseline" and row["ok"])
    good = [row for row in rows if row["ok"]]

    print("# baseline")
    print(json.dumps(summarize(baseline, baseline), sort_keys=True))

    print("# best_cycles")
    for row in sorted(good, key=lambda item: (item["cycles"], item["last_final_store_cycle"]))[
        : args.top
    ]:
        print(json.dumps(summarize(row, baseline), sort_keys=True))

    print("# lowest_valu_slots")
    for row in sorted(
        good,
        key=lambda item: (
            item["engine_slots"].get("valu", 10**9),
            item["cycles"],
            item["last_final_store_cycle"],
        ),
    )[: args.top]:
        print(json.dumps(summarize(row, baseline), sort_keys=True))

    print("# pareto_frontier")
    # Reuse the same frontier definition as the search tool by passing rows with
    # the expected top-level metrics.
    frontier = pareto_frontier(good)
    for row in frontier[: args.top]:
        print(json.dumps(summarize(row, baseline), sort_keys=True))

    losing_low_valu = [
        row
        for row in good
        if row["engine_slots"].get("valu", 10**9) < baseline["engine_slots"].get("valu", 0)
        and row["cycles"] > baseline["cycles"]
    ]
    print("# low_valu_but_slower")
    for row in sorted(
        losing_low_valu,
        key=lambda item: (
            item["engine_slots"].get("valu", 10**9),
            item["cycles"],
        ),
    )[: args.top]:
        print(json.dumps(summarize(row, baseline), sort_keys=True))

    print(
        json.dumps(
            {
                "count": len(rows),
                "good": len(good),
                "frontier": len(frontier),
                "low_valu_but_slower": len(losing_low_valu),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
