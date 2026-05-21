#!/usr/bin/env python3
"""Analyze load pressure and late load readiness near the final region."""

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
from perf_takehome import KernelBuilder, SLOT_LIMITS  # noqa: E402


VARIANTS = {
    "baseline": {},
    "low_valu": {
        "simd_hash_scalar_by_round": {
            5: {"h1": [3], "h2": [3], "combine": []},
            6: {"h1": [3], "h2": [], "combine": [3, 5]},
            8: {"h1": [3], "h2": [5], "combine": []},
        }
    },
    "mid_valu": {
        "simd_hash_scalar_by_round": {
            6: {"h1": [3], "h2": [], "combine": [3, 5]},
        }
    },
}


def evaluate(name: str, variant: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "name": name,
        "ok": ok,
        "cycles": machine.cycle,
        "kb": kb,
        "variant": variant,
    }


def load_cycle_usage(instrs: list[dict[str, list[tuple]]]) -> dict[str, Any]:
    usage = []
    saturated = []
    for cycle, instr in enumerate(instrs):
        slots = len(instr.get("load", []))
        if slots:
            usage.append({"cycle": cycle, "slots": slots})
        if slots >= SLOT_LIMITS["load"]:
            saturated.append(cycle)
    return {
        "active_cycles": len(usage),
        "total_slots": sum(row["slots"] for row in usage),
        "saturated_cycles": saturated,
        "saturated_count": len(saturated),
    }


def window_stats(usage: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    saturated = [cycle for cycle in usage["saturated_cycles"] if start <= cycle <= end]
    active = [
        row for row in usage["usage"] if start <= row["cycle"] <= end
    ] if "usage" in usage else []
    return {
        "start": start,
        "end": end,
        "active_count": len(active),
        "slot_count": sum(row["slots"] for row in active),
        "saturated_count": len(saturated),
        "saturated_cycles": saturated[:40],
    }


def latest_load_ops(kb: KernelBuilder, limit: int) -> list[dict[str, Any]]:
    rows = []
    for op_id, (op, cycle) in enumerate(zip(kb.ir_ops, kb.ir_op_cycles, strict=True)):
        if op.engine != "load" or cycle < 0:
            continue
        rows.append(
            {
                "op_id": op_id,
                "cycle": cycle,
                "slot_op": op.slot[0],
                "tag": op.tag,
                "region": op.region,
                "round": op.round,
                "level": op.level,
                "group": op.group,
                "reads": len(op.reads),
                "writes": len(op.writes),
            }
        )
    return sorted(rows, key=lambda row: (row["cycle"], row["op_id"]), reverse=True)[:limit]


def summarize(row: dict[str, Any], limit: int) -> dict[str, Any]:
    kb: KernelBuilder = row["kb"]
    usage = load_cycle_usage(kb.instrs)
    usage["usage"] = [
        {"cycle": cycle, "slots": len(instr.get("load", []))}
        for cycle, instr in enumerate(kb.instrs)
        if instr.get("load")
    ]
    summary = kb.ir_schedule_summary
    final_gather = summary.get("last_by_region_tag", {}).get("final_drain:gather")
    final_hash = summary.get("last_final_hash_cycle")
    final_store = summary.get("last_final_store_cycle")
    windows = {}
    for name, center in (
        ("last_80_cycles", row["cycles"] - 80),
        ("final_gather_to_hash", final_gather),
        ("final_hash_to_store", final_hash),
    ):
        if center is None:
            continue
        if name == "last_80_cycles":
            windows[name] = window_stats(usage, max(0, center), row["cycles"])
        elif name == "final_gather_to_hash":
            windows[name] = window_stats(usage, final_gather - 20, (final_hash or final_gather) + 5)
        else:
            windows[name] = window_stats(usage, final_hash - 10, (final_store or final_hash) + 5)

    return {
        "name": row["name"],
        "ok": row["ok"],
        "cycles": row["cycles"],
        "engine_active": summary.get("engine_active"),
        "engine_slots": summary.get("engine_slots"),
        "last_by_region_tag": summary.get("last_by_region_tag"),
        "last_final_hash_cycle": final_hash,
        "last_final_store_cycle": final_store,
        "load_pressure": {
            "active_cycles": usage["active_cycles"],
            "total_slots": usage["total_slots"],
            "saturated_count": usage["saturated_count"],
            "last_saturated_cycles": usage["saturated_cycles"][-limit:],
            "windows": windows,
        },
        "latest_load_ops": latest_load_ops(kb, limit),
        "variant": row["variant"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--variant-json", help="Analyze one custom variant")
    args = parser.parse_args()

    variants = (
        {"custom": json.loads(args.variant_json)}
        if args.variant_json
        else VARIANTS
    )
    for name, variant in variants.items():
        row = evaluate(name, variant)
        print(json.dumps(summarize(row, args.limit), sort_keys=True))


if __name__ == "__main__":
    main()
