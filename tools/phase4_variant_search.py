#!/usr/bin/env python3
"""Search Phase 4 IR variant families without modifying tests."""

from __future__ import annotations

from collections import Counter
from itertools import product
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

    return {
        "ok": ok,
        "cycles": machine.cycle,
        "scratch": kb.scratch_ptr,
        "variant": variant,
        "engine_slots": dict(engine_slots),
        "engine_active": dict(engine_active),
        "ir_summary": kb.ir_schedule_summary,
    }


def candidate_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [{}]

    for scheduler in ("fifo", "critical_path", "tail_weighted", "engine_balanced"):
        variants.append({"ir_scheduler": scheduler})

    for order in ("forward", "reverse", "even_odd", "odd_even"):
        variants.append(
            {
                "split_final_drain": True,
                "final_group_order": order,
            }
        )

    for rotation in range(0, 256, 16):
        variants.append(
            {
                "split_final_drain": True,
                "final_group_order": "rotate",
                "final_group_rotation": rotation,
            }
        )

    for single_temp_hash, drop_final_index, prune_droppable_final in product(
        (False, True), repeat=3
    ):
        variants.append(
            {
                "single_temp_hash": single_temp_hash,
                "drop_final_index": drop_final_index,
                "prune_droppable_final": prune_droppable_final,
                "emit_debug_pauses": not drop_final_index,
            }
        )

    return variants


def main() -> None:
    results = []
    seen: set[str] = set()
    for variant in candidate_variants():
        key = json.dumps(variant, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        try:
            results.append(evaluate(variant))
        except Exception as exc:  # noqa: BLE001 - search should report bad candidates.
            results.append(
                {
                    "ok": False,
                    "cycles": None,
                    "scratch": None,
                    "variant": variant,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    def sort_key(row: dict[str, Any]) -> tuple[int, int]:
        return (0 if row["ok"] else 1, row["cycles"] if row["cycles"] is not None else 10**12)

    for row in sorted(results, key=sort_key)[:20]:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
