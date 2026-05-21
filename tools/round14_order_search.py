#!/usr/bin/env python3
"""Focused search for pre-drain round-14 group ordering."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from phase4_variant_search import evaluate  # noqa: E402


GROUP_SIZE = 16


def swapped_sequence(left: int) -> list[int]:
    sequence = list(range(GROUP_SIZE))
    sequence[left], sequence[left + 1] = sequence[left + 1], sequence[left]
    return sequence


def move_sequence(src: int, dst: int) -> list[int]:
    sequence = list(range(GROUP_SIZE))
    item = sequence.pop(src)
    sequence.insert(dst, item)
    return sequence


def split_round14_variants() -> Iterable[tuple[str, dict[str, Any]]]:
    base = {
        "simd_predrain_split_rounds": [14],
        "simd_predrain_order_rounds": [14],
    }

    yield "split_forward", dict(base)
    for order in ("reverse", "even_odd", "odd_even", "odd_even_swap_3_4"):
        yield f"order_{order}", {**base, "simd_predrain_group_order": order}

    for rotation in range(GROUP_SIZE):
        yield (
            f"rotate_{rotation}",
            {
                **base,
                "simd_predrain_group_order": "rotate",
                "simd_predrain_group_rotation": rotation,
            },
        )

    for left in range(GROUP_SIZE - 1):
        yield (
            f"swap_{left}_{left + 1}",
            {
                **base,
                "simd_predrain_group_sequence": swapped_sequence(left),
            },
        )

    for src in range(GROUP_SIZE - 4, GROUP_SIZE):
        for dst in range(4):
            yield (
                f"move_{src}_to_{dst}",
                {
                    **base,
                    "simd_predrain_group_sequence": move_sequence(src, dst),
                },
            )


def tile13_variants() -> Iterable[tuple[str, dict[str, Any]]]:
    base = {"simd_predrain_order_rounds": [13]}

    yield "tile13_forward", {**base, "simd_predrain_group_order": "forward"}
    for order in ("reverse", "even_odd", "odd_even"):
        yield f"tile13_{order}", {**base, "simd_predrain_group_order": order}

    for rotation in range(GROUP_SIZE):
        yield (
            f"tile13_rotate_{rotation}",
            {
                **base,
                "simd_predrain_group_order": "rotate",
                "simd_predrain_group_rotation": rotation,
            },
        )


def candidate_variants() -> Iterable[tuple[str, dict[str, Any]]]:
    yield "current_default", {}
    yield from tile13_variants()
    yield from split_round14_variants()


def main() -> None:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, variant in candidate_variants():
        key = json.dumps(variant, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        error = None
        try:
            result = evaluate(variant)
        except Exception as exc:  # noqa: BLE001 - search should report bad variants.
            error = f"{type(exc).__name__}: {exc}"
            result = {
                "ok": False,
                "cycles": None,
                "scratch": None,
                "variant": variant,
            }
        summary = result.get("ir_summary", {})
        result = {
            "name": name,
            "ok": result["ok"],
            "cycles": result["cycles"],
            "scratch": result["scratch"],
            "last_final_hash_cycle": summary.get("last_final_hash_cycle"),
            "last_final_store_cycle": summary.get("last_final_store_cycle"),
            "engine_active": result.get("engine_active"),
            "variant": variant,
        }
        if error is not None:
            result["error"] = error
        results.append(result)

    def sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
        cycles = row["cycles"] if row["cycles"] is not None else 10**12
        final_store = (
            row["last_final_store_cycle"]
            if row["last_final_store_cycle"] is not None
            else 10**12
        )
        return (0 if row["ok"] else 1, cycles, final_store)

    for row in sorted(results, key=sort_key)[:25]:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
