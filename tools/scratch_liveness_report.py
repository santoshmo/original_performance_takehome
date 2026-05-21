#!/usr/bin/env python3
"""Report scratch allocation and IR liveness for selector-ring experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from frozen_problem import Input, Tree  # noqa: E402
from perf_takehome import KernelBuilder, VLEN  # noqa: E402


CONTEXT_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)$")


def normalize_name(name: str) -> tuple[str, int | None]:
    match = CONTEXT_RE.match(name)
    if not match:
        return name, None
    return match.group("prefix"), int(match.group("idx"))


def parse_variant(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw)


def build(variant: dict[str, Any]) -> KernelBuilder:
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16, variant=variant)
    return kb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-json", help="JSON variant override")
    args = parser.parse_args()

    variant = parse_variant(args.variant_json)
    kb = build(variant)

    role_words: Counter[str] = Counter()
    pool_words: Counter[str] = Counter()
    role_allocs: Counter[str] = Counter()
    vector_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    allocations = sorted(kb.scratch_allocations.values(), key=lambda alloc: alloc.base)

    for alloc in allocations:
        role_words[alloc.role] += alloc.length
        pool_words[alloc.pool or "none"] += alloc.length
        role_allocs[alloc.role] += 1
        prefix, idx = normalize_name(alloc.name)
        if idx is not None and alloc.length == VLEN:
            vector_groups[prefix].append(
                {
                    "idx": idx,
                    "base": alloc.base,
                    "role": alloc.role,
                    "pool": alloc.pool,
                    "persistent": alloc.persistent,
                }
            )

    context_arrays = {}
    for prefix, entries in sorted(vector_groups.items()):
        entries = sorted(entries, key=lambda row: row["idx"])
        context_arrays[prefix] = {
            "count": len(entries),
            "words": len(entries) * VLEN,
            "idxs": [row["idx"] for row in entries],
            "bases": [row["base"] for row in entries],
            "role": entries[0]["role"] if entries else None,
            "pool": entries[0]["pool"] if entries else None,
            "full_group16": len(entries) == 16,
        }

    alias_notes = {
        "selector0": "dedicated selector vector per context when depth >= 1",
        "selector1": "aliases select_tmp1/tmp3 by default when depth >= 2",
        "selector2": "aliases select_tmp0/tmp2 by default when depth >= 3",
        "hash_h2_tmp": "uses node temp by default when selector depth >= 3",
    }

    report = {
        "scratch_ptr": kb.scratch_ptr,
        "free_words": 1536 - kb.scratch_ptr,
        "role_words": dict(sorted(role_words.items())),
        "role_allocations": dict(sorted(role_allocs.items())),
        "pool_words": dict(sorted(pool_words.items())),
        "context_vector_arrays": context_arrays,
        "ir_liveness": kb.ir_liveness_summary,
        "schedule": {
            "cycles": kb.ir_schedule_summary.get("cycles"),
            "engine_active": kb.ir_schedule_summary.get("engine_active"),
            "engine_slots": kb.ir_schedule_summary.get("engine_slots"),
            "last_final_hash_cycle": kb.ir_schedule_summary.get("last_final_hash_cycle"),
            "last_final_store_cycle": kb.ir_schedule_summary.get("last_final_store_cycle"),
        },
        "alias_notes": alias_notes,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
