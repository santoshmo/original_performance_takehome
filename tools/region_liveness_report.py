#!/usr/bin/env python3
"""Report region-local liveness for prefetch scratch candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
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


CONTEXT_ARRAY_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)$")


def build_kernel(variant: dict[str, Any] | None = None) -> KernelBuilder:
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16, variant=variant)
    return kb


def touched(op: Any, addresses: set[int]) -> bool:
    return bool((op.reads | op.writes) & addresses)


def interval_touches(ops: list[Any], start: int, end: int, addresses: set[int]) -> list[dict[str, Any]]:
    """Return touches in the open interval (start, end)."""
    rows = []
    for op_id in range(start + 1, end):
        op = ops[op_id]
        reads = sorted(op.reads & addresses)
        writes = sorted(op.writes & addresses)
        if not reads and not writes:
            continue
        rows.append(
            {
                "op_id": op_id,
                "engine": op.engine,
                "tag": op.tag,
                "region": op.region,
                "round": op.round,
                "group": op.group,
                "reads": reads,
                "writes": writes,
            }
        )
    return rows


def context_arrays(kb: KernelBuilder) -> dict[str, dict[int, dict[str, Any]]]:
    arrays: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for alloc in kb.scratch_allocations.values():
        match = CONTEXT_ARRAY_RE.match(alloc.name)
        if not match or alloc.length != VLEN:
            continue
        prefix = match.group("prefix")
        idx = int(match.group("idx"))
        arrays[prefix][idx] = {
            "base": alloc.base,
            "addresses": set(range(alloc.base, alloc.base + alloc.length)),
            "role": alloc.role,
            "pool": alloc.pool,
        }
    return arrays


def op_index_by_block(ops: list[Any]) -> dict[int, dict[str, int]]:
    by_block: dict[int, dict[str, int]] = defaultdict(dict)
    for op_id, op in enumerate(ops):
        if op.group is None:
            continue
        if op.round == 14 and op.tag == "index":
            by_block[op.group]["round14_index_last"] = op_id
        if op.round == 15 and op.tag == "hash":
            by_block[op.group].setdefault("round15_hash_first", op_id)
        if op.round == 15 and op.tag == "gather":
            by_block[op.group].setdefault("round15_gather_first", op_id)
            by_block[op.group]["round15_gather_last"] = op_id
    return by_block


def summarize_window(
    ops: list[Any],
    arrays: dict[str, dict[int, dict[str, Any]]],
    *,
    group_size: int,
) -> dict[str, Any]:
    by_block = op_index_by_block(ops)
    full_arrays = {
        prefix: entries
        for prefix, entries in arrays.items()
        if sorted(entries) == list(range(group_size))
    }
    candidates: dict[str, dict[str, Any]] = {}
    context_reuse_hazards = []

    for prefix, entries in sorted(full_arrays.items()):
        blocked_windows = 0
        example_touches = []
        for block, markers in sorted(by_block.items()):
            start = markers.get("round14_index_last")
            end = markers.get("round15_hash_first")
            if start is None or end is None or end <= start:
                continue
            ctx_idx = block % group_size
            entry = entries.get(ctx_idx)
            if entry is None:
                continue
            touches_in_window = interval_touches(ops, start, end, entry["addresses"])
            if touches_in_window:
                blocked_windows += 1
                if len(example_touches) < 4:
                    example_touches.append(
                        {
                            "block": block,
                            "context": ctx_idx,
                            "start_op": start,
                            "end_op": end,
                            "touches": touches_in_window[:8],
                        }
                    )

        candidates[prefix] = {
            "role": next(iter(entries.values()))["role"],
            "pool": next(iter(entries.values()))["pool"],
            "words": len(entries) * VLEN,
            "blocked_windows": blocked_windows,
            "available_for_all_windows": blocked_windows == 0,
            "example_touches": example_touches,
        }

    for first_block in range(group_size):
        later_block = first_block + group_size
        if later_block not in by_block:
            continue
        first_prefetch = by_block[first_block].get("round14_index_last")
        first_final_hash = by_block[first_block].get("round15_hash_first")
        later_first_touch = min(
            (
                op_id
                for op_id, op in enumerate(ops)
                if op.group == later_block and op.round is not None and op.round < 15
            ),
            default=None,
        )
        if (
            first_prefetch is not None
            and first_final_hash is not None
            and later_first_touch is not None
            and first_prefetch < later_first_touch < first_final_hash
        ):
            context_reuse_hazards.append(
                {
                    "context": first_block,
                    "prefetch_owner_block": first_block,
                    "reused_by_block": later_block,
                    "prefetch_after_op": first_prefetch,
                    "reuse_op": later_first_touch,
                    "final_hash_op": first_final_hash,
                }
            )

    return {
        "candidate_full_vector_arrays": candidates,
        "context_reuse_hazards": context_reuse_hazards,
    }


def op_mix(ops: list[Any]) -> dict[str, Any]:
    by_region_tag = Counter(f"{op.region}:{op.tag}" for op in ops)
    by_round_tag = Counter(f"r{op.round}:{op.tag}" for op in ops if op.round is not None)
    return {
        "ops": len(ops),
        "by_region_tag": dict(sorted(by_region_tag.items())),
        "by_round_tag": dict(sorted(by_round_tag.items())),
    }


def main() -> None:
    kb = build_kernel()
    compact_kb = build_kernel({"region_local_compact_scratch": True})
    arrays = context_arrays(kb)
    report = {
        "scratch_ptr": kb.scratch_ptr,
        "free_words": 1536 - kb.scratch_ptr,
        "schedule": {
            "cycles": kb.ir_schedule_summary.get("cycles"),
            "engine_active": kb.ir_schedule_summary.get("engine_active"),
            "engine_slots": kb.ir_schedule_summary.get("engine_slots"),
            "last_final_hash_cycle": kb.ir_schedule_summary.get("last_final_hash_cycle"),
            "last_final_store_cycle": kb.ir_schedule_summary.get("last_final_store_cycle"),
        },
        "region_local_compaction": compact_kb.ir_schedule_summary.get(
            "region_local_allocation"
        ),
        "op_mix": op_mix(kb.ir_ops),
        "prefetch_window": summarize_window(kb.ir_ops, arrays, group_size=16),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
