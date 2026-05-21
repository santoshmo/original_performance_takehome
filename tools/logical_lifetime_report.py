#!/usr/bin/env python3
"""Inspect logical value lifetimes inside context vector allocations."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
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


def build_kernel() -> KernelBuilder:
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)
    return kb


def context_vector_allocations(kb: KernelBuilder) -> dict[str, dict[int, dict[str, Any]]]:
    arrays: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for allocation in kb.scratch_allocations.values():
        match = CONTEXT_ARRAY_RE.match(allocation.name)
        if not match or allocation.length != VLEN:
            continue
        prefix = match.group("prefix")
        idx = int(match.group("idx"))
        arrays[prefix][idx] = {
            "base": allocation.base,
            "addresses": set(range(allocation.base, allocation.base + VLEN)),
            "role": allocation.role,
            "pool": allocation.pool,
        }
    return arrays


def op_label(op: Any) -> dict[str, Any]:
    return {
        "engine": op.engine,
        "tag": op.tag,
        "region": op.region,
        "round": op.round,
        "group": op.group,
    }


def op_markers(ops: list[Any]) -> dict[int, dict[str, int]]:
    markers: dict[int, dict[str, int]] = defaultdict(dict)
    for op_id, op in enumerate(ops):
        if op.group is None:
            continue
        if op.round == 14 and op.tag == "index":
            markers[op.group]["round14_index_last"] = op_id
        if op.round == 15 and op.tag == "hash":
            markers[op.group].setdefault("round15_hash_first", op_id)
        if op.round == 15 and op.tag == "gather":
            markers[op.group].setdefault("round15_gather_first", op_id)
            markers[op.group]["round15_gather_last"] = op_id
    return markers


def touch_rows(ops: list[Any], start: int, end: int, addresses: set[int]) -> list[dict[str, Any]]:
    rows = []
    for op_id in range(start + 1, end):
        op = ops[op_id]
        reads = op.reads & addresses
        writes = op.writes & addresses
        if not reads and not writes:
            continue
        rows.append(
            {
                "op_id": op_id,
                **op_label(op),
                "read_lanes": len(reads),
                "write_lanes": len(writes),
            }
        )
    return rows


def classify_touch(owner_block: int, ctx_idx: int, touch: dict[str, Any], group_size: int) -> str:
    group = touch["group"]
    if group == owner_block:
        return "same_block"
    if group is not None and group % group_size == ctx_idx:
        return "same_context_reuse"
    if group is None:
        return "non_group"
    return "other_group"


def first_blockers_by_array(
    ops: list[Any],
    arrays: dict[str, dict[int, dict[str, Any]]],
    *,
    group_size: int,
) -> dict[str, Any]:
    markers = op_markers(ops)
    report = {}
    full_arrays = {
        prefix: entries
        for prefix, entries in arrays.items()
        if sorted(entries) == list(range(group_size))
    }
    for prefix, entries in sorted(full_arrays.items()):
        first_blocker_classes: Counter[str] = Counter()
        first_blocker_tags: Counter[str] = Counter()
        blocked_windows = 0
        samples = []
        for block, block_markers in sorted(markers.items()):
            start = block_markers.get("round14_index_last")
            end = block_markers.get("round15_hash_first")
            if start is None or end is None or end <= start:
                continue
            ctx_idx = block % group_size
            entry = entries[ctx_idx]
            touches = touch_rows(ops, start, end, entry["addresses"])
            if not touches:
                continue
            blocked_windows += 1
            first = touches[0]
            blocker_class = classify_touch(block, ctx_idx, first, group_size)
            first_blocker_classes[blocker_class] += 1
            first_blocker_tags[f"{first['round']}:{first['tag']}:{first['engine']}"] += 1
            if len(samples) < 8:
                samples.append(
                    {
                        "block": block,
                        "context": ctx_idx,
                        "start_op": start,
                        "end_op": end,
                        "first_blocker_class": blocker_class,
                        "first_blocker": first,
                    }
                )
        report[prefix] = {
            "role": next(iter(entries.values()))["role"],
            "pool": next(iter(entries.values()))["pool"],
            "blocked_windows": blocked_windows,
            "first_blocker_classes": dict(first_blocker_classes),
            "first_blocker_tags": dict(first_blocker_tags),
            "samples": samples,
        }
    return report


def write_runs(ops: list[Any], addresses: set[int]) -> list[dict[str, Any]]:
    runs = []
    current = None
    previous_op_id = None
    for op_id, op in enumerate(ops):
        writes = op.writes & addresses
        if not writes:
            continue
        key = (op.round, op.group, op.tag, op.engine)
        if current is not None and current["key"] == key and previous_op_id == op_id - 1:
            current["end_write_op"] = op_id
            current["write_lanes"] += len(writes)
        else:
            if current is not None:
                runs.append(current)
            current = {
                "key": key,
                "start_write_op": op_id,
                "end_write_op": op_id,
                "write_lanes": len(writes),
                **op_label(op),
            }
        previous_op_id = op_id
    if current is not None:
        runs.append(current)
    return runs


def generation_stats_for_allocation(ops: list[Any], addresses: set[int]) -> dict[str, Any]:
    runs = write_runs(ops, addresses)
    generations = []
    for idx, run in enumerate(runs):
        next_start = runs[idx + 1]["start_write_op"] if idx + 1 < len(runs) else len(ops)
        last_read = None
        read_count = 0
        for op_id in range(run["end_write_op"] + 1, next_start):
            if ops[op_id].reads & addresses:
                last_read = op_id
                read_count += 1
        end = last_read if last_read is not None else run["end_write_op"]
        generations.append(
            {
                **run,
                "end_op": end,
                "duration": end - run["start_write_op"] + 1,
                "read_count": read_count,
            }
        )

    durations = [generation["duration"] for generation in generations]
    by_tag = Counter(f"{generation['round']}:{generation['tag']}" for generation in generations)
    if not durations:
        return {"generations": 0}
    return {
        "generations": len(generations),
        "min_duration": min(durations),
        "median_duration": statistics.median(durations),
        "max_duration": max(durations),
        "by_round_tag": dict(by_tag),
        "longest_generations": sorted(
            generations,
            key=lambda generation: generation["duration"],
            reverse=True,
        )[:5],
    }


def logical_generation_summary(
    ops: list[Any],
    arrays: dict[str, dict[int, dict[str, Any]]],
    *,
    group_size: int,
) -> dict[str, Any]:
    summary = {}
    for prefix, entries in sorted(arrays.items()):
        if sorted(entries) != list(range(group_size)):
            continue
        per_context = {
            str(idx): generation_stats_for_allocation(ops, entry["addresses"])
            for idx, entry in sorted(entries.items())
        }
        generation_counts = [
            row.get("generations", 0) for row in per_context.values()
        ]
        max_durations = [
            row.get("max_duration", 0) for row in per_context.values()
        ]
        summary[prefix] = {
            "role": next(iter(entries.values()))["role"],
            "pool": next(iter(entries.values()))["pool"],
            "generation_count_range": [min(generation_counts), max(generation_counts)],
            "max_duration_range": [min(max_durations), max(max_durations)],
            "sample_contexts": {
                key: per_context[key]
                for key in sorted(per_context, key=int)[:2]
            },
        }
    return summary


def main() -> None:
    kb = build_kernel()
    arrays = context_vector_allocations(kb)
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
        "first_blockers_by_array": first_blockers_by_array(
            kb.ir_ops,
            arrays,
            group_size=16,
        ),
        "logical_generations": logical_generation_summary(
            kb.ir_ops,
            arrays,
            group_size=16,
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
