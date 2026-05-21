#!/usr/bin/env python3
"""Trace dependency chains into final hash/store tail operations."""

from __future__ import annotations

from collections import Counter, defaultdict
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
from perf_takehome import DependencyEdge, KernelBuilder, KernelOp  # noqa: E402
from pareto_policy_search import candidates  # noqa: E402


INTERESTING_VARIANTS = {
    "baseline": {},
    "r6_c_35": {
        "simd_hash_scalar_by_round": {6: {"h1": [3], "h2": [], "combine": [3, 5]}}
    },
    "r5_h2_3+r8_h2_5+r6_c_35": {
        "simd_hash_scalar_by_round": {
            5: {"h1": [3], "h2": [3], "combine": []},
            6: {"h1": [3], "h2": [], "combine": [3, 5]},
            8: {"h1": [3], "h2": [5], "combine": []},
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

    engine_slots: Counter[str] = Counter()
    engine_active: Counter[str] = Counter()
    for instr in kb.instrs:
        for engine, slots in instr.items():
            if engine == "debug":
                continue
            engine_slots[engine] += len(slots)
            engine_active[engine] += 1

    return {
        "name": name,
        "ok": ok,
        "cycles": machine.cycle,
        "scratch": kb.scratch_ptr,
        "engine_slots": dict(engine_slots),
        "engine_active": dict(engine_active),
        "summary": kb.ir_schedule_summary,
        "ops": kb.ir_ops,
        "edges": kb.ir_dependency_edges,
        "op_cycles": kb.ir_op_cycles,
        "variant": variant,
    }


def op_label(op_id: int, op: KernelOp, cycle: int) -> dict[str, Any]:
    return {
        "op_id": op_id,
        "cycle": cycle,
        "engine": op.engine,
        "slot_op": op.slot[0],
        "tag": op.tag,
        "region": op.region,
        "round": op.round,
        "level": op.level,
        "group": op.group,
        "value": op.contributes_value,
        "index": op.contributes_index,
        "reads": len(op.reads),
        "writes": len(op.writes),
    }


def target_ops(row: dict[str, Any], target: str) -> list[int]:
    ops: list[KernelOp] = row["ops"]
    cycles: list[int] = row["op_cycles"]
    rounds = [op.round for op in ops if op.round is not None]
    final_round = max(rounds)
    if target == "final_hash":
        ids = [
            op_id
            for op_id, op in enumerate(ops)
            if op.round == final_round and op.tag == "hash" and op.contributes_value
        ]
    elif target == "final_store":
        ids = [
            op_id
            for op_id, op in enumerate(ops)
            if op.round == final_round and op.tag == "store" and op.contributes_value
        ]
    else:
        raise ValueError(f"Unknown target {target!r}")
    return sorted(ids, key=lambda op_id: (cycles[op_id], op_id), reverse=True)


def predecessor_edges(edges: list[DependencyEdge]) -> dict[int, list[DependencyEdge]]:
    preds: dict[int, list[DependencyEdge]] = defaultdict(list)
    for edge in edges:
        preds[edge.succ].append(edge)
    return preds


def edge_priority(edge: DependencyEdge, cycles: list[int]) -> tuple[int, int, int]:
    kind_rank = {"RAW": 3, "WAW": 2, "MEM": 1, "WAR": 0}.get(edge.kind, 0)
    return (cycles[edge.pred] + edge.latency, cycles[edge.pred], kind_rank)


def causal_chain(
    row: dict[str, Any],
    target: str,
    max_depth: int,
    *,
    include_war: bool,
) -> dict[str, Any]:
    ops: list[KernelOp] = row["ops"]
    edges: list[DependencyEdge] = row["edges"]
    cycles: list[int] = row["op_cycles"]
    preds = predecessor_edges(edges)
    targets = target_ops(row, target)
    if not targets:
        return {"target": target, "chain": [], "blocking_predecessors": []}

    current = targets[0]
    chain = []
    edge_to_current: DependencyEdge | None = None
    seen = set()
    for _ in range(max_depth):
        if current in seen:
            break
        seen.add(current)
        entry = op_label(current, ops[current], cycles[current])
        if edge_to_current is not None:
            entry["edge_to_next"] = {
                "kind": edge_to_current.kind,
                "latency": edge_to_current.latency,
                "resource": edge_to_current.resource,
            }
        chain.append(entry)
        incoming = [
            edge
            for edge in preds.get(current, [])
            if include_war or edge.kind != "WAR"
        ]
        if not incoming:
            break
        edge_to_current = max(incoming, key=lambda edge: edge_priority(edge, cycles))
        current = edge_to_current.pred

    blocking = []
    for edge in sorted(
        [
            edge
            for edge in preds.get(targets[0], [])
            if include_war or edge.kind != "WAR"
        ],
        key=lambda item: edge_priority(item, cycles),
        reverse=True,
    )[:10]:
        blocking.append(
            {
                **op_label(edge.pred, ops[edge.pred], cycles[edge.pred]),
                "edge": {
                    "kind": edge.kind,
                    "latency": edge.latency,
                    "resource": edge.resource,
                    "ready_cycle": cycles[edge.pred] + edge.latency,
                },
            }
        )

    return {
        "target": target,
        "target_op": op_label(targets[0], ops[targets[0]], cycles[targets[0]]),
        "chain_from_target_back": chain,
        "chain_from_root_to_target": list(reversed(chain)),
        "blocking_predecessors": blocking,
    }


def comparable_summary(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    summary = row["summary"]
    base_summary = baseline["summary"]
    active = row["engine_active"]
    base_active = baseline["engine_active"]
    slots = row["engine_slots"]
    base_slots = baseline["engine_slots"]
    milestones = summary.get("last_by_region_tag", {})
    base_milestones = base_summary.get("last_by_region_tag", {})
    milestone_delta = {
        key: milestones.get(key, -1) - base_milestones.get(key, -1)
        for key in sorted(set(milestones) | set(base_milestones))
        if milestones.get(key, -1) != base_milestones.get(key, -1)
    }
    return {
        "name": row["name"],
        "ok": row["ok"],
        "cycles": row["cycles"],
        "cycle_delta": row["cycles"] - baseline["cycles"],
        "valu_active_delta": active.get("valu", 0) - base_active.get("valu", 0),
        "valu_slot_delta": slots.get("valu", 0) - base_slots.get("valu", 0),
        "alu_active_delta": active.get("alu", 0) - base_active.get("alu", 0),
        "alu_slot_delta": slots.get("alu", 0) - base_slots.get("alu", 0),
        "last_hash_delta": (summary.get("last_final_hash_cycle") or 0)
        - (base_summary.get("last_final_hash_cycle") or 0),
        "last_store_delta": (summary.get("last_final_store_cycle") or 0)
        - (base_summary.get("last_final_store_cycle") or 0),
        "milestone_delta": milestone_delta,
        "variant": row["variant"],
    }


def selected_variants(mode: str) -> list[tuple[str, dict[str, Any]]]:
    if mode == "interesting":
        return list(INTERESTING_VARIANTS.items())
    seen = set()
    selected = []
    for name, variant in candidates():
        key = json.dumps(variant, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        selected.append((name, variant))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("interesting", "all"), default="interesting")
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--include-war", action="store_true")
    args = parser.parse_args()

    rows = [evaluate(name, variant) for name, variant in selected_variants(args.mode)]
    baseline = next(row for row in rows if row["name"] == "baseline")

    for row in rows:
        print("# candidate")
        print(json.dumps(comparable_summary(row, baseline), sort_keys=True))
        print("# final_hash_chain")
        print(
            json.dumps(
                causal_chain(
                    row,
                    "final_hash",
                    args.max_depth,
                    include_war=args.include_war,
                ),
                sort_keys=True,
            )
        )
        print("# final_store_chain")
        print(
            json.dumps(
                causal_chain(
                    row,
                    "final_store",
                    args.max_depth,
                    include_war=args.include_war,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
