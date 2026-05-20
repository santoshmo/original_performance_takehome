#!/usr/bin/env python3
"""Compare explicit KernelBuilder deps against read/write inferred deps."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perf_takehome import KernelBuilder  # noqa: E402
from problem import VLEN  # noqa: E402


Edge = tuple[int, int]


def _vec(addr: int) -> set[int]:
    return set(range(addr, addr + VLEN))


def read_write_sets(engine: str, slot: tuple) -> tuple[set[int], set[int]]:
    """Return scratch read/write addresses for a slot.

    Memory aliases are not modeled because generated programs do not use memory
    as temporary communication between instructions.
    """
    op = slot[0]
    reads: set[int] = set()
    writes: set[int] = set()

    if engine == "alu":
        _op, dest, a, b = slot
        writes.add(dest)
        reads.update([a, b])
    elif engine == "valu":
        if op == "vbroadcast":
            _op, dest, src = slot
            writes.update(_vec(dest))
            reads.add(src)
        elif op == "multiply_add":
            _op, dest, a, b, c = slot
            writes.update(_vec(dest))
            reads.update(_vec(a))
            reads.update(_vec(b))
            reads.update(_vec(c))
        else:
            _op, dest, a, b = slot
            writes.update(_vec(dest))
            reads.update(_vec(a))
            reads.update(_vec(b))
    elif engine == "load":
        if op == "const":
            _op, dest, _val = slot
            writes.add(dest)
        elif op == "load":
            _op, dest, addr = slot
            writes.add(dest)
            reads.add(addr)
        elif op == "load_offset":
            _op, dest, addr, offset = slot
            writes.add(dest + offset)
            reads.add(addr + offset)
        elif op == "vload":
            _op, dest, addr = slot
            writes.update(_vec(dest))
            reads.add(addr)
    elif engine == "store":
        if op == "store":
            _op, addr, src = slot
            reads.update([addr, src])
        elif op == "vstore":
            _op, addr, src = slot
            reads.add(addr)
            reads.update(_vec(src))
    elif engine == "flow":
        if op == "select":
            _op, dest, cond, a, b = slot
            writes.add(dest)
            reads.update([cond, a, b])
        elif op == "vselect":
            _op, dest, cond, a, b = slot
            writes.update(_vec(dest))
            reads.update(_vec(cond))
            reads.update(_vec(a))
            reads.update(_vec(b))
        elif op == "add_imm":
            _op, dest, a, _imm = slot
            writes.add(dest)
            reads.add(a)
        elif op in {"cond_jump", "cond_jump_rel", "jump_indirect", "trace_write"}:
            for value in slot[1:]:
                if isinstance(value, int):
                    reads.add(value)
        elif op in {"jump", "halt", "pause", "coreid"}:
            if op == "coreid":
                writes.add(slot[1])
    return reads, writes


def infer_edges(nodes: list[dict[str, Any]]) -> tuple[dict[Edge, set[str]], Counter[str]]:
    last_writer: dict[int, int] = {}
    readers_since_write: dict[int, list[int]] = defaultdict(list)
    inferred: dict[Edge, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()

    for node_id, node in enumerate(nodes):
        reads, writes = read_write_sets(node["engine"], node["slot"])

        for addr in reads:
            if addr in last_writer:
                if last_writer[addr] != node_id:
                    inferred[(last_writer[addr], node_id)].add("RAW")
                    counts["RAW"] += 1
            readers_since_write[addr].append(node_id)

        for addr in writes:
            if addr in last_writer:
                if last_writer[addr] != node_id:
                    inferred[(last_writer[addr], node_id)].add("WAW")
                    counts["WAW"] += 1
            for reader in readers_since_write[addr]:
                if reader != node_id:
                    inferred[(reader, node_id)].add("WAR0")
                    counts["WAR0"] += 1
            readers_since_write[addr].clear()
            last_writer[addr] = node_id

    return inferred, counts


def diagnose(variant: dict[str, Any]) -> dict[str, Any]:
    kb = KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16, variant=variant)
    nodes = kb.schedule_nodes_debug
    inferred, inferred_counts = infer_edges(nodes)

    explicit: set[Edge] = set()
    for node_id, node in enumerate(nodes):
        for dep in node["deps"]:
            explicit.add((dep, node_id))

    inferred_edges = set(inferred)
    redundant = explicit - inferred_edges
    missing = inferred_edges - explicit

    redundant_by_component: Counter[str] = Counter()
    missing_by_kind: Counter[str] = Counter()
    missing_by_component: Counter[str] = Counter()
    for pred, succ in redundant:
        redundant_by_component[f"{nodes[pred]['component']}->{nodes[succ]['component']}"] += 1
    for edge in missing:
        pred, succ = edge
        for kind in inferred[edge]:
            missing_by_kind[kind] += 1
        missing_by_component[f"{nodes[pred]['component']}->{nodes[succ]['component']}"] += 1

    return {
        "cycles": len(kb.instrs),
        "scratch": kb.scratch_ptr,
        "nodes": len(nodes),
        "explicit_edges": len(explicit),
        "inferred_edges": len(inferred_edges),
        "inferred_edge_events": dict(inferred_counts),
        "redundant_explicit_edges": len(redundant),
        "missing_inferred_edges": len(missing),
        "redundant_by_component": redundant_by_component.most_common(20),
        "missing_by_kind": dict(missing_by_kind),
        "missing_by_component": missing_by_component.most_common(20),
        "sample_redundant": [
            {
                "pred": pred,
                "succ": succ,
                "pred_component": nodes[pred]["component"],
                "succ_component": nodes[succ]["component"],
                "pred_engine": nodes[pred]["engine"],
                "succ_engine": nodes[succ]["engine"],
                "pred_slot": nodes[pred]["slot"],
                "succ_slot": nodes[succ]["slot"],
            }
            for pred, succ in sorted(redundant)[:20]
        ],
    }


def print_human(result: dict[str, Any]) -> None:
    print(
        "cycles",
        result["cycles"],
        "scratch",
        result["scratch"],
        "nodes",
        result["nodes"],
    )
    print("explicit_edges", result["explicit_edges"])
    print("inferred_edges", result["inferred_edges"], result["inferred_edge_events"])
    print("redundant_explicit_edges", result["redundant_explicit_edges"])
    print("missing_inferred_edges", result["missing_inferred_edges"], result["missing_by_kind"])
    print("\nredundant_by_component")
    for item, count in result["redundant_by_component"]:
        print(f"  {count:5} {item}")
    print("\nmissing_by_component")
    for item, count in result["missing_by_component"]:
        print(f"  {count:5} {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-json", default="{}", help="Variant dictionary as JSON")
    parser.add_argument("--json", action="store_true", help="Print full JSON")
    args = parser.parse_args()

    result = diagnose(json.loads(args.variant_json))
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print_human(result)


if __name__ == "__main__":
    main()
