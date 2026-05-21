"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import random
import unittest
from typing import Any

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


@dataclass(frozen=True)
class KernelOp:
    """Instruction-level IR node with scheduling metadata."""

    engine: str
    slot: tuple
    tag: str = "body"
    round: int | None = None
    level: int | None = None
    group: int | None = None
    lane: int | None = None
    contributes_value: bool = False
    contributes_index: bool = False
    droppable_final: bool = False
    region: str = "body"
    reads: frozenset[Any] = field(default_factory=frozenset)
    writes: frozenset[Any] = field(default_factory=frozenset)
    read_roles: frozenset[str] = field(default_factory=frozenset)
    write_roles: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class DependencyEdge:
    """Dependency edge between IR nodes.

    RAW/WAW edges have latency 1 because writes commit at cycle end. WAR edges
    have latency 0 because same-cycle read-before-write is legal on this
    machine.
    """

    pred: int
    succ: int
    latency: int
    kind: str
    resource: Any


@dataclass(frozen=True)
class ScratchAllocation:
    """Scratch allocation annotated with its logical compiler role."""

    base: int
    length: int
    name: str
    role: str
    pool: str | None = None
    persistent: bool = False


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.ir_ops: list[KernelOp] = []
        self.ir_dependency_edges: list[DependencyEdge] = []
        self.ir_schedule_summary: dict[str, Any] = {}
        self.ir_liveness_summary: dict[str, Any] = {}
        self.ir_tail_reschedule_summary: dict[str, Any] = {}
        self.region_local_allocation_summary: dict[str, Any] = {}
        self.scratch_allocations: dict[int, ScratchAllocation] = {}
        self.scratch_addr_to_alloc_base: dict[int, int] = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def _vec_range(self, base: int) -> range:
        return range(base, base + VLEN)

    def _slot_rw(self, engine: str, slot: tuple) -> tuple[set[int], set[int]]:
        """Return exact scratch reads and writes for a slot."""
        reads: set[int] = set()
        writes: set[int] = set()
        op = slot[0]

        if engine == "alu":
            _op, dest, a, b = slot
            writes.add(dest)
            reads.update((a, b))
        elif engine == "valu":
            if op == "vbroadcast":
                _op, dest, src = slot
                writes.update(self._vec_range(dest))
                reads.add(src)
            elif op == "multiply_add":
                _op, dest, a, b, c = slot
                writes.update(self._vec_range(dest))
                reads.update(self._vec_range(a))
                reads.update(self._vec_range(b))
                reads.update(self._vec_range(c))
            else:
                _op, dest, a, b = slot
                writes.update(self._vec_range(dest))
                reads.update(self._vec_range(a))
                reads.update(self._vec_range(b))
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
                writes.update(self._vec_range(dest))
                reads.add(addr)
        elif engine == "store":
            if op == "store":
                _op, addr, src = slot
                reads.update((addr, src))
            elif op == "vstore":
                _op, addr, src = slot
                reads.add(addr)
                reads.update(self._vec_range(src))
        elif engine == "flow":
            if op == "select":
                _op, dest, cond, a, b = slot
                writes.add(dest)
                reads.update((cond, a, b))
            elif op == "add_imm":
                _op, dest, a, _imm = slot
                writes.add(dest)
                reads.add(a)
            elif op == "vselect":
                _op, dest, cond, a, b = slot
                writes.update(self._vec_range(dest))
                reads.update(self._vec_range(cond))
                reads.update(self._vec_range(a))
                reads.update(self._vec_range(b))
        elif engine == "debug":
            if op == "compare":
                _op, loc, _key = slot
                reads.add(loc)
            elif op == "vcompare":
                _op, loc, _keys = slot
                reads.update(self._vec_range(loc))

        return reads, writes

    def make_ir_op(self, engine: str, slot: tuple, **metadata) -> KernelOp:
        reads, writes = self._slot_rw(engine, slot)
        return KernelOp(
            engine=engine,
            slot=slot,
            reads=frozenset(reads),
            writes=frozenset(writes),
            read_roles=frozenset(self.scratch_role_for_addr(addr) for addr in reads),
            write_roles=frozenset(self.scratch_role_for_addr(addr) for addr in writes),
            **metadata,
        )

    def scratch_role_for_addr(self, addr: int) -> str:
        alloc_base = self.scratch_addr_to_alloc_base.get(addr)
        if alloc_base is None:
            return "unknown"
        return self.scratch_allocations[alloc_base].role

    def analyze_ir_liveness(self, ops: list[KernelOp]) -> dict[str, Any]:
        """Summarize scratch liveness by logical role for allocator experiments."""
        intervals: dict[int, dict[str, Any]] = {}
        for op_id, op in enumerate(ops):
            touched = op.reads | op.writes
            for addr in touched:
                role = self.scratch_role_for_addr(addr)
                interval = intervals.setdefault(
                    addr,
                    {
                        "start": op_id,
                        "end": op_id,
                        "role": role,
                        "reads": 0,
                        "writes": 0,
                    },
                )
                interval["start"] = min(interval["start"], op_id)
                interval["end"] = max(interval["end"], op_id)
                if addr in op.reads:
                    interval["reads"] += 1
                if addr in op.writes:
                    interval["writes"] += 1

        role_addresses: dict[str, int] = defaultdict(int)
        role_intervals: dict[str, int] = defaultdict(int)
        live_events: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for addr, interval in intervals.items():
            role = interval["role"]
            role_addresses[role] += 1
            role_intervals[role] += 1
            live_events[role].append((interval["start"], 1))
            live_events[role].append((interval["end"] + 1, -1))

        max_live_by_role = {}
        for role, events in live_events.items():
            live = 0
            max_live = 0
            for _op_id, delta in sorted(events):
                live += delta
                max_live = max(max_live, live)
            max_live_by_role[role] = max_live

        allocations_by_role: dict[str, int] = defaultdict(int)
        words_by_role: dict[str, int] = defaultdict(int)
        pools_by_role: dict[str, set[str]] = defaultdict(set)
        for allocation in self.scratch_allocations.values():
            allocations_by_role[allocation.role] += 1
            words_by_role[allocation.role] += allocation.length
            if allocation.pool is not None:
                pools_by_role[allocation.role].add(allocation.pool)

        return {
            "addresses_by_role": dict(role_addresses),
            "intervals_by_role": dict(role_intervals),
            "max_live_addresses_by_role": max_live_by_role,
            "allocations_by_role": dict(allocations_by_role),
            "allocated_words_by_role": dict(words_by_role),
            "pools_by_role": {
                role: sorted(pools) for role, pools in pools_by_role.items()
            },
        }

    def _build_ir_dependencies(
        self,
        ops: list[KernelOp],
        preserve_memory_order: bool = True,
    ) -> tuple[list[list[tuple[int, int]]], list[DependencyEdge]]:
        edge_latency: dict[tuple[int, int], int] = {}
        edge_debug: dict[tuple[int, int], DependencyEdge] = {}
        last_writer: dict[Any, int] = {}
        readers_since_write: dict[Any, list[int]] = defaultdict(list)
        last_memory_user: int | None = None

        def add_edge(pred: int, succ: int, latency: int, kind: str, resource: Any):
            if pred == succ:
                return
            key = (pred, succ)
            if latency > edge_latency.get(key, -1):
                edge_latency[key] = latency
                edge_debug[key] = DependencyEdge(pred, succ, latency, kind, resource)

        for op_id, op in enumerate(ops):
            for addr in op.reads:
                if addr in last_writer:
                    add_edge(last_writer[addr], op_id, 1, "RAW", addr)
                readers_since_write[addr].append(op_id)

            for addr in op.writes:
                if addr in last_writer:
                    add_edge(last_writer[addr], op_id, 1, "WAW", addr)
                for reader in readers_since_write[addr]:
                    add_edge(reader, op_id, 0, "WAR", addr)
                readers_since_write[addr].clear()
                last_writer[addr] = op_id

            if preserve_memory_order and op.engine in ("load", "store") and op.slot[0] != "const":
                if last_memory_user is not None:
                    add_edge(last_memory_user, op_id, 1, "MEM", "memory")
                last_memory_user = op_id

        succ_edges: list[list[tuple[int, int]]] = [[] for _ in ops]
        for (pred, succ), latency in edge_latency.items():
            succ_edges[pred].append((succ, latency))

        return succ_edges, list(edge_debug.values())

    def _critical_heights(
        self, ops: list[KernelOp], succ_edges: list[list[tuple[int, int]]]
    ) -> list[int]:
        heights = [0] * len(ops)
        for op_id in range(len(ops) - 1, -1, -1):
            height = 0
            for succ, latency in succ_edges[op_id]:
                height = max(height, heights[succ] + 1 + latency)
            heights[op_id] = height
        return heights

    def _tail_scores(
        self,
        ops: list[KernelOp],
        *,
        tail_rounds: int = 3,
        final_store_bonus: int = 80,
        final_hash_bonus: int = 30,
        droppable_penalty: int = 40,
    ) -> list[int]:
        rounds = [op.round for op in ops if op.round is not None]
        if not rounds:
            return [0] * len(ops)
        max_round = max(rounds)
        tail_start = max(0, max_round - tail_rounds + 1)
        scores = [0] * len(ops)
        for op_id, op in enumerate(ops):
            if op.round is None or op.round < tail_start:
                continue
            round_weight = op.round - tail_start + 1
            scores[op_id] += 10 * round_weight
            if op.contributes_value:
                scores[op_id] += 10 * round_weight
            if op.tag == "hash" and op.contributes_value:
                scores[op_id] += final_hash_bonus
            if op.tag == "store" and op.contributes_value:
                scores[op_id] += final_store_bonus
            if op.droppable_final:
                scores[op_id] -= droppable_penalty
        return scores

    def _schedule_position_summary(
        self, ops: list[KernelOp], op_cycle: list[int]
    ) -> dict[str, Any]:
        """Summarize where important scheduled IR classes land."""
        rounds = [op.round for op in ops if op.round is not None]
        final_round = max(rounds) if rounds else None
        last_by_region_tag: dict[str, int] = {}
        last_final_hash = -1
        last_final_store = -1

        for op, cycle in zip(ops, op_cycle, strict=True):
            if cycle < 0:
                continue
            key = f"{op.region}:{op.tag}"
            last_by_region_tag[key] = max(last_by_region_tag.get(key, -1), cycle)
            if final_round is not None and op.round == final_round:
                if op.tag == "hash" and op.contributes_value:
                    last_final_hash = max(last_final_hash, cycle)
                elif op.tag == "store" and op.contributes_value:
                    last_final_store = max(last_final_store, cycle)

        return {
            "last_by_region_tag": last_by_region_tag,
            "last_final_hash_cycle": None if last_final_hash < 0 else last_final_hash,
            "last_final_store_cycle": None if last_final_store < 0 else last_final_store,
        }

    def _schedule_ir_asap(
        self,
        ops: list[KernelOp],
        mode_name: str,
        weights: dict[str, int] | None,
        succ_edges: list[list[tuple[int, int]]],
    ) -> list[dict[str, list[tuple]]]:
        """ASAP packer using IR read/write sets and zero-latency WAR."""
        weights = weights or {}
        cycles: list[dict[str, list[tuple]]] = []
        cycle_op_ids: list[list[int]] = []
        usage: list[dict[str, int]] = []
        ready_time: dict[int, int] = defaultdict(int)
        last_write: dict[int, int] = defaultdict(lambda: -1)
        last_read: dict[int, int] = defaultdict(lambda: -1)
        engine_slots = defaultdict(int)
        engine_active = defaultdict(int)
        self.ir_tail_reschedule_summary = {}

        def ensure_cycle(cycle: int) -> None:
            while len(cycles) <= cycle:
                cycles.append({})
                cycle_op_ids.append([])
                usage.append(defaultdict(int))

        def find_cycle(engine: str, earliest: int) -> int:
            cycle = earliest
            limit = SLOT_LIMITS[engine]
            while True:
                ensure_cycle(cycle)
                if usage[cycle][engine] < limit:
                    return cycle
                cycle += 1

        op_cycle = [-1] * len(ops)
        for op_id, op in enumerate(ops):
            earliest = 0
            for addr in op.reads:
                if isinstance(addr, int):
                    earliest = max(earliest, ready_time[addr])
            for addr in op.writes:
                if isinstance(addr, int):
                    earliest = max(earliest, last_write[addr] + 1, last_read[addr])

            cycle = find_cycle(op.engine, earliest)
            cycles[cycle].setdefault(op.engine, []).append(op.slot)
            cycle_op_ids[cycle].append(op_id)
            op_cycle[op_id] = cycle
            usage[cycle][op.engine] += 1

            for addr in op.reads:
                if isinstance(addr, int) and last_read[addr] < cycle:
                    last_read[addr] = cycle
            for addr in op.writes:
                if isinstance(addr, int):
                    last_write[addr] = cycle
                    ready_time[addr] = cycle + 1

        instrs = []
        nonempty_op_ids = []
        cycle_index: dict[int, int] = {}
        for cycle, instr in enumerate(cycles):
            if not instr:
                continue
            cycle_index[cycle] = len(instrs)
            instrs.append(instr)
            nonempty_op_ids.append(cycle_op_ids[cycle])
        compact_op_cycle = [
            cycle_index[cycle] if cycle >= 0 else -1
            for cycle in op_cycle
        ]
        tail_window = weights.get("tail_reschedule_window", 0)
        if tail_window:
            instrs = self._reschedule_tail_window(
                ops,
                instrs,
                nonempty_op_ids,
                op_cycle,
                succ_edges,
                max(0, len(instrs) - tail_window),
                weights,
            )
        for instr in instrs:
            for engine, slots in instr.items():
                engine_slots[engine] += len(slots)
                engine_active[engine] += 1

        position_summary = self._schedule_position_summary(ops, compact_op_cycle)
        self.ir_schedule_summary = {
            "mode": mode_name,
            "cycles": len(instrs),
            "ops": len(ops),
            "dependency_edges": len(self.ir_dependency_edges),
            "edge_kinds": {
                kind: sum(1 for edge in self.ir_dependency_edges if edge.kind == kind)
                for kind in ("RAW", "WAW", "WAR", "MEM")
            },
            "engine_slots": dict(engine_slots),
            "engine_active": dict(engine_active),
            "scratch_liveness": self.ir_liveness_summary,
            "tail_reschedule": self.ir_tail_reschedule_summary,
            "region_local_allocation": self.region_local_allocation_summary,
            **position_summary,
        }
        return instrs

    def _reschedule_tail_window(
        self,
        ops: list[KernelOp],
        instrs: list[dict[str, list[tuple]]],
        cycle_op_ids: list[list[int]],
        op_cycle: list[int],
        succ_edges: list[list[tuple[int, int]]],
        cutoff: int,
        weights: dict[str, int],
    ) -> list[dict[str, list[tuple]]]:
        """Freeze prefix bundles and reschedule only ops in the final window."""
        tail_ops = {
            op_id
            for cycle_ids in cycle_op_ids[cutoff:]
            for op_id in cycle_ids
        }
        if not tail_ops:
            self.ir_tail_reschedule_summary = {
                "status": "skipped_empty",
                "cutoff": cutoff,
            }
            return instrs

        pred_edges: list[list[tuple[int, int]]] = [[] for _ in ops]
        for pred, edges in enumerate(succ_edges):
            for succ, latency in edges:
                pred_edges[succ].append((pred, latency))

        critical_height = self._critical_heights(ops, succ_edges)
        tail_score = self._tail_scores(
            ops,
            tail_rounds=weights.get("tail_rounds", 3),
            final_store_bonus=weights.get("final_store_bonus", 80),
            final_hash_bonus=weights.get("final_hash_bonus", 30),
            droppable_penalty=weights.get("droppable_penalty", 40),
        )
        order = weights.get("tail_reschedule_order", "tail_weighted")

        wait_prev = {op_id: 0 for op_id in tail_ops}
        wait_same = {op_id: 0 for op_id in tail_ops}
        earliest = {op_id: cutoff for op_id in tail_ops}
        for op_id in tail_ops:
            for pred, latency in pred_edges[op_id]:
                if pred in tail_ops:
                    if latency:
                        wait_prev[op_id] += 1
                    else:
                        wait_same[op_id] += 1
                else:
                    earliest[op_id] = max(earliest[op_id], op_cycle[pred] + latency)

        ready = {
            engine: [
                op_id
                for op_id in tail_ops
                if ops[op_id].engine == engine
                and wait_prev[op_id] == 0
                and wait_same[op_id] == 0
            ]
            for engine in SLOT_LIMITS
        }
        scheduled: set[int] = set()
        tail_instrs: list[dict[str, list[tuple]]] = []
        engine_order = ("load", "valu", "alu", "store", "flow", "debug")

        def priority(op_id: int) -> tuple[int, int]:
            if order == "fifo":
                return (op_id, 0)
            if order == "store_first":
                store_bonus = 1000 if ops[op_id].tag == "store" else 0
                return (-(store_bonus + critical_height[op_id] + tail_score[op_id]), op_id)
            return (-(critical_height[op_id] + tail_score[op_id]), op_id)

        while len(scheduled) < len(tail_ops):
            cycle = cutoff + len(tail_instrs)
            instr: dict[str, list[tuple]] = {}
            chosen: list[int] = []
            chosen_set: set[int] = set()
            capacity = {engine: SLOT_LIMITS[engine] for engine in engine_order}

            progressed = True
            while progressed:
                progressed = False
                for engine in engine_order:
                    queue = ready[engine]
                    queue.sort(key=priority)
                    index = 0
                    while index < len(queue) and capacity[engine] > 0:
                        op_id = queue[index]
                        if op_id in scheduled or op_id in chosen_set:
                            queue.pop(index)
                            continue
                        if earliest[op_id] > cycle:
                            index += 1
                            continue
                        queue.pop(index)
                        instr.setdefault(engine, []).append(ops[op_id].slot)
                        chosen.append(op_id)
                        chosen_set.add(op_id)
                        capacity[engine] -= 1
                        progressed = True
                        for succ, latency in succ_edges[op_id]:
                            if succ in tail_ops and latency == 0:
                                wait_same[succ] -= 1
                                if (
                                    wait_prev[succ] == 0
                                    and wait_same[succ] == 0
                                    and succ not in scheduled
                                    and succ not in chosen_set
                                ):
                                    ready[ops[succ].engine].append(succ)

            if not chosen:
                # A gap cannot be represented as an empty bundle, so preserve
                # the original tail when this local schedule cannot progress.
                self.ir_tail_reschedule_summary = {
                    "status": "stalled",
                    "cutoff": cutoff,
                    "original_tail_cycles": len(instrs) - cutoff,
                    "scheduled_tail_ops": len(scheduled),
                    "tail_ops": len(tail_ops),
                    "order": order,
                }
                return instrs
            tail_instrs.append(instr)

            for op_id in chosen:
                scheduled.add(op_id)
                for succ, latency in succ_edges[op_id]:
                    if succ in tail_ops and latency:
                        wait_prev[succ] -= 1
                        earliest[succ] = max(earliest[succ], cycle + latency)
                        if (
                            wait_prev[succ] == 0
                            and wait_same[succ] == 0
                            and succ not in scheduled
                        ):
                            ready[ops[succ].engine].append(succ)

        candidate = instrs[:cutoff] + tail_instrs
        self.ir_tail_reschedule_summary = {
            "status": "accepted" if len(candidate) <= len(instrs) else "rejected_longer",
            "cutoff": cutoff,
            "original_tail_cycles": len(instrs) - cutoff,
            "new_tail_cycles": len(tail_instrs),
            "tail_ops": len(tail_ops),
            "order": order,
        }
        if len(candidate) <= len(instrs):
            return candidate
        return instrs

    def schedule_ir(
        self,
        ops: list[KernelOp],
        mode: str = "fifo",
        weights: dict[str, int] | None = None,
        preserve_memory_order: bool = True,
    ) -> list[dict[str, list[tuple]]]:
        """Schedule IR nodes into VLIW bundles using explicit dependencies."""
        weights = weights or {}
        self.ir_ops = ops
        succ_edges, dep_edges = self._build_ir_dependencies(
            ops, preserve_memory_order=preserve_memory_order
        )
        self.ir_dependency_edges = dep_edges
        self.ir_liveness_summary = self.analyze_ir_liveness(ops)
        if mode == "asap":
            return self._schedule_ir_asap(ops, mode, weights, succ_edges)
        critical_height = self._critical_heights(ops, succ_edges)
        tail_score = self._tail_scores(
            ops,
            tail_rounds=weights.get("tail_rounds", 3),
            final_store_bonus=weights.get("final_store_bonus", 80),
            final_hash_bonus=weights.get("final_hash_bonus", 30),
            droppable_penalty=weights.get("droppable_penalty", 40),
        )

        wait_prev = [0] * len(ops)
        wait_same = [0] * len(ops)
        for pred_edges in succ_edges:
            for succ, latency in pred_edges:
                if latency:
                    wait_prev[succ] += 1
                else:
                    wait_same[succ] += 1

        ready = {
            engine: [
                op_id
                for op_id, op in enumerate(ops)
                if op.engine == engine and wait_prev[op_id] == 0 and wait_same[op_id] == 0
            ]
            for engine in SLOT_LIMITS
        }
        scheduled = [False] * len(ops)
        op_cycle = [-1] * len(ops)
        remaining = len(ops)
        instrs: list[dict[str, list[tuple]]] = []
        base_engine_order = ("load", "valu", "alu", "store", "flow", "debug")
        engine_rank = {engine: i for i, engine in enumerate(base_engine_order)}
        engine_slots = defaultdict(int)
        engine_active = defaultdict(int)

        def priority(op_id: int) -> tuple[int, int]:
            if mode == "fifo":
                return (op_id, 0)
            if mode == "critical_path":
                return (-critical_height[op_id], op_id)
            if mode == "tail_weighted":
                score = critical_height[op_id] + tail_score[op_id]
                return (-score, op_id)
            if mode == "engine_balanced":
                op = ops[op_id]
                target_fill = engine_slots[op.engine] / max(1, SLOT_LIMITS[op.engine])
                score = critical_height[op_id] - int(target_fill)
                return (-score, op_id)
            raise ValueError(f"Unknown IR scheduler mode {mode!r}")

        def engine_order_for_cycle() -> list[str]:
            if mode != "engine_balanced":
                return list(base_engine_order)
            return sorted(
                base_engine_order,
                key=lambda engine: (
                    -len(ready[engine]) / max(1, SLOT_LIMITS[engine]),
                    engine_active[engine],
                    engine_rank[engine],
                ),
            )

        while remaining:
            instr: dict[str, list[tuple]] = {}
            chosen: list[int] = []
            chosen_set: set[int] = set()
            capacity = {engine: SLOT_LIMITS[engine] for engine in base_engine_order}

            progressed = True
            while progressed:
                progressed = False
                for engine in engine_order_for_cycle():
                    queue = ready[engine]
                    if mode != "fifo":
                        queue.sort(key=priority)
                    while queue and capacity[engine] > 0:
                        op_id = queue.pop(0)
                        if scheduled[op_id] or op_id in chosen_set:
                            continue
                        instr.setdefault(engine, []).append(ops[op_id].slot)
                        chosen.append(op_id)
                        chosen_set.add(op_id)
                        capacity[engine] -= 1
                        progressed = True

                        for succ, latency in succ_edges[op_id]:
                            if latency == 0:
                                wait_same[succ] -= 1
                                if (
                                    wait_prev[succ] == 0
                                    and wait_same[succ] == 0
                                    and not scheduled[succ]
                                    and succ not in chosen_set
                                ):
                                    ready[ops[succ].engine].append(succ)

            assert chosen, "IR scheduler stalled"
            instrs.append(instr)
            remaining -= len(chosen)
            for engine, slots in instr.items():
                engine_slots[engine] += len(slots)
                engine_active[engine] += 1

            for op_id in chosen:
                scheduled[op_id] = True
                op_cycle[op_id] = len(instrs) - 1
                for succ, latency in succ_edges[op_id]:
                    if latency:
                        wait_prev[succ] -= 1
                        if (
                            wait_prev[succ] == 0
                            and wait_same[succ] == 0
                            and not scheduled[succ]
                        ):
                            ready[ops[succ].engine].append(succ)

        position_summary = self._schedule_position_summary(ops, op_cycle)
        self.ir_schedule_summary = {
            "mode": mode,
            "cycles": len(instrs),
            "ops": len(ops),
            "dependency_edges": len(dep_edges),
            "edge_kinds": {
                kind: sum(1 for edge in dep_edges if edge.kind == kind)
                for kind in ("RAW", "WAW", "WAR", "MEM")
            },
            "engine_slots": dict(engine_slots),
            "engine_active": dict(engine_active),
            "scratch_liveness": self.ir_liveness_summary,
            "region_local_allocation": self.region_local_allocation_summary,
            **position_summary,
        }
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(
        self,
        name=None,
        length=1,
        *,
        role: str = "scratch",
        pool: str | None = None,
        persistent: bool = False,
    ):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        allocation_name = name if name is not None else f"anon_{addr}"
        self.scratch_allocations[addr] = ScratchAllocation(
            base=addr,
            length=length,
            name=allocation_name,
            role=role,
            pool=pool,
            persistent=persistent,
        )
        for offset in range(length):
            self.scratch_addr_to_alloc_base[addr + offset] = addr
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def alloc_role(
        self,
        name: str,
        role: str,
        length: int = 1,
        *,
        pool: str | None = None,
        persistent: bool = False,
    ):
        return self.alloc_scratch(
            name,
            length,
            role=role,
            pool=pool,
            persistent=persistent,
        )

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(
                name,
                role="const",
                pool="const",
                persistent=True,
            )
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def alloc_vec(
        self,
        name: str,
        role: str = "scratch",
        *,
        pool: str | None = None,
        persistent: bool = False,
    ):
        return self.alloc_role(name, role, VLEN, pool=pool, persistent=persistent)

    def _remap_slot_addresses(self, engine: str, slot: tuple, addr_map: dict[int, int]) -> tuple:
        def remap(addr: int) -> int:
            return addr_map.get(addr, addr)

        op = slot[0]
        if engine == "alu":
            _op, dest, a, b = slot
            return (_op, remap(dest), remap(a), remap(b))
        if engine == "valu":
            if op == "vbroadcast":
                _op, dest, src = slot
                return (_op, remap(dest), remap(src))
            if op == "multiply_add":
                _op, dest, a, b, c = slot
                return (_op, remap(dest), remap(a), remap(b), remap(c))
            _op, dest, a, b = slot
            return (_op, remap(dest), remap(a), remap(b))
        if engine == "load":
            if op == "const":
                _op, dest, val = slot
                return (_op, remap(dest), val)
            if op == "load":
                _op, dest, addr = slot
                return (_op, remap(dest), remap(addr))
            if op == "load_offset":
                _op, dest, addr, offset = slot
                return (_op, remap(dest), remap(addr), offset)
            if op == "vload":
                _op, dest, addr = slot
                return (_op, remap(dest), remap(addr))
        if engine == "store":
            if op == "store":
                _op, addr, src = slot
                return (_op, remap(addr), remap(src))
            if op == "vstore":
                _op, addr, src = slot
                return (_op, remap(addr), remap(src))
        if engine == "flow":
            if op == "select":
                _op, dest, cond, a, b = slot
                return (_op, remap(dest), remap(cond), remap(a), remap(b))
            if op == "add_imm":
                _op, dest, a, imm = slot
                return (_op, remap(dest), remap(a), imm)
            if op == "vselect":
                _op, dest, cond, a, b = slot
                return (_op, remap(dest), remap(cond), remap(a), remap(b))
        if engine == "debug":
            if op == "compare":
                _op, loc, key = slot
                return (_op, remap(loc), key)
            if op == "vcompare":
                _op, loc, keys = slot
                return (_op, remap(loc), keys)
        return slot

    def _remap_kernel_op(self, op: KernelOp, addr_map: dict[int, int]) -> KernelOp:
        return self.make_ir_op(
            op.engine,
            self._remap_slot_addresses(op.engine, op.slot, addr_map),
            tag=op.tag,
            round=op.round,
            level=op.level,
            group=op.group,
            lane=op.lane,
            contributes_value=op.contributes_value,
            contributes_index=op.contributes_index,
            droppable_final=op.droppable_final,
            region=op.region,
        )

    def _allocation_intervals(
        self,
        ops: list[KernelOp],
        eligible_bases: set[int],
    ) -> dict[int, tuple[int, int]]:
        intervals: dict[int, list[int]] = {}
        for op_id, op in enumerate(ops):
            for addr in op.reads | op.writes:
                base = self.scratch_addr_to_alloc_base.get(addr)
                if base not in eligible_bases:
                    continue
                interval = intervals.setdefault(base, [op_id, op_id])
                interval[0] = min(interval[0], op_id)
                interval[1] = max(interval[1], op_id)
        return {base: (start, end) for base, (start, end) in intervals.items()}

    def _setup_touched_addresses(self) -> set[int]:
        touched: set[int] = set()
        for instr in self.instrs:
            for engine, slots in instr.items():
                for slot in slots:
                    reads, writes = self._slot_rw(engine, slot)
                    touched.update(reads)
                    touched.update(writes)
        return touched

    def apply_region_local_scratch_allocation(
        self,
        ops: list[KernelOp],
        *,
        pools: set[str],
        roles: set[str] | None = None,
    ) -> list[KernelOp]:
        """Color non-overlapping vector scratch allocations onto shared storage."""
        eligible = [
            allocation
            for allocation in self.scratch_allocations.values()
            if (
                allocation.length == VLEN
                and not allocation.persistent
                and allocation.pool in pools
                and (roles is None or allocation.role in roles)
            )
        ]
        intervals = self._allocation_intervals(ops, {allocation.base for allocation in eligible})
        eligible = [allocation for allocation in eligible if allocation.base in intervals]
        colors: list[dict[str, Any]] = []
        allocation_to_color: dict[int, int] = {}

        for allocation in sorted(eligible, key=lambda item: (intervals[item.base][0], item.base)):
            start, end = intervals[allocation.base]
            chosen = None
            for color in colors:
                if color["end"] < start:
                    chosen = color
                    break
            if chosen is None:
                chosen = {"base": allocation.base, "end": end, "members": []}
                colors.append(chosen)
            else:
                chosen["end"] = end
            chosen["members"].append(
                {
                    "base": allocation.base,
                    "name": allocation.name,
                    "start": start,
                    "end": end,
                }
            )
            allocation_to_color[allocation.base] = chosen["base"]

        addr_map: dict[int, int] = {}
        remapped_allocations = 0
        for allocation in eligible:
            color_base = allocation_to_color[allocation.base]
            if color_base == allocation.base:
                continue
            remapped_allocations += 1
            for offset in range(allocation.length):
                addr_map[allocation.base + offset] = color_base + offset

        if not addr_map:
            self.region_local_allocation_summary = {
                "enabled": True,
                "pools": sorted(pools),
                "roles": None if roles is None else sorted(roles),
                "eligible_allocations": len(eligible),
                "colors": len(colors),
                "remapped_allocations": 0,
                "effective_scratch_ptr": self.scratch_ptr,
                "colors_detail": colors,
            }
            return ops

        remapped_ops = [self._remap_kernel_op(op, addr_map) for op in ops]
        used_addresses = self._setup_touched_addresses()
        for op in remapped_ops:
            used_addresses.update(op.reads)
            used_addresses.update(op.writes)
        effective_scratch_ptr = max(used_addresses, default=-1) + 1
        old_scratch_ptr = self.scratch_ptr
        self.scratch_ptr = effective_scratch_ptr
        self.region_local_allocation_summary = {
            "enabled": True,
            "pools": sorted(pools),
            "roles": None if roles is None else sorted(roles),
            "eligible_allocations": len(eligible),
            "colors": len(colors),
            "remapped_allocations": remapped_allocations,
            "old_scratch_ptr": old_scratch_ptr,
            "effective_scratch_ptr": effective_scratch_ptr,
            "saved_words": old_scratch_ptr - effective_scratch_ptr,
            "colors_detail": colors,
        }
        return remapped_ops

    def _schedule_ir_slots(
        self,
        slots: list[tuple[str, tuple]],
        *,
        tag: str,
        preserve_memory_order: bool = False,
        mode: str = "fifo",
        weights: dict[str, int] | None = None,
    ) -> list[dict[str, list[tuple]]]:
        ops = [self.make_ir_op(engine, slot, tag=tag) for engine, slot in slots]
        return self.schedule_ir(
            ops,
            mode=mode,
            weights=weights,
            preserve_memory_order=preserve_memory_order,
        )

    def build_simd_ir_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        variant: dict[str, Any],
    ):
        """SIMD active-context kernel emitted through the explicit IR."""
        assert batch_size % VLEN == 0
        assert forest_height == 10 and n_nodes == 2047

        group_size = variant.get("group_size", 16)
        round_tile = variant.get("round_tile", 11)
        simd_body_tile_boundaries = variant.get("simd_body_tile_boundaries", (0, 11, 15))
        scalar_xor_levels = set(
            variant.get("scalar_xor_levels", (0, 1, 2, 3, 4, 5, 6, 7, 8, 10))
        )
        scalar_xor_rounds = {int(round_i) for round_i in variant.get("scalar_xor_rounds", ())}
        vector_xor_rounds = {int(round_i) for round_i in variant.get("vector_xor_rounds", ())}
        scalar_hash_h1_stages = set(variant.get("scalar_hash_h1_stages", (3,)))
        scalar_hash_h2_stages = set(variant.get("scalar_hash_h2_stages", ()))
        scalar_hash_combine_stages = set(variant.get("scalar_hash_combine_stages", ()))
        scalar_hash_depths = set(variant.get("scalar_hash_depths", (0, 3)))
        final_scalar_hash_h1_stages = set(
            variant.get("final_scalar_hash_h1_stages", scalar_hash_h1_stages)
        )
        final_scalar_hash_h2_stages = set(
            variant.get("final_scalar_hash_h2_stages", scalar_hash_h2_stages)
        )
        final_scalar_hash_combine_stages = set(
            variant.get("final_scalar_hash_combine_stages", scalar_hash_combine_stages)
        )
        predrain_scalar_hash_h1_stages = set(
            variant.get("predrain_scalar_hash_h1_stages", (3,))
        )
        predrain_scalar_hash_h2_stages = set(
            variant.get("predrain_scalar_hash_h2_stages", ())
        )
        predrain_scalar_hash_combine_stages = set(
            variant.get("predrain_scalar_hash_combine_stages", (1, 5))
        )
        default_hash_scalar_by_round = {
            5: {"h1": (3,), "h2": (3,), "combine": ()},
        }
        simd_hash_scalar_by_round = {
            int(round_i): {
                "h1": set(policy.get("h1", ())),
                "h2": set(policy.get("h2", ())),
                "combine": set(policy.get("combine", ())),
            }
            for round_i, policy in variant.get(
                "simd_hash_scalar_by_round", default_hash_scalar_by_round
            ).items()
        }
        scalar_index_levels = set(variant.get("scalar_index_levels", ()))
        scalar_index_rounds = {
            int(round_i) for round_i in variant.get("scalar_index_rounds", ())
        }
        final_tile_rotation = variant.get("final_tile_rotation", 2)
        split_simd_final_drain = variant.get("split_simd_final_drain", True)
        simd_final_drain_rounds = variant.get("simd_final_drain_rounds", 1)
        simd_final_group_order = variant.get("simd_final_group_order", "forward")
        simd_final_group_sequence = variant.get("simd_final_group_sequence")
        simd_final_group_rotation = variant.get(
            "simd_final_group_rotation", final_tile_rotation
        )
        simd_predrain_split_rounds = {
            int(round_i) for round_i in variant.get("simd_predrain_split_rounds", ())
        }
        simd_predrain_order_rounds = {
            int(round_i) for round_i in variant.get("simd_predrain_order_rounds", (11,))
        }
        simd_predrain_group_order = variant.get("simd_predrain_group_order", "rotate")
        simd_predrain_group_sequence = variant.get("simd_predrain_group_sequence")
        simd_predrain_group_rotation = variant.get("simd_predrain_group_rotation", 1)
        simd_predrain_hash_rounds = {
            int(round_i) for round_i in variant.get("simd_predrain_hash_rounds", (13,))
        }
        simd_selector_ring_depth = variant.get("simd_selector_ring_depth", 3)
        simd_alias_selector1_tmp3 = variant.get("simd_alias_selector1_tmp3", True)
        simd_alias_selector2_tmp2 = variant.get("simd_alias_selector2_tmp2", True)
        simd_hash_h2_tmp_node = variant.get(
            "simd_hash_h2_tmp_node", simd_selector_ring_depth >= 3
        )
        simd_drop_final_index = variant.get("simd_drop_final_index", True)
        simd_final_wavefront = variant.get("simd_final_wavefront", False)
        simd_interleave_final_stores = variant.get("simd_interleave_final_stores", False)
        prune_dead_tail = variant.get("prune_dead_tail", True)
        ir_scheduler = variant.get("ir_scheduler", "asap")
        ir_scheduler_weights = variant.get("ir_scheduler_weights", {})
        region_local_compact_scratch = variant.get("region_local_compact_scratch", False)
        region_local_compact_pools = {
            str(pool) for pool in variant.get("region_local_compact_pools", ("context",))
        }
        region_local_compact_roles = variant.get("region_local_compact_roles")
        if region_local_compact_roles is not None:
            region_local_compact_roles = {str(role) for role in region_local_compact_roles}

        tmp_init = self.alloc_role("tmp_init", "addr_tmp", pool="setup")
        tmp_init2 = self.alloc_role("tmp_init2", "addr_tmp", pool="setup")
        tmp_addr = self.alloc_role("tmp_addr", "addr_tmp", pool="io")
        tmp_addr2 = self.alloc_role("tmp_addr2", "addr_tmp", pool="io")

        forest_values_p = 7
        inp_indices_p = forest_values_p + n_nodes
        inp_values_p = inp_indices_p + batch_size

        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for name in init_vars:
            self.alloc_role(name, "header", persistent=True)

        init_slots: list[tuple[str, tuple]] = [
            ("load", ("const", self.scratch["forest_values_p"], forest_values_p)),
            ("load", ("const", self.scratch["inp_indices_p"], inp_indices_p)),
            ("load", ("const", self.scratch["inp_values_p"], inp_values_p)),
        ]
        scalar_consts: dict[int, int] = {}
        vconsts: dict[int, int] = {}

        def scalar_const(value: int, name: str | None = None) -> int:
            if value not in scalar_consts:
                addr = self.alloc_scratch(
                    name,
                    role="const",
                    pool="const",
                    persistent=True,
                )
                scalar_consts[value] = addr
                init_slots.append(("load", ("const", addr, value)))
            return scalar_consts[value]

        def vector_const(value: int, name: str | None = None) -> int:
            if value not in vconsts:
                scalar = scalar_const(value)
                addr = self.alloc_vec(
                    name or f"vconst_{value}",
                    "const",
                    pool="vconst",
                    persistent=True,
                )
                vconsts[value] = addr
                init_slots.append(("valu", ("vbroadcast", addr, scalar)))
            return vconsts[value]

        zero_vec = vector_const(0, "v_zero")
        one_vec = vector_const(1, "v_one")
        two_vec = vector_const(2, "v_two")
        one_const = scalar_const(1, "one")
        level_base_vecs = {
            level: vector_const(forest_values_p + (1 << level) - 1, f"v_level_base_{level}")
            for level in range(4, forest_height + 1)
        }
        four_vec = vector_const(4, "v_four")

        node_vecs = []
        for node_idx in range(15):
            node_scalar = self.alloc_role(f"node_{node_idx}", "node_cache", persistent=True)
            node_vec = self.alloc_vec(
                f"v_node_{node_idx}",
                "node_cache",
                pool="top_nodes",
                persistent=True,
            )
            node_offset = scalar_const(node_idx)
            addr_reg = tmp_init if node_idx % 2 == 0 else tmp_init2
            init_slots.append(("alu", ("+", addr_reg, self.scratch["forest_values_p"], node_offset)))
            init_slots.append(("load", ("load", node_scalar, addr_reg)))
            init_slots.append(("valu", ("vbroadcast", node_vec, node_scalar)))
            node_vecs.append(node_vec)

        hash_vec_consts1 = []
        hash_vec_consts3 = []
        hash_mul_vecs = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            hash_vec_consts1.append(vector_const(val1))
            hash_vec_consts3.append(vector_const(val3))
            if op1 == "+" and op2 == "+" and op3 == "<<":
                hash_mul_vecs.append(vector_const(1 + (1 << val3)))
            else:
                hash_mul_vecs.append(None)

        idx_base = self.alloc_role("path", "path", batch_size, pool="batch_state", persistent=True)
        val_base = self.alloc_role("vals", "value", batch_size, pool="batch_state", persistent=True)

        offset = self.alloc_role("offset", "addr_tmp", pool="io")
        init_slots.append(("load", ("const", offset, 0)))
        vlen_const = scalar_const(VLEN, "vlen")
        blocks_per_round = batch_size // VLEN
        final_store_offsets = (
            [
                scalar_const(block * VLEN, f"final_store_offset_{block}")
                for block in range(blocks_per_round)
            ]
            if simd_interleave_final_stores
            else []
        )

        self.instrs.extend(
            self._schedule_ir_slots(
                init_slots,
                tag="setup",
                preserve_memory_order=False,
                mode=ir_scheduler,
                weights=ir_scheduler_weights,
            )
        )

        load_slots: list[tuple[str, tuple]] = []
        for block in range(blocks_per_round):
            load_slots.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], offset)))
            load_slots.append(("load", ("vload", idx_base + block * VLEN, tmp_addr)))
            load_slots.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], offset)))
            load_slots.append(("load", ("vload", val_base + block * VLEN, tmp_addr)))
            load_slots.append(("alu", ("+", offset, offset, vlen_const)))

        contexts = []
        for gi in range(group_size):
            node_tmp = self.alloc_vec(f"node_tmp_{gi}", "node_tmp", pool="context")
            hash_tmp = self.alloc_vec(f"hash_tmp_{gi}", "hash_tmp", pool="context")
            select_tmp0 = self.alloc_vec(f"select_tmp0_{gi}", "select_tmp", pool="context")
            select_tmp1 = self.alloc_vec(f"select_tmp1_{gi}", "select_tmp", pool="context")
            selector0 = (
                self.alloc_vec(
                    f"selector_latest_{gi}",
                    "select_tmp",
                    pool="selector_ring",
                )
                if simd_selector_ring_depth >= 1
                else None
            )
            selector1 = (
                (
                    select_tmp1
                    if simd_alias_selector1_tmp3
                    else self.alloc_vec(
                        f"selector_prev_{gi}",
                        "select_tmp",
                        pool="selector_ring",
                    )
                )
                if simd_selector_ring_depth >= 2
                else None
            )
            selector2 = (
                (
                    select_tmp0
                    if simd_alias_selector2_tmp2
                    else self.alloc_vec(
                        f"selector_older_{gi}",
                        "select_tmp",
                        pool="selector_ring",
                    )
                )
                if simd_selector_ring_depth >= 3
                else None
            )
            contexts.append(
                {
                    "node": node_tmp,
                    "tmp1": hash_tmp,
                    "tmp2": select_tmp0,
                    "tmp3": select_tmp1,
                    "selector0": selector0,
                    "selector1": selector1,
                    "selector2": selector2,
                    "store_addr": (
                        self.alloc_role(
                            f"store_addr_{gi}",
                            "addr_tmp",
                            pool="context_store",
                        )
                        if simd_interleave_final_stores
                        else tmp_addr
                    ),
                }
            )

        body_ops: list[KernelOp] = [
            self.make_ir_op(engine, slot, tag="load")
            for engine, slot in load_slots
        ]
        interleaved_store_blocks: set[int] = set()

        def emit_simd_op(
            engine: str,
            slot: tuple,
            *,
            tag: str,
            round_i: int,
            group: int,
            contributes_value: bool = False,
            contributes_index: bool = False,
            droppable_final: bool = False,
        ):
            final_round = round_i == rounds - 1
            if final_round:
                region = "final_drain"
            elif round_i in simd_predrain_split_rounds:
                region = "pre_drain"
            else:
                region = "final_drain" if round_i >= max(0, rounds - 3) else "body"
            if final_round and tag == "store":
                region = "store_tail"
            body_ops.append(
                self.make_ir_op(
                    engine,
                    slot,
                    tag=tag,
                    round=round_i,
                    level=round_i % (forest_height + 1),
                    group=group,
                    contributes_value=contributes_value,
                    contributes_index=contributes_index,
                    droppable_final=droppable_final,
                    region=region,
                )
            )

        def ordered_group_indices(
            order: str,
            sequence: Any,
            rotation: int,
            *,
            label: str,
        ):
            if sequence is not None:
                gi_order = [int(gi) for gi in sequence]
                if sorted(gi_order) != list(range(group_size)):
                    raise ValueError(f"{label} must be a permutation of context indices")
            elif order == "reverse":
                gi_order = range(group_size - 1, -1, -1)
            elif order == "even_odd":
                gi_order = list(range(0, group_size, 2)) + list(range(1, group_size, 2))
            elif order == "odd_even":
                gi_order = list(range(1, group_size, 2)) + list(range(0, group_size, 2))
            elif order == "odd_even_swap_3_4":
                gi_order = list(range(1, group_size, 2)) + list(range(0, group_size, 2))
                gi_order[3], gi_order[4] = gi_order[4], gi_order[3]
            elif order == "rotate":
                rot = rotation % group_size
                gi_order = list(range(rot, group_size)) + list(range(rot))
            elif order == "forward":
                gi_order = range(group_size)
            else:
                raise ValueError(f"Unknown SIMD group order {order!r}")
            return gi_order

        def group_indices(is_final_drain: bool, round_i: int | None = None):
            if is_final_drain:
                return ordered_group_indices(
                    simd_final_group_order,
                    simd_final_group_sequence,
                    simd_final_group_rotation,
                    label="simd_final_group_sequence",
                )
            if round_i in simd_predrain_order_rounds:
                return ordered_group_indices(
                    simd_predrain_group_order,
                    simd_predrain_group_sequence,
                    simd_predrain_group_rotation,
                    label="simd_predrain_group_sequence",
                )
            return range(group_size)

        def hash_scalar_policy(round_i: int, level: int):
            if round_i == rounds - 1:
                return (
                    final_scalar_hash_h1_stages,
                    final_scalar_hash_h2_stages,
                    final_scalar_hash_combine_stages,
                    True,
                )
            if round_i in simd_hash_scalar_by_round:
                policy = simd_hash_scalar_by_round[round_i]
                return (
                    policy["h1"],
                    policy["h2"],
                    policy["combine"],
                    True,
                )
            if round_i in simd_predrain_hash_rounds:
                return (
                    predrain_scalar_hash_h1_stages,
                    predrain_scalar_hash_h2_stages,
                    predrain_scalar_hash_combine_stages,
                    True,
                )
            return (
                scalar_hash_h1_stages,
                scalar_hash_h2_stages,
                scalar_hash_combine_stages,
                level in scalar_hash_depths,
            )

        def tile_plan():
            if not split_simd_final_drain:
                for group_start in range(0, blocks_per_round, group_size):
                    for round_start in range(0, rounds, round_tile):
                        round_end = min(rounds, round_start + round_tile)
                        is_final = round_start >= max(0, rounds - simd_final_drain_rounds)
                        yield group_start, round_start, round_end, is_final
                return

            final_start = max(0, rounds - simd_final_drain_rounds)
            split_rounds = sorted(
                round_i for round_i in simd_predrain_split_rounds if 0 <= round_i < final_start
            )

            def body_tile_ranges():
                if simd_body_tile_boundaries is not None:
                    boundaries = sorted(
                        {
                            int(boundary)
                            for boundary in simd_body_tile_boundaries
                            if 0 <= int(boundary) <= final_start
                        }
                    )
                    if not boundaries or boundaries[0] != 0:
                        boundaries.insert(0, 0)
                    if boundaries[-1] != final_start:
                        boundaries.append(final_start)
                    ranges = [
                        (start, end)
                        for start, end in zip(boundaries, boundaries[1:])
                        if start < end
                    ]
                    if sum(end - start for start, end in ranges) != final_start:
                        raise ValueError("simd_body_tile_boundaries must cover body rounds")
                    return ranges

                ranges = []
                current_round = 0
                for split_round in split_rounds:
                    while current_round < split_round:
                        round_end = min(split_round, current_round + round_tile)
                        ranges.append((current_round, round_end))
                        current_round = round_end
                    ranges.append((split_round, split_round + 1))
                    current_round = split_round + 1
                while current_round < final_start:
                    round_end = min(final_start, current_round + round_tile)
                    ranges.append((current_round, round_end))
                    current_round = round_end
                return ranges

            body_ranges = body_tile_ranges()
            for group_start in range(0, blocks_per_round, group_size):
                for round_start, round_end in body_ranges:
                    yield group_start, round_start, round_end, False
            for group_start in range(0, blocks_per_round, group_size):
                yield group_start, final_start, rounds, True

        for group_start, round_start, round_end, is_final_drain in tile_plan():
                if (
                    simd_final_wavefront
                    and is_final_drain
                    and round_end - round_start == 1
                ):
                    round_i = round_start
                    level = round_i % (forest_height + 1)
                    if level >= 4:
                        final_groups = []
                        for gi in group_indices(True, round_i):
                            block = group_start + gi
                            if block >= blocks_per_round:
                                break
                            final_groups.append((block, contexts[gi]))

                        addr_base_vec = level_base_vecs[level]
                        for block, ctx in final_groups:
                            idx_vec = idx_base + block * VLEN
                            for lane in range(VLEN):
                                emit_simd_op(
                                    "alu",
                                    ("+", ctx["tmp1"] + lane, addr_base_vec + lane, idx_vec + lane),
                                    tag="gather",
                                    round_i=round_i,
                                    group=block,
                                )
                        for _block, ctx in final_groups:
                            for lane in range(VLEN):
                                emit_simd_op(
                                    "load",
                                    ("load", ctx["node"] + lane, ctx["tmp1"] + lane),
                                    tag="gather",
                                    round_i=round_i,
                                    group=_block,
                                )
                        for block, ctx in final_groups:
                            val_vec = val_base + block * VLEN
                            if level in scalar_xor_levels:
                                for lane in range(VLEN):
                                    emit_simd_op(
                                        "alu",
                                        ("^", val_vec + lane, val_vec + lane, ctx["node"] + lane),
                                        tag="hash",
                                        round_i=round_i,
                                        group=block,
                                        contributes_value=True,
                                    )
                            else:
                                emit_simd_op(
                                    "valu",
                                    ("^", val_vec, val_vec, ctx["node"]),
                                    tag="hash",
                                    round_i=round_i,
                                    group=block,
                                    contributes_value=True,
                                )
                        for hi, (op1, _val1, op2, op3, _val3) in enumerate(HASH_STAGES):
                            mul_vec = hash_mul_vecs[hi]
                            for block, ctx in final_groups:
                                val_vec = val_base + block * VLEN
                                if mul_vec is not None:
                                    emit_simd_op(
                                        "valu",
                                        ("multiply_add", val_vec, val_vec, mul_vec, hash_vec_consts1[hi]),
                                        tag="hash",
                                        round_i=round_i,
                                        group=block,
                                        contributes_value=True,
                                    )
                                else:
                                    (
                                        h1_stages,
                                        h2_stages,
                                        combine_stages,
                                        hash_depth_enabled,
                                    ) = hash_scalar_policy(round_i, level)
                                    h1_scalar = hi in h1_stages and hash_depth_enabled
                                    h2_scalar = hi in h2_stages and hash_depth_enabled
                                    combine_scalar = hi in combine_stages and hash_depth_enabled
                                    h2_tmp = ctx["node"] if simd_hash_h2_tmp_node else ctx["tmp2"]
                                    if h1_scalar:
                                        c1 = scalar_const(HASH_STAGES[hi][1])
                                        for lane in range(VLEN):
                                            emit_simd_op(
                                                "alu",
                                                (op1, ctx["tmp1"] + lane, val_vec + lane, c1),
                                                tag="hash",
                                                round_i=round_i,
                                                group=block,
                                                contributes_value=True,
                                            )
                                    else:
                                        emit_simd_op(
                                            "valu",
                                            (op1, ctx["tmp1"], val_vec, hash_vec_consts1[hi]),
                                            tag="hash",
                                            round_i=round_i,
                                            group=block,
                                            contributes_value=True,
                                        )
                                    if h2_scalar:
                                        c3 = scalar_const(HASH_STAGES[hi][4])
                                        for lane in range(VLEN):
                                            emit_simd_op(
                                                "alu",
                                                (op3, h2_tmp + lane, val_vec + lane, c3),
                                                tag="hash",
                                                round_i=round_i,
                                                group=block,
                                                contributes_value=True,
                                            )
                                    else:
                                        emit_simd_op(
                                            "valu",
                                            (op3, h2_tmp, val_vec, hash_vec_consts3[hi]),
                                            tag="hash",
                                            round_i=round_i,
                                            group=block,
                                            contributes_value=True,
                                        )
                                    if combine_scalar:
                                        for lane in range(VLEN):
                                            emit_simd_op(
                                                "alu",
                                                (op2, val_vec + lane, ctx["tmp1"] + lane, h2_tmp + lane),
                                                tag="hash",
                                                round_i=round_i,
                                                group=block,
                                                contributes_value=True,
                                            )
                                    else:
                                        emit_simd_op(
                                            "valu",
                                            (op2, val_vec, ctx["tmp1"], h2_tmp),
                                            tag="hash",
                                            round_i=round_i,
                                            group=block,
                                            contributes_value=True,
                                        )
                        if not simd_drop_final_index:
                            for block, ctx in final_groups:
                                idx_vec = idx_base + block * VLEN
                                val_vec = val_base + block * VLEN
                                for lane in range(VLEN):
                                    emit_simd_op(
                                        "alu",
                                        ("&", ctx["tmp1"] + lane, val_vec + lane, one_const),
                                        tag="index",
                                        round_i=round_i,
                                        group=block,
                                        contributes_index=True,
                                        droppable_final=True,
                                    )
                                emit_simd_op(
                                    "valu",
                                    ("multiply_add", idx_vec, idx_vec, two_vec, ctx["tmp1"]),
                                    tag="index",
                                    round_i=round_i,
                                    group=block,
                                    contributes_index=True,
                                    droppable_final=True,
                                )
                        continue
                gi_order = group_indices(is_final_drain, round_start)
                for gi in gi_order:
                    block = group_start + gi
                    if block >= blocks_per_round:
                        break
                    ctx = contexts[gi]
                    idx_vec = idx_base + block * VLEN
                    val_vec = val_base + block * VLEN
                    base = block * VLEN

                    for round_i in range(round_start, round_end):
                        level = round_i % (forest_height + 1)

                        def add(engine: str, slot: tuple, tag: str = "body", value=False, index=False):
                            emit_simd_op(
                                engine,
                                slot,
                                tag=tag,
                                round_i=round_i,
                                group=block,
                                contributes_value=value or tag == "hash",
                                contributes_index=index or tag == "index",
                                droppable_final=round_i == rounds - 1 and tag == "index",
                            )

                        def emit_xor(node_vec: int) -> None:
                            if (
                                round_i in scalar_xor_rounds
                                or (round_i not in vector_xor_rounds and level in scalar_xor_levels)
                            ):
                                for lane in range(VLEN):
                                    add("alu", ("^", val_vec + lane, val_vec + lane, node_vec + lane), "hash")
                            else:
                                add("valu", ("^", val_vec, val_vec, node_vec), "hash")

                        def selector_bit(age: int) -> int | None:
                            if simd_selector_ring_depth < age:
                                return None
                            selector_idx = (round_i - age) % simd_selector_ring_depth
                            return ctx[f"selector{selector_idx}"]

                        def reusable_selector_tmp(*, avoid: set[int]) -> int:
                            """Pick a selector-backed temp whose bit is dead in this tree."""
                            for candidate in (ctx["selector2"], ctx["selector1"], ctx["selector0"]):
                                if candidate is not None and candidate not in avoid:
                                    return candidate
                            return ctx["tmp2"]

                        if level == 0:
                            emit_xor(node_vecs[0])
                        elif level == 1:
                            low_bit = selector_bit(1)
                            if low_bit is not None:
                                add("flow", ("vselect", ctx["node"], low_bit, node_vecs[2], node_vecs[1]), "select")
                            else:
                                add("valu", ("&", ctx["tmp1"], idx_vec, one_vec), "select")
                                add("flow", ("vselect", ctx["node"], ctx["tmp1"], node_vecs[2], node_vecs[1]), "select")
                            emit_xor(ctx["node"])
                        elif level == 2:
                            low_bit = selector_bit(1)
                            high_bit = selector_bit(2)
                            if low_bit is None:
                                low_bit = ctx["tmp2"]
                                add("valu", ("&", ctx["tmp2"], idx_vec, one_vec), "select")
                            if high_bit is None:
                                high_bit = ctx["node"]
                                add("valu", ("&", ctx["node"], idx_vec, two_vec), "select")
                            add("flow", ("vselect", ctx["tmp1"], low_bit, node_vecs[4], node_vecs[3]), "select")
                            if high_bit == ctx["tmp2"]:
                                p1_tmp = reusable_selector_tmp(avoid={low_bit, high_bit})
                            else:
                                p1_tmp = ctx["tmp2"]
                            add("flow", ("vselect", p1_tmp, low_bit, node_vecs[6], node_vecs[5]), "select")
                            add("flow", ("vselect", ctx["node"], high_bit, p1_tmp, ctx["tmp1"]), "select")
                            emit_xor(ctx["node"])
                        elif level == 3:
                            low_bit = selector_bit(1)
                            mid_bit = selector_bit(2)
                            high_bit = selector_bit(3)
                            if low_bit is None:
                                low_bit = ctx["tmp2"]
                                add("valu", ("&", ctx["tmp2"], idx_vec, one_vec), "select")
                            if mid_bit is None:
                                mid_bit = ctx["tmp3"]
                                add("valu", ("&", ctx["tmp3"], idx_vec, two_vec), "select")
                            add("flow", ("vselect", ctx["node"], low_bit, node_vecs[8], node_vecs[7]), "select")
                            add("flow", ("vselect", ctx["tmp1"], low_bit, node_vecs[10], node_vecs[9]), "select")
                            add("flow", ("vselect", ctx["tmp1"], mid_bit, ctx["tmp1"], ctx["node"]), "select")
                            add("flow", ("vselect", ctx["node"], low_bit, node_vecs[12], node_vecs[11]), "select")
                            if high_bit == ctx["tmp2"]:
                                # The low selector has already fed all four pair selects.
                                p3_tmp = low_bit
                            else:
                                p3_tmp = ctx["tmp2"]
                            add("flow", ("vselect", p3_tmp, low_bit, node_vecs[14], node_vecs[13]), "select")
                            add("flow", ("vselect", ctx["node"], mid_bit, p3_tmp, ctx["node"]), "select")
                            if high_bit is None:
                                high_bit = ctx["tmp2"]
                                add("valu", ("&", ctx["tmp2"], idx_vec, four_vec), "select")
                            add("flow", ("vselect", ctx["node"], high_bit, ctx["node"], ctx["tmp1"]), "select")
                            emit_xor(ctx["node"])
                        else:
                            addr_base_vec = level_base_vecs[level]
                            for lane in range(VLEN):
                                add("alu", ("+", ctx["tmp1"] + lane, addr_base_vec + lane, idx_vec + lane), "gather")
                            for lane in range(VLEN):
                                add("load", ("load", ctx["node"] + lane, ctx["tmp1"] + lane), "gather")
                            emit_xor(ctx["node"])

                        for hi, (op1, _val1, op2, op3, _val3) in enumerate(HASH_STAGES):
                            mul_vec = hash_mul_vecs[hi]
                            if mul_vec is not None:
                                add("valu", ("multiply_add", val_vec, val_vec, mul_vec, hash_vec_consts1[hi]), "hash")
                            else:
                                (
                                    h1_stages,
                                    h2_stages,
                                    combine_stages,
                                    hash_depth_enabled,
                                ) = hash_scalar_policy(round_i, level)
                                h1_scalar = hi in h1_stages and hash_depth_enabled
                                h2_scalar = hi in h2_stages and hash_depth_enabled
                                combine_scalar = hi in combine_stages and hash_depth_enabled
                                h2_tmp = ctx["node"] if simd_hash_h2_tmp_node else ctx["tmp2"]
                                if h1_scalar:
                                    c1 = scalar_const(HASH_STAGES[hi][1])
                                    for lane in range(VLEN):
                                        add("alu", (op1, ctx["tmp1"] + lane, val_vec + lane, c1), "hash")
                                else:
                                    add("valu", (op1, ctx["tmp1"], val_vec, hash_vec_consts1[hi]), "hash")
                                if h2_scalar:
                                    c3 = scalar_const(HASH_STAGES[hi][4])
                                    for lane in range(VLEN):
                                        add("alu", (op3, h2_tmp + lane, val_vec + lane, c3), "hash")
                                else:
                                    add("valu", (op3, h2_tmp, val_vec, hash_vec_consts3[hi]), "hash")
                                if combine_scalar:
                                    for lane in range(VLEN):
                                        add("alu", (op2, val_vec + lane, ctx["tmp1"] + lane, h2_tmp + lane), "hash")
                                else:
                                    add("valu", (op2, val_vec, ctx["tmp1"], h2_tmp), "hash")

                        if simd_drop_final_index and round_i == rounds - 1:
                            pass
                        elif level == forest_height:
                            add("valu", ("+", idx_vec, zero_vec, zero_vec), "index")
                        else:
                            parity_vec = (
                                ctx[f"selector{round_i % simd_selector_ring_depth}"]
                                if simd_selector_ring_depth >= 1
                                else ctx["tmp1"]
                            )
                            for lane in range(VLEN):
                                add("alu", ("&", parity_vec + lane, val_vec + lane, one_const), "index")
                            if level in scalar_index_levels or round_i in scalar_index_rounds:
                                for lane in range(VLEN):
                                    add("alu", ("<<", idx_vec + lane, idx_vec + lane, one_const), "index")
                                    add("alu", ("+", idx_vec + lane, idx_vec + lane, parity_vec + lane), "index")
                            else:
                                add("valu", ("multiply_add", idx_vec, idx_vec, two_vec, parity_vec), "index")

                        if simd_interleave_final_stores and round_i == rounds - 1:
                            emit_simd_op(
                                "alu",
                                ("+", ctx["store_addr"], self.scratch["inp_values_p"], final_store_offsets[block]),
                                tag="store",
                                round_i=round_i,
                                group=block,
                                contributes_value=True,
                            )
                            emit_simd_op(
                                "store",
                                ("vstore", ctx["store_addr"], val_vec),
                                tag="store",
                                round_i=round_i,
                                group=block,
                                contributes_value=True,
                            )
                            interleaved_store_blocks.add(block)

        body_ops.append(self.make_ir_op("load", ("const", offset, 0), tag="store"))
        for block in range(blocks_per_round):
            if block in interleaved_store_blocks:
                continue
            emit_simd_op(
                "alu",
                ("+", tmp_addr, self.scratch["inp_values_p"], offset),
                tag="store",
                round_i=rounds - 1,
                group=block,
                contributes_value=True,
            )
            emit_simd_op(
                "store",
                ("vstore", tmp_addr, val_base + block * VLEN),
                tag="store",
                round_i=rounds - 1,
                group=block,
                contributes_value=True,
            )
            emit_simd_op(
                "alu",
                ("+", offset, offset, vlen_const),
                tag="store",
                round_i=rounds - 1,
                group=block,
                contributes_value=True,
            )

        if region_local_compact_scratch:
            body_ops = self.apply_region_local_scratch_allocation(
                body_ops,
                pools=region_local_compact_pools,
                roles=region_local_compact_roles,
            )
        else:
            self.region_local_allocation_summary = {"enabled": False}

        scheduled = self.schedule_ir(
            body_ops,
            mode=ir_scheduler,
            weights=ir_scheduler_weights,
            preserve_memory_order=False,
        )
        if prune_dead_tail and scheduled:
            last_instr = scheduled[-1]
            last_valu = last_instr.get("valu", [])
            if (
                len(last_instr) == 1
                and len(last_valu) == 1
                and last_valu[0][0] == "multiply_add"
                and idx_base <= last_valu[0][1] < idx_base + batch_size
                and last_valu[0][1] == last_valu[0][2]
                and last_valu[0][3] == two_vec
            ):
                scheduled.pop()
        self.instrs.extend(scheduled)
        return

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_hash_single_temp(self, val_hash_addr, tmp, round, i):
        """Hash stage codegen that uses one temporary scratch word.

        The second hash input must be computed from the old value before the
        first input updates `val_hash_addr` in place.
        """
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op3, tmp, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op1, val_hash_addr, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op2, val_hash_addr, val_hash_addr, tmp)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        variant: dict[str, Any] | None = None,
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        variant = variant or {}
        if variant.get("simd_ir", True):
            self.build_simd_ir_kernel(forest_height, n_nodes, batch_size, rounds, variant)
            return
        ir_scheduler = variant.get("ir_scheduler", "fifo")
        ir_scheduler_weights = variant.get("ir_scheduler_weights", {})
        preserve_memory_order = variant.get("preserve_memory_order", True)
        single_temp_hash = variant.get("single_temp_hash", False)
        split_final_drain = variant.get("split_final_drain", False)
        final_drain_rounds = variant.get("final_drain_rounds", 3)
        body_group_order = variant.get("body_group_order", "forward")
        final_group_order = variant.get("final_group_order", "forward")
        final_group_rotation = variant.get("final_group_rotation", 0)
        drop_final_index = variant.get("drop_final_index", False)
        prune_droppable_final = variant.get("prune_droppable_final", False)
        emit_debug_pauses = variant.get("emit_debug_pauses", True)
        tmp1 = self.alloc_role("tmp1", "hash_tmp", pool="scalar_context")
        tmp2 = self.alloc_role("tmp2", "hash_tmp", pool="scalar_context")
        tmp3 = self.alloc_role("tmp3", "select_tmp", pool="scalar_context")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_role(v, "header", persistent=True)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        if emit_debug_pauses:
            self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        # Scalar scratch registers
        tmp_idx = self.alloc_role("tmp_idx", "path", pool="scalar_context")
        tmp_val = self.alloc_role("tmp_val", "value", pool="scalar_context")
        tmp_node_val = self.alloc_role("tmp_node_val", "node_tmp", pool="scalar_context")
        tmp_addr = self.alloc_role("tmp_addr", "addr_tmp", pool="scalar_context")

        body: list[KernelOp] = []

        def emit(
            engine: str,
            slot: tuple,
            *,
            tag: str,
            round_i: int,
            group: int,
            contributes_value: bool = False,
            contributes_index: bool = False,
            droppable_final: bool = False,
        ):
            final_round = round_i == rounds - 1
            region = "body"
            if round_i >= max(0, rounds - 3):
                region = "final_drain"
            if final_round and tag == "store":
                region = "store_tail"
            body.append(
                self.make_ir_op(
                    engine,
                    slot,
                    tag=tag,
                    round=round_i,
                    level=round_i % (forest_height + 1),
                    group=group,
                    contributes_value=contributes_value,
                    contributes_index=contributes_index,
                    droppable_final=droppable_final,
                    region=region,
                )
            )

        def ordered_groups(order: str, rotation: int = 0):
            groups = list(range(batch_size))
            if order == "reverse":
                groups.reverse()
            elif order == "even_odd":
                groups = list(range(0, batch_size, 2)) + list(range(1, batch_size, 2))
            elif order == "odd_even":
                groups = list(range(1, batch_size, 2)) + list(range(0, batch_size, 2))
            elif order == "rotate":
                rot = rotation % batch_size
                groups = groups[rot:] + groups[:rot]
            elif order != "forward":
                raise ValueError(f"Unknown group order {order!r}")
            return groups

        def round_order():
            if not split_final_drain:
                return [(round_i, False) for round_i in range(rounds)]
            final_start = max(0, rounds - final_drain_rounds)
            return (
                [(round_i, False) for round_i in range(final_start)]
                + [(round_i, True) for round_i in range(final_start, rounds)]
            )

        for round, is_final_drain_region in round_order():
            group_order = (
                ordered_groups(final_group_order, final_group_rotation)
                if is_final_drain_region
                else ordered_groups(body_group_order)
            )
            for i in group_order:
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                emit("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const), tag="load", round_i=round, group=i)
                emit("load", ("load", tmp_idx, tmp_addr), tag="load", round_i=round, group=i)
                emit("debug", ("compare", tmp_idx, (round, i, "idx")), tag="load", round_i=round, group=i)
                # val = mem[inp_values_p + i]
                emit("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const), tag="load", round_i=round, group=i)
                emit("load", ("load", tmp_val, tmp_addr), tag="load", round_i=round, group=i, contributes_value=True)
                emit("debug", ("compare", tmp_val, (round, i, "val")), tag="load", round_i=round, group=i)
                # node_val = mem[forest_values_p + idx]
                emit("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx), tag="gather", round_i=round, group=i)
                emit("load", ("load", tmp_node_val, tmp_addr), tag="gather", round_i=round, group=i)
                emit("debug", ("compare", tmp_node_val, (round, i, "node_val")), tag="gather", round_i=round, group=i)
                # val = myhash(val ^ node_val)
                emit("alu", ("^", tmp_val, tmp_val, tmp_node_val), tag="hash", round_i=round, group=i, contributes_value=True)
                hash_slots = (
                    self.build_hash_single_temp(tmp_val, tmp1, round, i)
                    if single_temp_hash
                    else self.build_hash(tmp_val, tmp1, tmp2, round, i)
                )
                for engine, slot in hash_slots:
                    emit(engine, slot, tag="hash", round_i=round, group=i, contributes_value=True)
                emit("debug", ("compare", tmp_val, (round, i, "hashed_val")), tag="hash", round_i=round, group=i)
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                final_round = round == rounds - 1
                if not (drop_final_index and final_round):
                    emit("alu", ("%", tmp1, tmp_val, two_const), tag="index", round_i=round, group=i, contributes_index=True, droppable_final=final_round)
                    emit("alu", ("==", tmp1, tmp1, zero_const), tag="index", round_i=round, group=i, contributes_index=True, droppable_final=final_round)
                    emit("flow", ("select", tmp3, tmp1, one_const, two_const), tag="index", round_i=round, group=i, contributes_index=True, droppable_final=final_round)
                    emit("alu", ("*", tmp_idx, tmp_idx, two_const), tag="index", round_i=round, group=i, contributes_index=True, droppable_final=final_round)
                    emit("alu", ("+", tmp_idx, tmp_idx, tmp3), tag="index", round_i=round, group=i, contributes_index=True, droppable_final=final_round)
                    emit("debug", ("compare", tmp_idx, (round, i, "next_idx")), tag="index", round_i=round, group=i, contributes_index=True, droppable_final=final_round)
                    # idx = 0 if idx >= n_nodes else idx
                    emit("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"]), tag="index", round_i=round, group=i, contributes_index=True, droppable_final=final_round)
                    emit("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const), tag="index", round_i=round, group=i, contributes_index=True, droppable_final=final_round)
                    emit("debug", ("compare", tmp_idx, (round, i, "wrapped_idx")), tag="index", round_i=round, group=i, contributes_index=True, droppable_final=final_round)
                    # mem[inp_indices_p + i] = idx
                    emit("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const), tag="store", round_i=round, group=i, contributes_index=True, droppable_final=final_round)
                    emit("store", ("store", tmp_addr, tmp_idx), tag="store", round_i=round, group=i, contributes_index=True, droppable_final=final_round)
                # mem[inp_values_p + i] = val
                emit("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const), tag="store", round_i=round, group=i, contributes_value=True)
                emit("store", ("store", tmp_addr, tmp_val), tag="store", round_i=round, group=i, contributes_value=True)

        if prune_droppable_final:
            body = [op for op in body if not op.droppable_final]

        body_instrs = self.schedule_ir(
            body,
            mode=ir_scheduler,
            weights=ir_scheduler_weights,
            preserve_memory_order=preserve_memory_order,
        )
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        if emit_debug_pauses:
            self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    for ref_mem in reference_kernel2(mem, value_trace):
        pass
    inp_values_p = ref_mem[6]
    if prints:
        print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
        print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect final values"
    inp_indices_p = ref_mem[5]
    if prints:
        print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
    # Updating these in memory isn't required, but you can enable this check for debugging
    # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
