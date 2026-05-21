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
import random
import unittest

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


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.schedule_trace = []
        self.schedule_nodes_debug = []

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def _compress_tmp2_lifetimes(self, tmp2_base: int, tmp2_len: int):
        """Post-schedule recolor of tmp2 live ranges into a smaller pool."""
        tmp2_end = tmp2_base + tmp2_len

        def in_tmp2(addr):
            return isinstance(addr, int) and tmp2_base <= addr < tmp2_end

        def vec(addr):
            return list(range(addr, addr + VLEN))

        def slot_rw(engine, slot):
            op = slot[0]
            reads = []
            writes = []
            if engine == "alu":
                _op, dest, a, b = slot
                writes = [dest]
                reads = [a, b]
            elif engine == "valu":
                if op == "vbroadcast":
                    _op, dest, src = slot
                    writes = vec(dest)
                    reads = [src]
                elif op == "multiply_add":
                    _op, dest, a, b, c = slot
                    writes = vec(dest)
                    reads = vec(a) + vec(b) + vec(c)
                else:
                    _op, dest, a, b = slot
                    writes = vec(dest)
                    reads = vec(a) + vec(b)
            elif engine == "load":
                if op == "const":
                    _op, dest, _val = slot
                    writes = [dest]
                elif op == "load":
                    _op, dest, addr = slot
                    writes = [dest]
                    reads = [addr]
                elif op == "load_offset":
                    _op, dest, addr, offset = slot
                    writes = [dest + offset]
                    reads = [addr + offset]
                elif op == "vload":
                    _op, dest, addr = slot
                    writes = vec(dest)
                    reads = [addr]
            elif engine == "store":
                if op == "store":
                    _op, addr, src = slot
                    reads = [addr, src]
                elif op == "vstore":
                    _op, addr, src = slot
                    reads = [addr] + vec(src)
            elif engine == "flow":
                if op == "select":
                    _op, dest, cond, a, b = slot
                    writes = [dest]
                    reads = [cond, a, b]
                elif op == "vselect":
                    _op, dest, cond, a, b = slot
                    writes = vec(dest)
                    reads = vec(cond) + vec(a) + vec(b)
            return [a for a in reads if in_tmp2(a)], [a for a in writes if in_tmp2(a)]

        intervals = []
        current = {}
        read_counts = []
        for cycle, instr in enumerate(self.instrs):
            reads = []
            writes = []
            for engine, slots in instr.items():
                for slot in slots:
                    slot_reads, slot_writes = slot_rw(engine, slot)
                    reads.extend(slot_reads)
                    if slot_writes:
                        ordered = sorted(slot_writes)
                        if len(ordered) == VLEN and ordered == list(range(ordered[0], ordered[0] + VLEN)):
                            writes.append(tuple(ordered))
                        else:
                            writes.extend((addr,) for addr in ordered)
            single_writes = {item[0] for item in writes if len(item) == 1}
            merged_writes = []
            consumed_writes = set()
            for addr in sorted(single_writes):
                if addr in consumed_writes:
                    continue
                vec_base = addr - ((addr - tmp2_base) % VLEN)
                vec_addrs = tuple(range(vec_base, vec_base + VLEN))
                if all(a in single_writes for a in vec_addrs):
                    merged_writes.append(vec_addrs)
                    consumed_writes.update(vec_addrs)
                else:
                    merged_writes.append((addr,))
                    consumed_writes.add(addr)
            merged_writes.extend(item for item in writes if len(item) != 1)
            writes = merged_writes
            for addr in reads:
                if addr not in current:
                    current[addr] = len(intervals)
                    intervals.append({"start": 0, "end": cycle, "width": 1, "addrs": (addr,)})
                    read_counts.append(1)
                else:
                    interval = intervals[current[addr]]
                    interval["end"] = max(interval["end"], cycle)
                    read_counts[current[addr]] += 1
            for addrs in writes:
                current_id = len(intervals)
                intervals.append({"start": cycle, "end": cycle, "width": len(addrs), "addrs": addrs})
                read_counts.append(0)
                for addr in addrs:
                    current[addr] = current_id

        active = []
        placement = {}
        pool_words = 0
        for interval_id, interval in sorted(
            enumerate(intervals),
            key=lambda item: (item[1]["start"], -item[1]["width"], item[1]["end"]),
        ):
            active = [item for item in active if item[0] >= interval["start"]]
            used = [(offset, offset + width) for _end, offset, width in active]
            offset = 0
            while True:
                if interval["width"] == VLEN and offset % VLEN:
                    offset += VLEN - (offset % VLEN)
                if all(offset + interval["width"] <= start or offset >= end for start, end in used):
                    break
                offset += 1
            placement[interval_id] = offset
            active.append((interval["end"], offset, interval["width"]))
            pool_words = max(pool_words, offset + interval["width"])

        pool_words = ((pool_words + VLEN - 1) // VLEN) * VLEN

        current = {}
        write_queues = defaultdict(list)
        for interval_id, interval in enumerate(intervals):
            write_queues[(interval["start"], interval["addrs"])].append(interval_id)

        def mapped_read(addr):
            if not in_tmp2(addr):
                return addr
            interval_id = current[addr]
            offset = placement[interval_id] + (addr - intervals[interval_id]["addrs"][0])
            return tmp2_base + offset

        def mapped_write(addr, cycle, width=1, addrs=None):
            if not in_tmp2(addr):
                return addr
            if addrs is None:
                addrs = (addr,)
            interval_id = write_queues[(cycle, tuple(addrs))].pop(0)
            for a in addrs:
                current[a] = interval_id
            offset = placement[interval_id] + (addr - intervals[interval_id]["addrs"][0])
            return tmp2_base + offset

        def rewrite_slot(engine, slot, cycle):
            op = slot[0]
            if engine == "alu":
                _op, dest, a, b = slot
                return (_op, mapped_write(dest, cycle), mapped_read(a), mapped_read(b))
            if engine == "valu":
                if op == "vbroadcast":
                    _op, dest, src = slot
                    addrs = tuple(vec(dest)) if in_tmp2(dest) else None
                    return (_op, mapped_write(dest, cycle, VLEN, addrs), mapped_read(src))
                if op == "multiply_add":
                    _op, dest, a, b, c = slot
                    addrs = tuple(vec(dest)) if in_tmp2(dest) else None
                    return (
                        _op,
                        mapped_write(dest, cycle, VLEN, addrs),
                        mapped_read(a),
                        mapped_read(b),
                        mapped_read(c),
                    )
                _op, dest, a, b = slot
                addrs = tuple(vec(dest)) if in_tmp2(dest) else None
                return (_op, mapped_write(dest, cycle, VLEN, addrs), mapped_read(a), mapped_read(b))
            if engine == "load":
                if op == "load":
                    _op, dest, addr = slot
                    return (_op, mapped_write(dest, cycle), mapped_read(addr))
                if op == "load_offset":
                    _op, dest, addr, offset = slot
                    addrs = (dest + offset,) if in_tmp2(dest + offset) else None
                    new_dest_lane = mapped_write(dest + offset, cycle, 1, addrs)
                    return (_op, new_dest_lane - offset, mapped_read(addr), offset)
                if op == "vload":
                    _op, dest, addr = slot
                    addrs = tuple(vec(dest)) if in_tmp2(dest) else None
                    return (_op, mapped_write(dest, cycle, VLEN, addrs), mapped_read(addr))
                return slot
            if engine == "store":
                if op == "store":
                    _op, addr, src = slot
                    return (_op, mapped_read(addr), mapped_read(src))
                if op == "vstore":
                    _op, addr, src = slot
                    return (_op, mapped_read(addr), mapped_read(src))
                return slot
            if engine == "flow":
                if op == "select":
                    _op, dest, cond, a, b = slot
                    return (_op, mapped_write(dest, cycle), mapped_read(cond), mapped_read(a), mapped_read(b))
                if op == "vselect":
                    _op, dest, cond, a, b = slot
                    addrs = tuple(vec(dest)) if in_tmp2(dest) else None
                    return (
                        _op,
                        mapped_write(dest, cycle, VLEN, addrs),
                        mapped_read(cond),
                        mapped_read(a),
                        mapped_read(b),
                    )
            return slot

        rewritten = []
        for cycle, instr in enumerate(self.instrs):
            rewritten.append(
                {
                    engine: [rewrite_slot(engine, slot, cycle) for slot in slots]
                    for engine, slots in instr.items()
                }
            )
        self.instrs = rewritten
        self.scratch_debug[tmp2_base] = ("tmp2", pool_words)
        if tmp2_base + pool_words == self.scratch_ptr or tmp2_base + tmp2_len == self.scratch_ptr:
            self.scratch_ptr = tmp2_base + pool_words

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        variant: dict | None = None,
    ):
        """
        SIMD/VLIW implementation of reference_kernel2.

        The submission workload is 256 independent traversals. Keep all values
        and indices resident in scratch vectors, process the full batch one
        vector operation at a time, and only write the final values back.
        """
        assert batch_size % VLEN == 0
        variant = variant or {}
        self.schedule_trace = []
        self.schedule_nodes_debug = []
        n_vecs = batch_size // VLEN
        forest_values_p = 7
        inp_values_p = forest_values_p + n_nodes + batch_size
        execution_tile_groups = variant.get("execution_tile_groups")
        if execution_tile_groups is not None and execution_tile_groups < n_vecs:
            assert n_vecs % execution_tile_groups == 0
            full_inp_values_p = inp_values_p
            child_variant = dict(variant)
            child_variant.pop("execution_tile_groups", None)
            child_variant.setdefault("cache_depth3_tree", True)
            child_variant.setdefault("cache_depth3_tree_private_start_group", 0)
            child_variant.setdefault("private_d3_pool_vectors", 4)
            child_batch_size = execution_tile_groups * VLEN

            for tile_start in range(0, n_vecs, execution_tile_groups):
                child = KernelBuilder()
                child.build_kernel(
                    forest_height,
                    n_nodes,
                    child_batch_size,
                    rounds,
                    variant=child_variant,
                )
                if tile_start == 0:
                    self.scratch = child.scratch
                    self.scratch_debug = child.scratch_debug
                    self.scratch_ptr = child.scratch_ptr
                    self.const_map = child.const_map
                else:
                    assert self.scratch_ptr == child.scratch_ptr

                idxs_addr = child.scratch["idxs"]
                vzero_addr = child.scratch["vzero"]
                reset_slots = [
                    ("+", idxs_addr + g * VLEN, vzero_addr, vzero_addr)
                    for g in range(execution_tile_groups)
                ]
                for i in range(0, len(reset_slots), SLOT_LIMITS["valu"]):
                    self.instrs.append({"valu": reset_slots[i : i + SLOT_LIMITS["valu"]]})

                value_const_addrs = {
                    child.scratch[f"value_addr_{g}"]: full_inp_values_p + (tile_start + g) * VLEN
                    for g in range(execution_tile_groups)
                }
                for instr in child.instrs:
                    patched = {}
                    for engine, slots in instr.items():
                        patched_slots = []
                        for slot in slots:
                            if (
                                engine == "load"
                                and slot[0] == "const"
                                and slot[1] in value_const_addrs
                            ):
                                patched_slots.append(("const", slot[1], value_const_addrs[slot[1]]))
                            else:
                                patched_slots.append(slot)
                        patched[engine] = patched_slots
                    self.instrs.append(patched)
                self.schedule_trace.extend(child.schedule_trace)
            return
        cache_depth3 = variant.get("cache_depth3_onehot", True)
        cache_depth3_start_group = variant.get("cache_depth3_start_group", 14)
        cache_depth3_select = variant.get("cache_depth3_select", "muladd")
        path_indices = variant.get("path_indices", True)
        # When set, depth-3 cached select uses a 3-level binary tree of
        # vselects against a shared scratch pool, instead of the linear
        # one-hot mul_add chain. Trades valu for flow ops.
        cache_depth3_tree = variant.get("cache_depth3_tree", True)
        cache_depth3_tree_private_start_group = variant.get(
            "cache_depth3_tree_private_start_group", 31
        )
        cache_depth3_shared_tree = variant.get("cache_depth3_shared_tree", False)
        compact_private_d3_tree = variant.get("compact_private_d3_tree", False)
        private_d3_pool_vectors = variant.get("private_d3_pool_vectors", 4)
        d3_tree_arena_size = variant.get("d3_tree_arena_size", 0)
        node_tmp_pool_size = variant.get("node_tmp_pool_size", n_vecs)
        gather_into_addr_tmp = variant.get("gather_into_addr_tmp", True)
        depth1_select_into_cond = variant.get("depth1_select_into_cond", True)
        depth2_select_into_tmp1 = variant.get("depth2_select_into_tmp1", True)
        depth3_onehot_select_into_tmp1 = variant.get("depth3_onehot_select_into_tmp1", True)
        need_d3_diffs = cache_depth3_select in ("tree", "pairdiff") and cache_depth3
        scalar_gather_start_group = variant.get(
            "scalar_gather_start_group", 24
        )
        scatter_vector_xor_depths = set(variant.get("scatter_vector_xor_depths", ()))
        compress_tmp2_lifetimes = variant.get("compress_tmp2_lifetimes", False)

        def emit_packed(engine, slots):
            limit = SLOT_LIMITS[engine]
            for i in range(0, len(slots), limit):
                self.instrs.append({engine: slots[i : i + limit]})

        def emit_many(engine_slots):
            for engine, slots in engine_slots.items():
                assert len(slots) <= SLOT_LIMITS[engine]
            self.instrs.append({engine: slots for engine, slots in engine_slots.items() if slots})

        def alloc_vecs(name, count=n_vecs):
            return self.alloc_scratch(name, count * VLEN)

        const_slots = []
        scalar_consts = {}

        def scalar_const(val, name=None):
            if val not in scalar_consts:
                addr = self.alloc_scratch(name)
                scalar_consts[val] = addr
                const_slots.append(("const", addr, val))
            return scalar_consts[val]

        one = scalar_const(1, "one")

        value_addrs = []
        for g in range(n_vecs):
            value_addrs.append(scalar_const(inp_values_p + g * VLEN, f"value_addr_{g}"))

        vec_consts = {}

        def vector_const(val, name=None):
            if val not in vec_consts:
                vec_consts[val] = self.alloc_scratch(name or f"vconst_{val}", VLEN)
            scalar_const(val)
            return vec_consts[val]

        vzero = self.alloc_scratch("vzero", VLEN)
        vone = vector_const(1, "vone")
        vforest = None if path_indices else vector_const(forest_values_p, "vforest_values_p")
        top_cache_nodes = 15 if cache_depth3 else 7
        vector_const(2)
        if path_indices:
            vec_consts[0] = vzero
            vector_const(3)
            if scalar_gather_start_group > 0:
                for depth in range(3, forest_height + 1):
                    vector_const(forest_values_p + (1 << depth) - 1)
            else:
                for depth in range(3, forest_height + 1):
                    scalar_const(forest_values_p + (1 << depth) - 1)
        # With path-bit indices, cached-node comparisons use compact path
        # values. Depth 3 needs 0..7, not heap node ids 7..14.
        node_const_stop = min(top_cache_nodes, 8) if path_indices else top_cache_nodes
        for node_idx in range(4, node_const_stop):
            vector_const(node_idx)
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            vector_const(val1)
            if op1 == "+" and op2 == "+" and op3 == "<<":
                vector_const((1 << val3) + 1)
            else:
                vector_const(val3)

        top_node_load_slots = []
        top_node_broadcast_slots = []
        top_node_vecs = {}
        top_node_scalars = self.alloc_scratch("top_node_scalars", top_cache_nodes)
        # Shallow tree nodes are contiguous in memory, so cache them with wide
        # loads before broadcasting each node into a lane-uniform vec.
        if top_cache_nodes > VLEN:
            top_node_load_slots.append(
                ("vload", top_node_scalars, scalar_const(forest_values_p))
            )
            last_start = top_cache_nodes - VLEN
            top_node_load_slots.append(
                (
                    "vload",
                    top_node_scalars + last_start,
                    scalar_const(forest_values_p + last_start),
                )
            )
        elif top_cache_nodes == VLEN:
            top_node_load_slots.append(
                ("vload", top_node_scalars, scalar_const(forest_values_p))
            )
        else:
            top_node_load_slots.extend(
                ("load_offset", top_node_scalars, scalar_const(forest_values_p), lane)
                for lane in range(top_cache_nodes)
            )
        for node_idx in range(top_cache_nodes):
            node_vec = self.alloc_scratch(f"vtop_node_{node_idx}", VLEN)
            top_node_vecs[node_idx] = node_vec
            top_node_broadcast_slots.append(
                ("vbroadcast", node_vec, top_node_scalars + node_idx)
            )

        # Diffs for the depth-3 tree mux (node_2k - node_(2k-1) for k=4..7).
        d3_diff_vecs = {}
        d3_diff_post_setup = []
        if need_d3_diffs:
            for k in (8, 10, 12, 14):
                d3_diff_vecs[k] = self.alloc_scratch(f"vd3_diff_{k}", VLEN)
                d3_diff_post_setup.append(
                    ("-", d3_diff_vecs[k], top_node_vecs[k], top_node_vecs[k - 1])
                )

        # Shared scratch pool for the depth-3 vselect tree mux.
        d3_tree_pool = []
        d3_tree_bit2 = None
        d3_private_tree = {}
        d3_tree_arena = []
        if (cache_depth3_tree or cache_depth3_select == "pairdiff") and cache_depth3:
            for g in range(cache_depth3_tree_private_start_group, n_vecs):
                tree_state = {
                    "pool": [
                        self.alloc_scratch(f"vd3_private_pool_{g}_{i}", VLEN)
                        for i in range(private_d3_pool_vectors)
                    ],
                }
                if not compact_private_d3_tree:
                    tree_state["bit2"] = self.alloc_scratch(f"vd3_private_bit2_{g}", VLEN)
                d3_private_tree[g] = tree_state
            for arena_i in range(d3_tree_arena_size):
                tree_state = {
                    "pool": [
                        self.alloc_scratch(f"vd3_arena_{arena_i}_pool_{i}", VLEN)
                        for i in range(private_d3_pool_vectors)
                    ],
                }
                if not compact_private_d3_tree:
                    tree_state["bit2"] = self.alloc_scratch(f"vd3_arena_{arena_i}_bit2", VLEN)
                d3_tree_arena.append(tree_state)
            if cache_depth3_shared_tree and cache_depth3_tree_private_start_group > cache_depth3_start_group:
                for i in range(4):
                    d3_tree_pool.append(self.alloc_scratch(f"vd3_pool_{i}", VLEN))
                d3_tree_bit2 = self.alloc_scratch("vd3_bit2", VLEN)

        vals = alloc_vecs("vals")
        idxs = alloc_vecs("idxs")
        tmp1 = alloc_vecs("tmp1")
        tmp2 = alloc_vecs("tmp2", node_tmp_pool_size)

        def node_tmp_addr(group):
            return tmp2 + (group % node_tmp_pool_size) * VLEN

        emit_packed("load", const_slots)
        emit_packed("load", top_node_load_slots)
        emit_packed(
            "valu",
            [
                ("vbroadcast", addr, scalar_consts[val])
                for val, addr in vec_consts.items()
                if val in scalar_consts
            ]
            + top_node_broadcast_slots,
        )
        if d3_diff_post_setup:
            emit_packed("valu", d3_diff_post_setup)
        nodes = []
        ready_by_engine = {engine: [] for engine in SLOT_LIMITS}
        current_component = ["setup"]

        def set_component(name):
            current_component[0] = name

        def as_deps(deps):
            if deps is None:
                return []
            if isinstance(deps, int):
                return [deps]
            return list(deps)

        def add_node(engine, slot, deps=None, component=None):
            deps = as_deps(deps)
            node_id = len(nodes)
            node = {
                "engine": engine,
                "slot": slot,
                "component": component or current_component[0],
                "deps": tuple(deps),
                "deps_left": len(deps),
                "succs": [],
            }
            nodes.append(node)
            for dep in deps:
                nodes[dep]["succs"].append(node_id)
            if not deps:
                ready_by_engine[engine].append(node_id)
            return node_id

        scheduler_mode = variant.get("scheduler", "fifo")
        dep_mode = variant.get("dep_mode", "explicit")

        def schedule_nodes():
            if dep_mode.startswith("rw"):
                def vec(addr):
                    return range(addr, addr + VLEN)

                def rw_sets(engine, slot):
                    op = slot[0]
                    reads, writes = set(), set()
                    if engine == "alu":
                        _op, dest, a, b = slot
                        writes.add(dest)
                        reads.update((a, b))
                    elif engine == "valu":
                        if op == "vbroadcast":
                            _op, dest, src = slot
                            writes.update(vec(dest))
                            reads.add(src)
                        elif op == "multiply_add":
                            _op, dest, a, b, c = slot
                            writes.update(vec(dest))
                            reads.update(vec(a))
                            reads.update(vec(b))
                            reads.update(vec(c))
                        else:
                            _op, dest, a, b = slot
                            writes.update(vec(dest))
                            reads.update(vec(a))
                            reads.update(vec(b))
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
                            writes.update(vec(dest))
                            reads.add(addr)
                    elif engine == "store":
                        if op == "store":
                            _op, addr, src = slot
                            reads.update((addr, src))
                        elif op == "vstore":
                            _op, addr, src = slot
                            reads.add(addr)
                            reads.update(vec(src))
                    elif engine == "flow":
                        if op == "select":
                            _op, dest, cond, a, b = slot
                            writes.add(dest)
                            reads.update((cond, a, b))
                        elif op == "vselect":
                            _op, dest, cond, a, b = slot
                            writes.update(vec(dest))
                            reads.update(vec(cond))
                            reads.update(vec(a))
                            reads.update(vec(b))
                    return reads, writes

                if dep_mode == "rw_latency":
                    edge_latency = {}
                    last_writer = {}
                    readers_since_write = defaultdict(list)
                    for node_id, node in enumerate(nodes):
                        reads, writes = rw_sets(node["engine"], node["slot"])
                        for addr in reads:
                            if addr in last_writer:
                                edge = (last_writer[addr], node_id)
                                if edge[0] != edge[1]:
                                    edge_latency[edge] = max(edge_latency.get(edge, 0), 1)
                            readers_since_write[addr].append(node_id)
                        for addr in writes:
                            if addr in last_writer:
                                edge = (last_writer[addr], node_id)
                                if edge[0] != edge[1]:
                                    edge_latency[edge] = max(edge_latency.get(edge, 0), 1)
                            for reader in readers_since_write[addr]:
                                edge = (reader, node_id)
                                if edge[0] != edge[1]:
                                    edge_latency.setdefault(edge, 0)
                            readers_since_write[addr].clear()
                            last_writer[addr] = node_id

                    succ_edges = [[] for _ in nodes]
                    wait_prev = [0] * len(nodes)
                    wait_same = [0] * len(nodes)
                    deps_for_debug = [set() for _ in nodes]
                    for (pred, succ), latency in edge_latency.items():
                        succ_edges[pred].append((succ, latency))
                        deps_for_debug[succ].add(pred)
                        if latency:
                            wait_prev[succ] += 1
                        else:
                            wait_same[succ] += 1
                    for node_id, node in enumerate(nodes):
                        node["deps"] = tuple(sorted(deps_for_debug[node_id]))
                        node["deps_left"] = wait_prev[node_id] + wait_same[node_id]
                        node["succs"] = [succ for succ, _latency in succ_edges[node_id]]

                    ready_now = {
                        engine: [
                            node_id
                            for node_id, node in enumerate(nodes)
                            if node["engine"] == engine
                            and wait_prev[node_id] == 0
                            and wait_same[node_id] == 0
                        ]
                        for engine in SLOT_LIMITS
                    }
                    scheduled = [False] * len(nodes)
                    remaining = len(nodes)

                    while remaining:
                        instr = {}
                        chosen = []
                        chosen_set = set()
                        capacity = {
                            engine: SLOT_LIMITS[engine]
                            for engine in ("load", "valu", "alu", "store", "flow")
                        }
                        progressed = True
                        while progressed:
                            progressed = False
                            for engine in ("load", "valu", "alu", "store", "flow"):
                                ready = ready_now[engine]
                                while ready and capacity[engine] > 0:
                                    node_id = ready.pop(0)
                                    if scheduled[node_id] or node_id in chosen_set:
                                        continue
                                    instr.setdefault(engine, []).append(nodes[node_id]["slot"])
                                    chosen.append(node_id)
                                    chosen_set.add(node_id)
                                    capacity[engine] -= 1
                                    progressed = True
                                    for succ, latency in succ_edges[node_id]:
                                        if latency == 0:
                                            wait_same[succ] -= 1
                                            if (
                                                wait_prev[succ] == 0
                                                and wait_same[succ] == 0
                                                and not scheduled[succ]
                                                and succ not in chosen_set
                                            ):
                                                ready_now[nodes[succ]["engine"]].append(succ)
                        assert chosen, "scheduler stalled"
                        self.instrs.append(instr)
                        self.schedule_trace.append(
                            [
                                (
                                    nodes[node_id]["component"],
                                    nodes[node_id]["engine"],
                                    nodes[node_id]["slot"],
                                )
                                for node_id in chosen
                            ]
                        )
                        remaining -= len(chosen)
                        newly_ready_next = []
                        for node_id in chosen:
                            scheduled[node_id] = True
                            for succ, latency in succ_edges[node_id]:
                                if latency:
                                    wait_prev[succ] -= 1
                                    if (
                                        wait_prev[succ] == 0
                                        and wait_same[succ] == 0
                                        and not scheduled[succ]
                                    ):
                                        newly_ready_next.append(succ)
                        for succ in newly_ready_next:
                            ready_now[nodes[succ]["engine"]].append(succ)
                    return

                for node in nodes:
                    node["succs"] = []
                    node["deps_left"] = 0
                last_writer = {}
                readers_since_write = defaultdict(list)
                for node_id, node in enumerate(nodes):
                    reads, writes = rw_sets(node["engine"], node["slot"])
                    deps = set()
                    for addr in reads:
                        if addr in last_writer:
                            deps.add(last_writer[addr])
                        readers_since_write[addr].append(node_id)
                    for addr in writes:
                        if addr in last_writer:
                            deps.add(last_writer[addr])
                        if dep_mode == "rw_with_war":
                            deps.update(readers_since_write[addr])
                        readers_since_write[addr].clear()
                        last_writer[addr] = node_id
                    node["deps"] = tuple(sorted(deps))
                    node["deps_left"] = len(deps)
                    for dep in deps:
                        nodes[dep]["succs"].append(node_id)
                for engine in ready_by_engine:
                    ready_by_engine[engine].clear()
                for node_id, node in enumerate(nodes):
                    if node["deps_left"] == 0:
                        ready_by_engine[node["engine"]].append(node_id)

            # Compute critical-path height per node for some scheduler modes.
            height = None
            valu_height = None
            if scheduler_mode in ("cp", "lp", "valu_cp"):
                height = [0] * len(nodes)
                for nid in range(len(nodes) - 1, -1, -1):
                    h = 0
                    for s in nodes[nid]["succs"]:
                        if height[s] + 1 > h:
                            h = height[s] + 1
                    height[nid] = h
            if scheduler_mode == "valu_cp":
                # Height through valu ops only - measures the remaining
                # bottleneck chain on the limiting engine.
                valu_height = [0] * len(nodes)
                for nid in range(len(nodes) - 1, -1, -1):
                    h = 0
                    for s in nodes[nid]["succs"]:
                        s_h = valu_height[s] + (1 if nodes[s]["engine"] == "valu" else 0)
                        if s_h > h:
                            h = s_h
                    valu_height[nid] = h

            sort_key_by_mode = {
                "cp": lambda n: -height[n],
                "lp": lambda n: height[n],
                "valu_cp": lambda n: -valu_height[n],
            }
            sort_key = sort_key_by_mode.get(scheduler_mode)

            if sort_key is not None:
                for engine in ready_by_engine:
                    ready_by_engine[engine].sort(key=sort_key)

            remaining = len(nodes)
            while remaining:
                instr = {}
                chosen = []
                if sort_key is not None:
                    for engine in ("load", "valu", "alu", "store", "flow"):
                        ready = ready_by_engine[engine]
                        if not ready:
                            continue
                        limit = SLOT_LIMITS[engine]
                        ready.sort(key=sort_key)
                        take = ready[:limit]
                        del ready[:limit]
                        instr[engine] = [nodes[node_id]["slot"] for node_id in take]
                        chosen.extend(take)
                else:
                    for engine in ("load", "valu", "alu", "store", "flow"):
                        ready = ready_by_engine[engine]
                        limit = SLOT_LIMITS[engine]
                        if ready:
                            take = ready[:limit]
                            del ready[:limit]
                            instr[engine] = [nodes[node_id]["slot"] for node_id in take]
                            chosen.extend(take)
                assert chosen, "scheduler stalled"
                self.instrs.append(instr)
                self.schedule_trace.append(
                    [
                        (
                            nodes[node_id]["component"],
                            nodes[node_id]["engine"],
                            nodes[node_id]["slot"],
                        )
                        for node_id in chosen
                    ]
                )
                remaining -= len(chosen)
                for node_id in chosen:
                    for succ in nodes[node_id]["succs"]:
                        nodes[succ]["deps_left"] -= 1
                        if nodes[succ]["deps_left"] == 0:
                            ready_by_engine[nodes[succ]["engine"]].append(succ)

        value_ready = []
        idx_ready = []
        set_component("setup")
        for g in range(n_vecs):
            value_ready.append(add_node("load", ("vload", vals + g * VLEN, value_addrs[g])))
            idx_ready.append([])

        # For hash stages without multiply_add, let vector slots handle most
        # groups and scalar ALU slots work on the tail lanes in parallel.
        scalar_hash_start_group = variant.get("scalar_hash_start_group", 26)
        scalar_index_start_group = variant.get(
            "scalar_index_start_group", 26
        )
        scalar_hash_start_by_stage = {
            int(k): v
            for k, v in variant.get(
                "scalar_hash_start_by_stage", {1: 26, 3: 23, 5: 25}
            ).items()
        }
        # Optional per-depth override (depth -> stage -> threshold).
        scalar_hash_start_by_depth_stage = {
            int(d): {int(s): v for s, v in stages.items()}
            for d, stages in variant.get("scalar_hash_start_by_depth_stage", {}).items()
        }
        scalar_muladd_hash_start_group = variant.get(
            "scalar_muladd_hash_start_group", n_vecs
        )
        hybrid_hash_start_by_stage = {
            int(k): v
            for k, v in variant.get("hybrid_hash_start_by_stage", {}).items()
        }
        # If True, non-mul_add hash stages (1, 3, 5) route h2 through `vals`
        # (and h1 through tmp1) instead of tmp2, freeing tmp2 during hash so
        # that the `doubled` op of the index update can be scheduled earlier
        # (during the load-bottlenecked dip).
        early_doubled = variant.get("early_doubled", False)
        idx_doubled_in_idx = variant.get("idx_doubled_in_idx", True)
        single_temp_hash = variant.get("single_temp_hash", True)
        hash_h2_in_vals = variant.get("hash_h2_in_vals", single_temp_hash)

        group_order = variant.get("group_order", "forward")
        group_tile_size = variant.get("group_tile_size", n_vecs)
        round_tile_size = variant.get("round_tile_size", 1)
        # State for tree mux pool serialization across cached groups within
        # a single depth-3 round. Keys: 0,1,2,3 for pool[0..3], 'bit2' for
        # shared bit2 scratch. Values: last writer node id (or None).
        d3_pool_last_users = {0: None, 1: None, 2: None, 3: None, "bit2": None}
        d3_arena_last_users = [
            {0: None, 1: None, 2: None, 3: None, "bit2": None}
            for _ in d3_tree_arena
        ]
        node_tmp_last_users = {slot: None for slot in range(node_tmp_pool_size)}

        def ordered_groups(start=0, stop=n_vecs):
            if group_order == "reverse":
                return list(range(stop - 1, start - 1, -1))
            if group_order == "interleave":
                return list(range(start, stop, 2)) + list(range(start + 1, stop, 2))
            return list(range(start, stop))

        iteration_order = []
        if group_tile_size >= n_vecs and round_tile_size <= 1:
            for round_i in range(rounds):
                for g in ordered_groups():
                    iteration_order.append((round_i, g))
        else:
            for round_start in range(0, rounds, round_tile_size):
                round_stop = min(round_start + round_tile_size, rounds)
                for group_start in range(0, n_vecs, group_tile_size):
                    group_stop = min(group_start + group_tile_size, n_vecs)
                    for round_i in range(round_start, round_stop):
                        for g in ordered_groups(group_start, group_stop):
                            iteration_order.append((round_i, g))

        last_round_i = None
        for round_i, g in iteration_order:
            depth = round_i % (forest_height + 1)
            if round_i != last_round_i:
                # Reset pool serialization state per round.
                d3_pool_last_users = {0: None, 1: None, 2: None, 3: None, "bit2": None}
                d3_arena_last_users = [
                    {0: None, 1: None, 2: None, 3: None, "bit2": None}
                    for _ in d3_tree_arena
                ]
                last_round_i = round_i
            if True:
                base = g * VLEN
                deps = as_deps(value_ready[g]) + as_deps(idx_ready[g])
                node_tmp = node_tmp_addr(g)
                node_tmp_slot = g % node_tmp_pool_size
                node_tmp_extra_deps = (
                    as_deps(node_tmp_last_users[node_tmp_slot])
                    if node_tmp_last_users[node_tmp_slot] is not None
                    else []
                )

                addr_ready = None
                set_component("gather")
                if depth == 0:
                    if g >= scalar_gather_start_group:
                        cur = [
                            add_node(
                                "alu",
                                ("^", vals + base + lane, vals + base + lane, top_node_vecs[0] + lane),
                                value_ready[g],
                            )
                            for lane in range(VLEN)
                        ]
                    else:
                        cur = add_node(
                            "valu",
                            ("^", vals + base, vals + base, top_node_vecs[0]),
                            value_ready[g],
                        )
                elif depth == 1:
                    set_component("select")
                    cond = add_node(
                        "valu",
                        (
                            "==",
                            tmp1 + base,
                            idxs + base,
                            vec_consts[1] if path_indices else vec_consts[2],
                        ),
                        deps,
                    )
                    depth1_selected_addr = tmp1 + base if depth1_select_into_cond else node_tmp
                    selected = add_node(
                        "flow",
                        (
                            "vselect",
                            depth1_selected_addr,
                            tmp1 + base,
                            top_node_vecs[2],
                            top_node_vecs[1],
                        ),
                        [cond] + ([] if depth1_select_into_cond else node_tmp_extra_deps),
                    )
                    if g >= scalar_gather_start_group:
                        cur = [
                            add_node(
                                "alu",
                                ("^", vals + base + lane, vals + base + lane, depth1_selected_addr + lane),
                                as_deps(value_ready[g]) + [selected],
                            )
                            for lane in range(VLEN)
                        ]
                    else:
                        cur = add_node(
                            "valu",
                            ("^", vals + base, vals + base, depth1_selected_addr),
                            as_deps(value_ready[g]) + [selected],
                        )
                    if not depth1_select_into_cond:
                        node_tmp_last_users[node_tmp_slot] = cur
                elif depth == 2:
                    set_component("select")
                    depth2_selected_addr = tmp1 + base if depth2_select_into_tmp1 else node_tmp
                    depth2_cond_addr = node_tmp if depth2_select_into_tmp1 else tmp1 + base
                    selected = add_node(
                        "valu",
                        ("+", depth2_selected_addr, top_node_vecs[3], vzero),
                        deps + node_tmp_extra_deps,
                    )
                    for node_idx in range(4, 7):
                        compare_val = node_idx - 3 if path_indices else node_idx
                        cond = add_node(
                            "valu",
                            ("==", depth2_cond_addr, idxs + base, vec_consts[compare_val]),
                            as_deps(idx_ready[g]) + [selected],
                        )
                        selected = add_node(
                            "flow",
                            (
                                "vselect",
                                depth2_selected_addr,
                                depth2_cond_addr,
                                top_node_vecs[node_idx],
                                depth2_selected_addr,
                            ),
                            cond,
                        )
                    if g >= scalar_gather_start_group:
                        cur = [
                            add_node(
                                "alu",
                                ("^", vals + base + lane, vals + base + lane, depth2_selected_addr + lane),
                                as_deps(value_ready[g]) + [selected],
                            )
                            for lane in range(VLEN)
                        ]
                    else:
                        cur = add_node(
                            "valu",
                            ("^", vals + base, vals + base, depth2_selected_addr),
                            as_deps(value_ready[g]) + [selected],
                        )
                    node_tmp_last_users[node_tmp_slot] = selected if depth2_select_into_tmp1 else cur
                elif depth == 3 and cache_depth3 and g >= cache_depth3_start_group:
                    set_component("select")
                    if cache_depth3_select == "pairdiff" and g in d3_private_tree:
                        tree_state = d3_private_tree[g]
                        tree_pool = tree_state["pool"]
                        tree_bit2 = tree_state.get("bit2")
                        bit0_addr = tmp1 + base
                        bit1_addr = node_tmp
                        bit2_addr = tree_bit2 if tree_bit2 is not None else bit0_addr
                        idx_deps = as_deps(idx_ready[g])
                        bit0 = add_node(
                            "valu",
                            ("&", bit0_addr, idxs + base, vone),
                            idx_deps,
                        )
                        shifted1 = add_node(
                            "valu",
                            (">>", bit1_addr, idxs + base, vone),
                            idx_deps + node_tmp_extra_deps,
                        )
                        bit1 = add_node(
                            "valu",
                            ("&", bit1_addr, bit1_addr, vone),
                            shifted1,
                        )
                        if tree_bit2 is not None:
                            bit2 = add_node(
                                "valu",
                                (">>", bit2_addr, idxs + base, vec_consts[2]),
                                idx_deps,
                            )
                        else:
                            bit2 = None

                        pair0 = add_node(
                            "valu",
                            ("multiply_add", tree_pool[0], bit0_addr, d3_diff_vecs[8], top_node_vecs[7]),
                            bit0,
                        )
                        pair1 = add_node(
                            "valu",
                            ("multiply_add", tree_pool[1], bit0_addr, d3_diff_vecs[10], top_node_vecs[9]),
                            bit0,
                        )
                        pair2 = add_node(
                            "valu",
                            ("multiply_add", tree_pool[2], bit0_addr, d3_diff_vecs[12], top_node_vecs[11]),
                            bit0,
                        )
                        pair3 = add_node(
                            "valu",
                            ("multiply_add", tree_pool[3], bit0_addr, d3_diff_vecs[14], top_node_vecs[13]),
                            bit0,
                        )
                        if bit2 is None:
                            bit2 = add_node(
                                "valu",
                                (">>", bit2_addr, idxs + base, vec_consts[2]),
                                [pair0, pair1, pair2, pair3] + idx_deps,
                            )
                        diff_a = add_node(
                            "valu",
                            ("-", tree_pool[1], tree_pool[1], tree_pool[0]),
                            [pair0, pair1],
                        )
                        quad_a = add_node(
                            "valu",
                            ("multiply_add", tree_pool[0], bit1_addr, tree_pool[1], tree_pool[0]),
                            [bit1, diff_a],
                        )
                        diff_b = add_node(
                            "valu",
                            ("-", tree_pool[3], tree_pool[3], tree_pool[2]),
                            [pair2, pair3],
                        )
                        quad_b = add_node(
                            "valu",
                            ("multiply_add", tree_pool[2], bit1_addr, tree_pool[3], tree_pool[2]),
                            [bit1, diff_b],
                        )
                        diff_final = add_node(
                            "valu",
                            ("-", tree_pool[2], tree_pool[2], tree_pool[0]),
                            [quad_a, quad_b],
                        )
                        selected = add_node(
                            "valu",
                            ("multiply_add", tree_pool[0], bit2_addr, tree_pool[2], tree_pool[0]),
                            [bit2, diff_final],
                        )
                        if g >= scalar_gather_start_group:
                            cur = [
                                add_node(
                                    "alu",
                                    ("^", vals + base + lane, vals + base + lane, tree_pool[0] + lane),
                                    as_deps(value_ready[g]) + [selected],
                                )
                                for lane in range(VLEN)
                            ]
                        else:
                            cur = add_node(
                                "valu",
                                ("^", vals + base, vals + base, tree_pool[0]),
                                as_deps(value_ready[g]) + [selected],
                            )
                        cur_deps = as_deps(cur)
                        gather_done_deps = list(cur_deps)
                        node_tmp_last_users[node_tmp_slot] = cur
                        set_component("hash")
                        for hash_stage in range(len(HASH_STAGES)):
                            cur_deps = add_hash_stage(hash_stage, cur_deps)
                        value_ready[g] = cur_deps
                        if round_i == rounds - 1:
                            continue
                        if depth == forest_height:
                            idx_ready[g] = []
                            continue
                        set_component("index")
                        if g >= scalar_index_start_group:
                            next_idx = []
                            for lane in range(VLEN):
                                parity = add_node(
                                    "alu",
                                    ("&", tmp1 + base + lane, vals + base + lane, one),
                                    cur_deps,
                                )
                                doubled = add_node(
                                    "alu",
                                    (
                                        "<<",
                                        (idxs if idx_doubled_in_idx and path_indices else tmp2)
                                        + base
                                        + lane,
                                        idxs + base + lane,
                                        one,
                                    ),
                                    cur_deps + as_deps(idx_ready[g]),
                                )
                                next_idx.append(add_node(
                                    "alu",
                                    (
                                        "+",
                                        idxs + base + lane,
                                        (idxs if idx_doubled_in_idx else tmp2) + base + lane,
                                        tmp1 + base + lane,
                                    ),
                                    [doubled, parity],
                                ))
                            idx_ready[g] = next_idx
                        else:
                            parity = add_node(
                                "valu",
                                ("&", tmp1 + base, vals + base, vone),
                                cur_deps,
                            )
                            idx_ready[g] = add_node(
                                "valu",
                                ("multiply_add", idxs + base, idxs + base, vec_consts[2], tmp1 + base),
                                as_deps(idx_ready[g]) + [parity],
                            )
                        continue

                    if cache_depth3_tree and (
                        g in d3_private_tree or d3_tree_arena or cache_depth3_shared_tree
                    ):
                        # 3-level binary tree mux. Pairs use vselect (flow),
                        # upper layers use mul_add (valu). Private pools avoid
                        # serializing cached groups when scratch is available.
                        arena_last_users = None
                        tree_state = d3_private_tree.get(g)
                        if tree_state is not None:
                            tree_pool = tree_state["pool"]
                            tree_bit2 = tree_state.get("bit2")
                            private_tree_pool = True
                        elif d3_tree_arena:
                            arena_slot = (g - cache_depth3_start_group) % len(d3_tree_arena)
                            tree_state = d3_tree_arena[arena_slot]
                            tree_pool = tree_state["pool"]
                            tree_bit2 = tree_state.get("bit2")
                            arena_last_users = d3_arena_last_users[arena_slot]
                            private_tree_pool = False
                        else:
                            tree_pool = d3_tree_pool
                            tree_bit2 = d3_tree_bit2
                            arena_last_users = d3_pool_last_users
                            private_tree_pool = False
                        bit0_addr = tmp1 + base
                        bit1_addr = tmp2 + base
                        bit2_addr = tree_bit2 if tree_bit2 is not None else bit0_addr
                        idx_deps = as_deps(idx_ready[g])
                        bit0 = add_node(
                            "valu",
                            ("&", bit0_addr, idxs + base, vone),
                            idx_deps,
                        )
                        shifted1 = add_node(
                            "valu",
                            (">>", bit1_addr, idxs + base, vone),
                            idx_deps,
                        )
                        bit1 = add_node(
                            "valu",
                            ("&", bit1_addr, bit1_addr, vone),
                            shifted1,
                        )
                        bit2_extra_deps = (
                            [arena_last_users["bit2"]]
                            if not private_tree_pool
                            and arena_last_users is not None
                            and arena_last_users["bit2"] is not None
                            else []
                        )
                        if tree_bit2 is not None:
                            bit2 = add_node(
                                "valu",
                                (">>", bit2_addr, idxs + base, vec_consts[2]),
                                idx_deps + bit2_extra_deps,
                            )
                        else:
                            bit2 = None

                        def pool_pair(slot, bit_node, bit_addr, node_hi, node_lo, dep_writer):
                            extra = []
                            if dep_writer is not None and (
                                not private_tree_pool or len(tree_pool) < 4
                            ):
                                extra = list(dep_writer) if isinstance(dep_writer, list) else [dep_writer]
                            return add_node(
                                "flow",
                                (
                                    "vselect",
                                    tree_pool[slot],
                                    bit_addr,
                                    top_node_vecs[node_hi],
                                    top_node_vecs[node_lo],
                                ),
                                [bit_node] + extra,
                            )

                        p0 = pool_pair(
                            0,
                            bit0,
                            bit0_addr,
                            8,
                            7,
                            arena_last_users[0] if arena_last_users is not None else None,
                        )
                        p1 = pool_pair(
                            1,
                            bit0,
                            bit0_addr,
                            10,
                            9,
                            arena_last_users[1] if arena_last_users is not None else None,
                        )

                        # Upper layer via mul_add: quad_a = bit_1 * (pool[1] - pool[0]) + pool[0]
                        diff_a = add_node(
                            "valu",
                            ("-", tree_pool[1], tree_pool[1], tree_pool[0]),
                            [p0, p1],
                        )
                        quad_a = add_node(
                            "valu",
                            ("multiply_add", tree_pool[0], bit1_addr, tree_pool[1], tree_pool[0]),
                            [bit1, diff_a],
                        )
                        if len(tree_pool) >= 4:
                            p2 = pool_pair(
                                2,
                                bit0,
                                bit0_addr,
                                12,
                                11,
                                arena_last_users[2] if arena_last_users is not None else None,
                            )
                            p3 = pool_pair(
                                3,
                                bit0,
                                bit0_addr,
                                14,
                                13,
                                arena_last_users[3] if arena_last_users is not None else None,
                            )

                            if bit2 is None:
                                bit2 = add_node(
                                    "valu",
                                    (">>", bit2_addr, idxs + base, vec_consts[2]),
                                    [p0, p1, p2, p3] + idx_deps + bit2_extra_deps,
                                )
                            diff_b = add_node(
                                "valu",
                                ("-", tree_pool[3], tree_pool[3], tree_pool[2]),
                                [p2, p3],
                            )
                            quad_b = add_node(
                                "valu",
                                ("multiply_add", tree_pool[2], bit1_addr, tree_pool[3], tree_pool[2]),
                                [bit1, diff_b],
                            )
                            diff_final = add_node(
                                "valu",
                                ("-", tree_pool[2], tree_pool[2], tree_pool[0]),
                                [quad_a, quad_b],
                            )
                            selected = add_node(
                                "valu",
                                ("multiply_add", tree_pool[0], bit2_addr, tree_pool[2], tree_pool[0]),
                                [bit2, diff_final],
                            )
                        elif len(tree_pool) == 3:
                            # Three-pool variant: q0 in pool[0], second-half
                            # pairs in pool[1]/pool[2], q1 back into pool[1].
                            # This saves one vector per group versus the
                            # 4-pool form without reusing bit scratch for pair3.
                            p2 = pool_pair(1, bit0, bit0_addr, 12, 11, quad_a)
                            p3 = pool_pair(
                                2,
                                bit0,
                                bit0_addr,
                                14,
                                13,
                                arena_last_users[2] if arena_last_users is not None else None,
                            )
                            if bit2 is None:
                                bit2 = add_node(
                                    "valu",
                                    (">>", bit0_addr, idxs + base, vec_consts[2]),
                                    [p0, p1, p2, p3] + idx_deps + bit2_extra_deps,
                                )
                                bit2_addr = bit0_addr
                            diff_b = add_node(
                                "valu",
                                ("-", tree_pool[2], tree_pool[2], tree_pool[1]),
                                [p2, p3],
                            )
                            quad_b = add_node(
                                "valu",
                                ("multiply_add", tree_pool[1], bit1_addr, tree_pool[2], tree_pool[1]),
                                [bit1, diff_b],
                            )
                            diff_final = add_node(
                                "valu",
                                ("-", tree_pool[1], tree_pool[1], tree_pool[0]),
                                [quad_a, quad_b],
                            )
                            selected = add_node(
                                "valu",
                                ("multiply_add", tree_pool[0], bit2_addr, tree_pool[1], tree_pool[0]),
                                [bit2, diff_final],
                            )
                        else:
                            # Two-pool variant: true 7-vselect tree. Keep q0 in
                            # pool[0], use pool[1] for pair1/pair2/q1, reuse
                            # bit0 scratch for pair3 and then b2.
                            p2 = pool_pair(1, bit0, bit0_addr, 12, 11, quad_a)
                            p3 = add_node(
                                "flow",
                                (
                                    "vselect",
                                    bit0_addr,
                                    bit0_addr,
                                    top_node_vecs[14],
                                    top_node_vecs[13],
                                ),
                                [bit0, p0, p1, p2],
                            )
                            q1 = add_node(
                                "flow",
                                (
                                    "vselect",
                                    tree_pool[1],
                                    bit1_addr,
                                    bit0_addr,
                                    tree_pool[1],
                                ),
                                [bit1, p2, p3],
                            )
                            quad_b = q1
                            if bit2 is None:
                                bit2 = add_node(
                                    "valu",
                                    (">>", bit0_addr, idxs + base, vec_consts[2]),
                                    [q1] + idx_deps + bit2_extra_deps,
                                )
                                bit2_addr = bit0_addr
                            selected = add_node(
                                "flow",
                                (
                                    "vselect",
                                    tree_pool[0],
                                    bit2_addr,
                                    tree_pool[1],
                                    tree_pool[0],
                                ),
                                [bit2, quad_a, q1],
                            )
                        if g >= scalar_gather_start_group:
                            cur = [
                                add_node(
                                    "alu",
                                    ("^", vals + base + lane, vals + base + lane, tree_pool[0] + lane),
                                    as_deps(value_ready[g]) + [selected],
                                )
                                for lane in range(VLEN)
                            ]
                        else:
                            cur = add_node(
                                "valu",
                                ("^", vals + base, vals + base, tree_pool[0]),
                                as_deps(value_ready[g]) + [selected],
                            )
                        if not private_tree_pool and arena_last_users is not None:
                            # Update pool last users (the LAST reader of each
                            # pool slot is what the next group must wait for).
                            # pool[0] is finally read by the gather xor `cur`.
                            arena_last_users[0] = cur
                            arena_last_users[1] = quad_a
                            arena_last_users[2] = selected
                            arena_last_users[3] = quad_b
                            arena_last_users["bit2"] = selected
                        cur_deps = as_deps(cur)
                        gather_done_deps = list(cur_deps)
                        node_tmp_last_users[node_tmp_slot] = cur
                        # Skip the rest of the select branch.
                        # Continue to hash directly.
                        set_component("hash")
                        for hash_stage in range(len(HASH_STAGES)):
                            cur_deps = add_hash_stage(hash_stage, cur_deps)
                        value_ready[g] = cur_deps
                        if round_i == rounds - 1:
                            continue
                        if depth == forest_height:
                            idx_ready[g] = []
                            continue
                        # Standard index update fallthrough; reuse the standard
                        # path below by re-entering it. We use a continue-loop
                        # trick: jump to the index update logic by skipping the
                        # normal select/hash/index path.
                        # Easiest: duplicate index update logic here.
                        set_component("index")
                        # depth >= 1 path with path_indices true (cache_depth3 implies)
                        if g >= scalar_index_start_group:
                            next_idx = []
                            for lane in range(VLEN):
                                parity = add_node(
                                    "alu",
                                    ("&", tmp1 + base + lane, vals + base + lane, one),
                                    cur_deps,
                                )
                                doubled_deps = (
                                    gather_done_deps + as_deps(idx_ready[g])
                                    if early_doubled
                                    else cur_deps + as_deps(idx_ready[g])
                                )
                                doubled = add_node(
                                    "alu",
                                    (
                                        "<<",
                                        (idxs if idx_doubled_in_idx and path_indices else tmp2)
                                        + base
                                        + lane,
                                        idxs + base + lane,
                                        one,
                                    ),
                                    doubled_deps,
                                )
                                if path_indices:
                                    next_idx.append(add_node(
                                        "alu",
                                        (
                                            "+",
                                            idxs + base + lane,
                                            (idxs if idx_doubled_in_idx else tmp2) + base + lane,
                                            tmp1 + base + lane,
                                        ),
                                        [doubled, parity],
                                    ))
                                else:
                                    child = add_node(
                                        "alu", ("+", tmp1 + base + lane, tmp1 + base + lane, one), parity)
                                    next_idx.append(add_node(
                                        "alu", ("+", idxs + base + lane, tmp2 + base + lane, tmp1 + base + lane),
                                        [doubled, child],
                                    ))
                            idx_ready[g] = next_idx
                        else:
                            parity = add_node(
                                "valu",
                                ("&", tmp1 + base, vals + base, vone),
                                cur_deps,
                            )
                            if path_indices:
                                idx_ready[g] = add_node(
                                    "valu",
                                    ("multiply_add", idxs + base, idxs + base, vec_consts[2], tmp1 + base),
                                    as_deps(idx_ready[g]) + [parity],
                                )
                            else:
                                child = add_node(
                                    "valu",
                                    ("+", tmp1 + base, tmp1 + base, vone),
                                    parity,
                                )
                                idx_ready[g] = add_node(
                                    "valu",
                                    ("multiply_add", idxs + base, idxs + base, vec_consts[2], tmp1 + base),
                                    as_deps(idx_ready[g]) + [child],
                                )
                        continue
                    # cache_depth3_vselect_count: how many of the trailing 8
                    # one-hot accumulation steps use vselect (flow) instead of
                    # multiply_add (valu). For one-hot conditions, the two ops
                    # produce identical results but use different engines.
                    vsel_count = variant.get("cache_depth3_vselect_count", 0)
                    depth3_selected_addr = (
                        tmp1 + base if depth3_onehot_select_into_tmp1 else node_tmp
                    )
                    depth3_cond_addr = (
                        node_tmp if depth3_onehot_select_into_tmp1 else tmp1 + base
                    )
                    selected = add_node(
                        "valu",
                        ("+", depth3_selected_addr, vzero, vzero),
                        deps + node_tmp_extra_deps,
                    )
                    for i, node_idx in enumerate(range(7, 15)):
                        compare_val = node_idx - 7 if path_indices else node_idx
                        cond = add_node(
                            "valu",
                            ("==", depth3_cond_addr, idxs + base, vec_consts[compare_val]),
                            as_deps(idx_ready[g]) + [selected],
                        )
                        if i >= 8 - vsel_count:
                            selected = add_node(
                                "flow",
                                (
                                    "vselect",
                                    depth3_selected_addr,
                                    depth3_cond_addr,
                                    top_node_vecs[node_idx],
                                    depth3_selected_addr,
                                ),
                                cond,
                            )
                        else:
                            selected = add_node(
                                "valu",
                                (
                                    "multiply_add",
                                    depth3_selected_addr,
                                    depth3_cond_addr,
                                    top_node_vecs[node_idx],
                                    depth3_selected_addr,
                                ),
                                [selected, cond],
                            )
                    if g >= scalar_gather_start_group:
                        cur = [
                            add_node(
                                "alu",
                                ("^", vals + base + lane, vals + base + lane, depth3_selected_addr + lane),
                                as_deps(value_ready[g]) + [selected],
                            )
                            for lane in range(VLEN)
                        ]
                    else:
                        cur = add_node(
                            "valu",
                            ("^", vals + base, vals + base, depth3_selected_addr),
                            as_deps(value_ready[g]) + [selected],
                        )
                    node_tmp_last_users[node_tmp_slot] = (
                        selected if depth3_onehot_select_into_tmp1 else cur
                    )
                else:
                    addr_const = (
                        forest_values_p + (1 << depth) - 1
                        if path_indices
                        else forest_values_p
                    )
                    if g >= scalar_gather_start_group:
                        addr_ready = [
                            add_node(
                                "alu",
                                (
                                    "+",
                                    tmp1 + base + lane,
                                    idxs + base + lane,
                                    scalar_consts[addr_const],
                                ),
                                deps,
                            )
                            for lane in range(VLEN)
                        ]
                    else:
                        vaddr_const = vec_consts[addr_const] if path_indices else vforest
                        addr_ready = add_node(
                            "valu",
                            ("+", tmp1 + base, idxs + base, vaddr_const),
                            deps,
                        )
                    loads = [
                        add_node(
                            "load",
                            (
                                "load_offset",
                                tmp1 + base if gather_into_addr_tmp else node_tmp,
                                tmp1 + base,
                                lane,
                            ),
                            as_deps(
                                addr_ready[lane] if isinstance(addr_ready, list) else addr_ready
                            )
                            + ([] if gather_into_addr_tmp else node_tmp_extra_deps),
                        )
                        for lane in range(VLEN)
                    ]
                    loaded_node = tmp1 + base if gather_into_addr_tmp else node_tmp
                    if g >= scalar_gather_start_group and depth in scatter_vector_xor_depths:
                        cur = add_node(
                            "valu",
                            ("^", vals + base, vals + base, loaded_node),
                            loads + as_deps(value_ready[g]),
                        )
                    elif g >= scalar_gather_start_group:
                        cur = [
                            add_node(
                                "alu",
                                ("^", vals + base + lane, vals + base + lane, loaded_node + lane),
                                [loads[lane]] + as_deps(value_ready[g]),
                            )
                            for lane in range(VLEN)
                        ]
                    else:
                        cur = add_node("valu", ("^", vals + base, vals + base, loaded_node), loads + as_deps(value_ready[g]))
                    if not gather_into_addr_tmp:
                        node_tmp_last_users[node_tmp_slot] = cur
                cur_deps = as_deps(cur)
                gather_done_deps = list(cur_deps)

                set_component("hash")
                def add_hash_stage(hash_stage, stage_deps):
                    op1, val1, op2, op3, val3 = HASH_STAGES[hash_stage]
                    if op1 == "+" and op2 == "+" and op3 == "<<":
                        if g >= scalar_muladd_hash_start_group:
                            next_deps = []
                            for lane in range(VLEN):
                                mul = add_node(
                                    "alu",
                                    (
                                        "*",
                                        tmp1 + base + lane,
                                        vals + base + lane,
                                        scalar_consts[(1 << val3) + 1],
                                    ),
                                    cur_deps,
                                )
                                next_deps.append(
                                    add_node(
                                        "alu",
                                        (
                                            "+",
                                            vals + base + lane,
                                            tmp1 + base + lane,
                                            scalar_consts[val1],
                                        ),
                                        mul,
                                    )
                                )
                            return next_deps
                        cur = add_node(
                            "valu",
                            (
                                "multiply_add",
                                vals + base,
                                vals + base,
                                vec_consts[(1 << val3) + 1],
                                vec_consts[val1],
                            ),
                            stage_deps,
                        )
                        return [cur]
                    if g >= scalar_hash_start_by_depth_stage.get(
                        depth, {}
                    ).get(
                        hash_stage,
                        scalar_hash_start_by_stage.get(
                            hash_stage, scalar_hash_start_group
                        ),
                    ):
                        next_deps = []
                        for lane in range(VLEN):
                            h1 = add_node(
                                "alu",
                                (op1, tmp1 + base + lane, vals + base + lane, scalar_consts[val1]),
                                stage_deps,
                            )
                            if hash_h2_in_vals:
                                # h2 overwrites vals (same-cycle read-before-write
                                # semantics make this safe wrt h1).
                                h2 = add_node(
                                    "alu",
                                    (op3, vals + base + lane, vals + base + lane, scalar_consts[val3]),
                                    stage_deps,
                                )
                                next_deps.append(
                                    add_node(
                                        "alu",
                                        (
                                            op2,
                                            vals + base + lane,
                                            tmp1 + base + lane,
                                            vals + base + lane,
                                        ),
                                        [h1, h2],
                                    )
                                )
                            else:
                                h2 = add_node(
                                    "alu",
                                    (op3, tmp2 + base + lane, vals + base + lane, scalar_consts[val3]),
                                    stage_deps,
                                )
                                next_deps.append(
                                    add_node(
                                        "alu",
                                        (
                                            op2,
                                            vals + base + lane,
                                            tmp1 + base + lane,
                                            tmp2 + base + lane,
                                        ),
                                        [h1, h2],
                                    )
                                )
                        return next_deps
                    # Hybrid: vector h1, h2; scalar combine. Saves 1 valu for
                    # 8 alu per stage instance compared to pure vector.
                    if g >= hybrid_hash_start_by_stage.get(
                        hash_stage, n_vecs
                    ):
                        h1 = add_node(
                            "valu",
                            (op1, tmp1 + base, vals + base, vec_consts[val1]),
                            stage_deps,
                        )
                        h2 = add_node(
                            "valu",
                            (op3, tmp2 + base, vals + base, vec_consts[val3]),
                            stage_deps,
                        )
                        next_deps = []
                        for lane in range(VLEN):
                            next_deps.append(
                                add_node(
                                    "alu",
                                    (
                                        op2,
                                        vals + base + lane,
                                        tmp1 + base + lane,
                                        tmp2 + base + lane,
                                    ),
                                    [h1, h2],
                                )
                            )
                        return next_deps
                    if hash_h2_in_vals:
                        h1 = add_node(
                            "valu",
                            (op1, tmp1 + base, vals + base, vec_consts[val1]),
                            stage_deps,
                        )
                        h2 = add_node(
                            "valu",
                            (op3, vals + base, vals + base, vec_consts[val3]),
                            stage_deps,
                        )
                        cur = add_node(
                            "valu",
                            (op2, vals + base, tmp1 + base, vals + base),
                            [h1, h2],
                        )
                    else:
                        h1 = add_node(
                            "valu",
                            (op1, tmp1 + base, vals + base, vec_consts[val1]),
                            stage_deps,
                        )
                        h2 = add_node(
                            "valu",
                            (op3, tmp2 + base, vals + base, vec_consts[val3]),
                            stage_deps,
                        )
                        cur = add_node(
                            "valu",
                            (op2, vals + base, tmp1 + base, tmp2 + base),
                            [h1, h2],
                        )
                    return [cur]

                for hash_stage in range(len(HASH_STAGES)):
                    cur_deps = add_hash_stage(hash_stage, cur_deps)

                value_ready[g] = cur_deps
                if round_i == rounds - 1:
                    continue

                if depth == forest_height:
                    idx_ready[g] = []
                    continue

                set_component("index")
                # At the root, idx is known to be zero, so idx = 1 + (val & 1).
                if depth == 0:
                    if g >= scalar_index_start_group:
                        next_idx = []
                        for lane in range(VLEN):
                            if path_indices:
                                next_idx.append(
                                    add_node(
                                        "alu",
                                        ("&", idxs + base + lane, vals + base + lane, one),
                                        cur_deps,
                                    )
                                )
                            else:
                                parity = add_node(
                                    "alu",
                                    ("&", tmp1 + base + lane, vals + base + lane, one),
                                    cur_deps,
                                )
                                next_idx.append(
                                    add_node(
                                        "alu",
                                        ("+", idxs + base + lane, tmp1 + base + lane, one),
                                        parity,
                                    )
                                )
                        idx_ready[g] = next_idx
                    else:
                        if path_indices:
                            idx_ready[g] = add_node(
                                "valu",
                                ("&", idxs + base, vals + base, vone),
                                cur_deps,
                            )
                        else:
                            parity = add_node(
                                "valu",
                                ("&", tmp1 + base, vals + base, vone),
                                cur_deps,
                            )
                            idx_ready[g] = add_node(
                                "valu",
                                ("+", idxs + base, tmp1 + base, vone),
                                parity,
                            )
                    continue

                # Other depths use idx = 2 * idx + 1 + (val & 1).
                if g >= scalar_index_start_group:
                    next_idx = []
                    for lane in range(VLEN):
                        parity = add_node(
                            "alu",
                            ("&", tmp1 + base + lane, vals + base + lane, one),
                            cur_deps,
                        )
                        doubled_deps = (
                            gather_done_deps + as_deps(idx_ready[g])
                            if early_doubled
                            else cur_deps + as_deps(idx_ready[g])
                        )
                        doubled = add_node(
                            "alu",
                            (
                                "<<",
                                (idxs if idx_doubled_in_idx and path_indices else tmp2)
                                + base
                                + lane,
                                idxs + base + lane,
                                one,
                            ),
                            doubled_deps,
                        )
                        if path_indices:
                            next_idx.append(
                                add_node(
                                    "alu",
                                    (
                                        "+",
                                        idxs + base + lane,
                                        (idxs if idx_doubled_in_idx else tmp2) + base + lane,
                                        tmp1 + base + lane,
                                    ),
                                    [doubled, parity],
                                )
                            )
                        else:
                            child = add_node(
                                "alu",
                                ("+", tmp1 + base + lane, tmp1 + base + lane, one),
                                parity,
                            )
                            next_idx.append(
                                add_node(
                                    "alu",
                                    (
                                        "+",
                                        idxs + base + lane,
                                        tmp2 + base + lane,
                                        tmp1 + base + lane,
                                    ),
                                    [doubled, child],
                                )
                            )
                    idx_ready[g] = next_idx
                else:
                    parity = add_node(
                        "valu",
                        ("&", tmp1 + base, vals + base, vone),
                        cur_deps,
                    )
                    if early_doubled:
                        # Split mul_add into early doubled (runs during the
                        # load-heavy dip) and a final add.
                        doubled = add_node(
                            "valu",
                            ("<<", tmp2 + base, idxs + base, vone),
                            gather_done_deps + as_deps(idx_ready[g]),
                        )
                        if path_indices:
                            idx_ready[g] = add_node(
                                "valu",
                                ("+", idxs + base, tmp2 + base, tmp1 + base),
                                [doubled, parity],
                            )
                        else:
                            child = add_node(
                                "valu",
                                ("+", tmp1 + base, tmp1 + base, vone),
                                parity,
                            )
                            idx_ready[g] = add_node(
                                "valu",
                                ("+", idxs + base, tmp2 + base, tmp1 + base),
                                [doubled, child],
                            )
                    elif path_indices:
                        idx_ready[g] = add_node(
                            "valu",
                            ("multiply_add", idxs + base, idxs + base, vec_consts[2], tmp1 + base),
                            as_deps(idx_ready[g]) + [parity],
                        )
                    else:
                        child = add_node(
                            "valu",
                            ("+", tmp1 + base, tmp1 + base, vone),
                            parity,
                        )
                        idx_ready[g] = add_node(
                            "valu",
                            ("multiply_add", idxs + base, idxs + base, vec_consts[2], tmp1 + base),
                            as_deps(idx_ready[g]) + [child],
                        )

        set_component("store")
        for g in range(n_vecs):
            add_node("store", ("vstore", value_addrs[g], vals + g * VLEN), value_ready[g])

        schedule_nodes()
        if compress_tmp2_lifetimes:
            self._compress_tmp2_lifetimes(tmp2, n_vecs * VLEN)
        self.schedule_nodes_debug = nodes


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
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
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
