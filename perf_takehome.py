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
        n_vecs = batch_size // VLEN
        forest_values_p = 7
        inp_values_p = forest_values_p + n_nodes + batch_size
        cache_depth3 = variant.get("cache_depth3_onehot", True)
        cache_depth3_start_group = variant.get("cache_depth3_start_group", 14)

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
        vforest = vector_const(forest_values_p, "vforest_values_p")
        top_cache_nodes = 15 if cache_depth3 else 7
        vector_const(2)
        for node_idx in range(4, top_cache_nodes):
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
        for node_idx in range(top_cache_nodes):
            node_addr = scalar_const(forest_values_p + node_idx, f"top_node_addr_{node_idx}")
            node_scalar = self.alloc_scratch(f"top_node_{node_idx}")
            node_vec = self.alloc_scratch(f"vtop_node_{node_idx}", VLEN)
            top_node_vecs[node_idx] = node_vec
            top_node_load_slots.append(("load", node_scalar, node_addr))
            top_node_broadcast_slots.append(("vbroadcast", node_vec, node_scalar))

        vals = alloc_vecs("vals")
        idxs = alloc_vecs("idxs")
        tmp1 = alloc_vecs("tmp1")
        tmp2 = alloc_vecs("tmp2")

        emit_packed("load", const_slots)
        emit_packed("load", top_node_load_slots)
        emit_packed(
            "valu",
            [("vbroadcast", addr, scalar_consts[val]) for val, addr in vec_consts.items()]
            + top_node_broadcast_slots,
        )
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
                "deps_left": len(deps),
                "succs": [],
            }
            nodes.append(node)
            for dep in deps:
                nodes[dep]["succs"].append(node_id)
            if not deps:
                ready_by_engine[engine].append(node_id)
            return node_id

        def schedule_nodes():
            remaining = len(nodes)
            while remaining:
                instr = {}
                chosen = []
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
        scalar_gather_start_group = variant.get(
            "scalar_gather_start_group", 28
        )
        scalar_index_start_group = variant.get(
            "scalar_index_start_group", 18
        )
        scalar_hash_start_by_stage = variant.get(
            "scalar_hash_start_by_stage", {1: 28, 3: 25, 5: 25}
        )
        scalar_muladd_hash_start_group = variant.get(
            "scalar_muladd_hash_start_group", n_vecs
        )

        for round_i in range(rounds):
            depth = round_i % (forest_height + 1)
            for g in range(n_vecs):
                base = g * VLEN
                deps = as_deps(value_ready[g]) + as_deps(idx_ready[g])

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
                        ("==", tmp1 + base, idxs + base, vec_consts[2]),
                        deps,
                    )
                    selected = add_node(
                        "flow",
                        ("vselect", tmp2 + base, tmp1 + base, top_node_vecs[2], top_node_vecs[1]),
                        cond,
                    )
                    if g >= scalar_gather_start_group:
                        cur = [
                            add_node(
                                "alu",
                                ("^", vals + base + lane, vals + base + lane, tmp2 + base + lane),
                                as_deps(value_ready[g]) + [selected],
                            )
                            for lane in range(VLEN)
                        ]
                    else:
                        cur = add_node(
                            "valu",
                            ("^", vals + base, vals + base, tmp2 + base),
                            as_deps(value_ready[g]) + [selected],
                        )
                elif depth == 2:
                    set_component("select")
                    selected = add_node(
                        "valu",
                        ("+", tmp2 + base, top_node_vecs[3], vzero),
                        deps,
                    )
                    for node_idx in range(4, 7):
                        cond = add_node(
                            "valu",
                            ("==", tmp1 + base, idxs + base, vec_consts[node_idx]),
                            as_deps(idx_ready[g]) + [selected],
                        )
                        selected = add_node(
                            "flow",
                            (
                                "vselect",
                                tmp2 + base,
                                tmp1 + base,
                                top_node_vecs[node_idx],
                                tmp2 + base,
                            ),
                            cond,
                        )
                    if g >= scalar_gather_start_group:
                        cur = [
                            add_node(
                                "alu",
                                ("^", vals + base + lane, vals + base + lane, tmp2 + base + lane),
                                as_deps(value_ready[g]) + [selected],
                            )
                            for lane in range(VLEN)
                        ]
                    else:
                        cur = add_node(
                            "valu",
                            ("^", vals + base, vals + base, tmp2 + base),
                            as_deps(value_ready[g]) + [selected],
                        )
                elif depth == 3 and cache_depth3 and g >= cache_depth3_start_group:
                    set_component("select")
                    selected = add_node(
                        "valu",
                        ("+", tmp2 + base, vzero, vzero),
                        deps,
                    )
                    for node_idx in range(7, 15):
                        cond = add_node(
                            "valu",
                            ("==", tmp1 + base, idxs + base, vec_consts[node_idx]),
                            as_deps(idx_ready[g]) + [selected],
                        )
                        selected = add_node(
                            "valu",
                            (
                                "multiply_add",
                                tmp2 + base,
                                tmp1 + base,
                                top_node_vecs[node_idx],
                                tmp2 + base,
                            ),
                            [selected, cond],
                        )
                    if g >= scalar_gather_start_group:
                        cur = [
                            add_node(
                                "alu",
                                ("^", vals + base + lane, vals + base + lane, tmp2 + base + lane),
                                as_deps(value_ready[g]) + [selected],
                            )
                            for lane in range(VLEN)
                        ]
                    else:
                        cur = add_node(
                            "valu",
                            ("^", vals + base, vals + base, tmp2 + base),
                            as_deps(value_ready[g]) + [selected],
                        )
                else:
                    if g >= scalar_gather_start_group:
                        addr_ready = [
                            add_node(
                                "alu",
                                (
                                    "+",
                                    tmp1 + base + lane,
                                    idxs + base + lane,
                                    scalar_consts[forest_values_p],
                                ),
                                deps,
                            )
                            for lane in range(VLEN)
                        ]
                    else:
                        addr_ready = add_node(
                            "valu",
                            ("+", tmp1 + base, idxs + base, vforest),
                            deps,
                        )
                    loads = [
                        add_node(
                            "load",
                            ("load_offset", tmp2 + base, tmp1 + base, lane),
                            addr_ready[lane] if isinstance(addr_ready, list) else addr_ready,
                        )
                        for lane in range(VLEN)
                    ]
                    if g >= scalar_gather_start_group:
                        cur = [
                            add_node(
                                "alu",
                                ("^", vals + base + lane, vals + base + lane, tmp2 + base + lane),
                                [loads[lane]] + as_deps(value_ready[g]),
                            )
                            for lane in range(VLEN)
                        ]
                    else:
                        cur = add_node("valu", ("^", vals + base, vals + base, tmp2 + base), loads + as_deps(value_ready[g]))
                cur_deps = as_deps(cur)

                set_component("hash")
                for hash_stage, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
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
                            cur_deps = next_deps
                        else:
                            cur = add_node(
                                "valu",
                                (
                                    "multiply_add",
                                    vals + base,
                                    vals + base,
                                    vec_consts[(1 << val3) + 1],
                                    vec_consts[val1],
                                ),
                                cur_deps,
                            )
                            cur_deps = [cur]
                    elif g >= scalar_hash_start_by_stage.get(
                        hash_stage, scalar_hash_start_group
                    ):
                        next_deps = []
                        for lane in range(VLEN):
                            h1 = add_node(
                                "alu",
                                (op1, tmp1 + base + lane, vals + base + lane, scalar_consts[val1]),
                                cur_deps,
                            )
                            h2 = add_node(
                                "alu",
                                (op3, tmp2 + base + lane, vals + base + lane, scalar_consts[val3]),
                                cur_deps,
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
                        cur_deps = next_deps
                    else:
                        h1 = add_node(
                            "valu",
                            (op1, tmp1 + base, vals + base, vec_consts[val1]),
                            cur_deps,
                        )
                        h2 = add_node(
                            "valu",
                            (op3, tmp2 + base, vals + base, vec_consts[val3]),
                            cur_deps,
                        )
                        cur = add_node(
                            "valu",
                            (op2, vals + base, tmp1 + base, tmp2 + base),
                            [h1, h2],
                        )
                        cur_deps = [cur]

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
                        doubled = add_node(
                            "alu",
                            ("<<", tmp2 + base + lane, idxs + base + lane, one),
                            cur_deps + as_deps(idx_ready[g]),
                        )
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
