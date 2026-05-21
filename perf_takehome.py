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
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Blog-layout SIMD/VLIW kernel built from the baseline repo:
        path-offset state, one hash temp, dedicated node/select temps, and
        cached depth-3 tree selection.
        """
        assert batch_size % VLEN == 0
        n_vecs = batch_size // VLEN
        forest_values_p = 7
        inp_values_p = forest_values_p + n_nodes + batch_size

        nodes = []
        ready_by_engine = {engine: [] for engine in SLOT_LIMITS}
        component = ["setup"]

        def set_component(name):
            component[0] = name

        def deps_list(deps):
            if deps is None:
                return []
            if isinstance(deps, int):
                return [deps]
            return list(deps)

        def add_node(engine, slot, deps=None):
            deps = deps_list(deps)
            node_id = len(nodes)
            nodes.append({
                "engine": engine,
                "slot": slot,
                "component": component[0],
                "deps_left": len(deps),
                "succs": [],
            })
            for dep in deps:
                nodes[dep]["succs"].append(node_id)
            if not deps:
                ready_by_engine[engine].append(node_id)
            return node_id

        def emit_packed(engine, slots):
            for i in range(0, len(slots), SLOT_LIMITS[engine]):
                self.instrs.append({engine: slots[i : i + SLOT_LIMITS[engine]]})

        def schedule_nodes():
            remaining = len(nodes)
            while remaining:
                instr = {}
                chosen = []
                for engine in ("load", "valu", "alu", "store", "flow"):
                    ready = ready_by_engine[engine]
                    if ready:
                        take = ready[: SLOT_LIMITS[engine]]
                        del ready[: SLOT_LIMITS[engine]]
                        instr[engine] = [nodes[n]["slot"] for n in take]
                        chosen.extend(take)
                assert chosen, "scheduler stalled"
                self.instrs.append(instr)
                self.schedule_trace.append([
                    (nodes[n]["component"], nodes[n]["engine"], nodes[n]["slot"])
                    for n in chosen
                ])
                remaining -= len(chosen)
                for node_id in chosen:
                    for succ in nodes[node_id]["succs"]:
                        nodes[succ]["deps_left"] -= 1
                        if nodes[succ]["deps_left"] == 0:
                            ready_by_engine[nodes[succ]["engine"]].append(succ)

        const_slots = []
        scalar_consts = {}

        def scalar_const(val, name=None):
            if val not in scalar_consts:
                addr = self.alloc_scratch(name)
                scalar_consts[val] = addr
                const_slots.append(("const", addr, val))
            return scalar_consts[val]

        vec_consts = {}

        def vector_const(val, name=None):
            if val not in vec_consts:
                vec_consts[val] = self.alloc_scratch(name or f"vconst_{val}", VLEN)
            scalar_const(val)
            return vec_consts[val]

        one = scalar_const(1, "one")
        vzero = self.alloc_scratch("vzero", VLEN)
        vone = vector_const(1, "vone")
        vtwo = vector_const(2, "vtwo")
        for depth in range(3, forest_height + 1):
            scalar_const(forest_values_p + (1 << depth) - 1)
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if op1 == "+" and op2 == "+" and op3 == "<<":
                vector_const(val1)
                vector_const((1 << val3) + 1)
            else:
                scalar_const(val1)
                scalar_const(val3)

        value_addrs = [
            scalar_const(inp_values_p + g * VLEN, f"value_addr_{g}") for g in range(n_vecs)
        ]

        top_node_scalars = self.alloc_scratch("top_node_scalars", VLEN)
        top_node_vecs = {}
        for node_idx in range(15):
            node_vec = self.alloc_scratch(f"vtop_node_{node_idx}", VLEN)
            top_node_vecs[node_idx] = node_vec
        top_setup_instrs = [{"load": [("vload", top_node_scalars, scalar_const(forest_values_p))]}]
        first_broadcasts = [
            ("vbroadcast", top_node_vecs[node_idx], top_node_scalars + node_idx)
            for node_idx in range(7)
        ]
        for i in range(0, len(first_broadcasts), SLOT_LIMITS["valu"]):
            top_setup_instrs.append({"valu": first_broadcasts[i : i + SLOT_LIMITS["valu"]]})
        top_setup_instrs.append({"load": [("vload", top_node_scalars, scalar_const(forest_values_p + 7))]})
        second_broadcasts = [
            ("vbroadcast", top_node_vecs[node_idx], top_node_scalars + node_idx - 7)
            for node_idx in range(7, 15)
        ]
        for i in range(0, len(second_broadcasts), SLOT_LIMITS["valu"]):
            top_setup_instrs.append({"valu": second_broadcasts[i : i + SLOT_LIMITS["valu"]]})

        vals = self.alloc_scratch("vals", batch_size)
        path = self.alloc_scratch("path", batch_size)
        hash_tmp = self.alloc_scratch("hash_tmp", batch_size)
        node_tmp = self.alloc_scratch("node_tmp", batch_size)
        sel_tmp = self.alloc_scratch("sel_tmp", batch_size)

        emit_packed("load", const_slots)
        emit_packed(
            "valu",
            [
                ("vbroadcast", addr, scalar_consts[val])
                for val, addr in vec_consts.items()
                if val in scalar_consts
            ],
        )
        self.instrs.extend(top_setup_instrs)

        value_ready = []
        path_ready = [[] for _ in range(n_vecs)]
        set_component("setup")
        for g in range(n_vecs):
            value_ready.append(add_node("load", ("vload", vals + g * VLEN, value_addrs[g])))
        d3_bit_tmp = top_node_scalars
        d3_bit_tmp_last_user = None

        scalar_hash_start_by_stage = {1: 0, 3: 0, 5: 0}
        scalar_gather_start = 24
        scalar_index_start = 26

        for round_i in range(rounds):
            depth = round_i % (forest_height + 1)
            for g in range(n_vecs):
                base = g * VLEN
                deps = deps_list(value_ready[g]) + deps_list(path_ready[g])

                set_component("gather")
                if depth == 0:
                    if g >= scalar_gather_start:
                        gather_done = [
                            add_node("alu", ("^", vals + base + lane, vals + base + lane, top_node_vecs[0] + lane), value_ready[g])
                            for lane in range(VLEN)
                        ]
                    else:
                        gather_done = add_node("valu", ("^", vals + base, vals + base, top_node_vecs[0]), value_ready[g])
                elif depth == 1:
                    cond = add_node("valu", ("==", hash_tmp + base, path + base, vone), deps)
                    selected = add_node("flow", ("vselect", node_tmp + base, hash_tmp + base, top_node_vecs[2], top_node_vecs[1]), cond)
                    gather_done = [
                        add_node("alu", ("^", vals + base + lane, vals + base + lane, node_tmp + base + lane), deps_list(value_ready[g]) + [selected])
                        for lane in range(VLEN)
                    ] if g >= scalar_gather_start else add_node("valu", ("^", vals + base, vals + base, node_tmp + base), deps_list(value_ready[g]) + [selected])
                elif depth == 2:
                    bit0 = add_node("valu", ("&", hash_tmp + base, path + base, vone), deps)
                    p0 = add_node("flow", ("vselect", node_tmp + base, hash_tmp + base, top_node_vecs[4], top_node_vecs[3]), bit0)
                    p1 = add_node("flow", ("vselect", sel_tmp + base, hash_tmp + base, top_node_vecs[6], top_node_vecs[5]), bit0)
                    shifted = add_node("valu", (">>", hash_tmp + base, path + base, vone), [p0, p1])
                    bit1 = add_node("valu", ("&", hash_tmp + base, hash_tmp + base, vone), shifted)
                    selected = add_node("flow", ("vselect", node_tmp + base, hash_tmp + base, sel_tmp + base, node_tmp + base), [bit1, p0, p1])
                    gather_done = [
                        add_node("alu", ("^", vals + base + lane, vals + base + lane, node_tmp + base + lane), deps_list(value_ready[g]) + [selected])
                        for lane in range(VLEN)
                    ] if g >= scalar_gather_start else add_node("valu", ("^", vals + base, vals + base, node_tmp + base), deps_list(value_ready[g]) + [selected])
                elif depth == 3:
                    bit0 = add_node("valu", ("&", hash_tmp + base, path + base, vone), deps)
                    p0 = add_node("flow", ("vselect", node_tmp + base, hash_tmp + base, top_node_vecs[8], top_node_vecs[7]), bit0)
                    p1 = add_node("flow", ("vselect", sel_tmp + base, hash_tmp + base, top_node_vecs[10], top_node_vecs[9]), bit0)
                    shifted1 = add_node("valu", (">>", hash_tmp + base, path + base, vone), [p0, p1])
                    bit1 = add_node("valu", ("&", hash_tmp + base, hash_tmp + base, vone), shifted1)
                    q0 = add_node("flow", ("vselect", node_tmp + base, hash_tmp + base, sel_tmp + base, node_tmp + base), [bit1, p0, p1])
                    bit0b = add_node("valu", ("&", hash_tmp + base, path + base, vone), q0)
                    p2 = add_node("flow", ("vselect", sel_tmp + base, hash_tmp + base, top_node_vecs[12], top_node_vecs[11]), bit0b)
                    p3 = add_node("flow", ("vselect", hash_tmp + base, hash_tmp + base, top_node_vecs[14], top_node_vecs[13]), [bit0b, p2])
                    shifted1b = add_node("valu", (">>", hash_tmp + base, path + base, vone), [p2, p3])
                    bit1b = add_node("valu", ("&", hash_tmp + base, hash_tmp + base, vone), shifted1b)
                    q1 = add_node("flow", ("vselect", sel_tmp + base, hash_tmp + base, hash_tmp + base, sel_tmp + base), [bit1b, p2, p3])
                    bit2 = add_node("valu", (">>", hash_tmp + base, path + base, vtwo), [q0, q1])
                    selected = add_node("flow", ("vselect", node_tmp + base, hash_tmp + base, sel_tmp + base, node_tmp + base), [bit2, q0, q1])
                    gather_done = [
                        add_node("alu", ("^", vals + base + lane, vals + base + lane, node_tmp + base + lane), deps_list(value_ready[g]) + [selected])
                        for lane in range(VLEN)
                    ] if g >= scalar_gather_start else add_node("valu", ("^", vals + base, vals + base, node_tmp + base), deps_list(value_ready[g]) + [selected])
                else:
                    addr_const = scalar_consts[forest_values_p + (1 << depth) - 1]
                    addr = [
                        add_node("alu", ("+", hash_tmp + base + lane, path + base + lane, addr_const), deps)
                        for lane in range(VLEN)
                    ]
                    loads = [
                        add_node("load", ("load_offset", node_tmp + base, hash_tmp + base, lane), addr[lane])
                        for lane in range(VLEN)
                    ]
                    gather_done = [
                        add_node("alu", ("^", vals + base + lane, vals + base + lane, node_tmp + base + lane), [loads[lane]] + deps_list(value_ready[g]))
                        for lane in range(VLEN)
                    ] if g >= scalar_gather_start else add_node("valu", ("^", vals + base, vals + base, node_tmp + base), loads + deps_list(value_ready[g]))

                cur_deps = deps_list(gather_done)
                set_component("hash")
                for stage_i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    if op1 == "+" and op2 == "+" and op3 == "<<":
                        cur = add_node("valu", ("multiply_add", vals + base, vals + base, vec_consts[(1 << val3) + 1], vec_consts[val1]), cur_deps)
                        cur_deps = [cur]
                    elif g >= scalar_hash_start_by_stage.get(stage_i, n_vecs):
                        next_deps = []
                        for lane in range(VLEN):
                            h1 = add_node("alu", (op1, hash_tmp + base + lane, vals + base + lane, scalar_consts[val1]), cur_deps)
                            h2 = add_node("alu", (op3, vals + base + lane, vals + base + lane, scalar_consts[val3]), cur_deps)
                            next_deps.append(add_node("alu", (op2, vals + base + lane, hash_tmp + base + lane, vals + base + lane), [h1, h2]))
                        cur_deps = next_deps
                    else:
                        h1 = add_node("valu", (op1, hash_tmp + base, vals + base, vec_consts[val1]), cur_deps)
                        h2 = add_node("valu", (op3, vals + base, vals + base, vec_consts[val3]), cur_deps)
                        cur = add_node("valu", (op2, vals + base, hash_tmp + base, vals + base), [h1, h2])
                        cur_deps = [cur]
                value_ready[g] = cur_deps

                if round_i == rounds - 1:
                    continue
                if depth == forest_height:
                    path_ready[g] = []
                    continue
                set_component("index")
                if depth == 0:
                    if g >= scalar_index_start:
                        path_ready[g] = [
                            add_node("alu", ("&", path + base + lane, vals + base + lane, one), cur_deps)
                            for lane in range(VLEN)
                        ]
                    else:
                        path_ready[g] = add_node("valu", ("&", path + base, vals + base, vone), cur_deps)
                else:
                    if g >= scalar_index_start:
                        next_path = []
                        for lane in range(VLEN):
                            parity = add_node("alu", ("&", hash_tmp + base + lane, vals + base + lane, one), cur_deps)
                            doubled = add_node("alu", ("<<", path + base + lane, path + base + lane, one), deps_list(path_ready[g]) + [parity])
                            next_path.append(add_node("alu", ("+", path + base + lane, path + base + lane, hash_tmp + base + lane), [doubled, parity]))
                        path_ready[g] = next_path
                    else:
                        parity = add_node("valu", ("&", hash_tmp + base, vals + base, vone), cur_deps)
                        path_ready[g] = add_node("valu", ("multiply_add", path + base, path + base, vtwo, hash_tmp + base), deps_list(path_ready[g]) + [parity])

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
