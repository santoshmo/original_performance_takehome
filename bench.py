"""
Variant benchmark harness. Runs the kernel builder against the frozen simulator
with a given variant dict, returning (cycles, scratch_used) and (optionally)
detailed engine/component active counts.
"""
from collections import defaultdict
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))

from frozen_problem import (
    Machine,
    build_mem_image,
    reference_kernel2,
    Tree,
    Input,
    N_CORES,
    VLEN,
    SLOT_LIMITS,
)
from perf_takehome import KernelBuilder


def run_variant(variant=None, profile=False, seed=0):
    forest_height, rounds, batch_size = 10, 16, 256
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds, variant=variant)

    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    for ref_mem in reference_kernel2(mem):
        pass

    inp_values_p = ref_mem[6]
    ok = machine.mem[inp_values_p : inp_values_p + len(inp.values)] == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    if not ok:
        return {"cycles": None, "scratch": kb.scratch_ptr, "ok": False}

    res = {"cycles": machine.cycle, "scratch": kb.scratch_ptr, "ok": True, "n_instrs": len(kb.instrs)}

    if profile:
        slot_count = defaultdict(int)
        active = defaultdict(int)
        active_cycles = defaultdict(set)
        component_engine = defaultdict(lambda: defaultdict(int))
        for cycle, step in enumerate(kb.schedule_trace):
            for component, engine, slot in step:
                slot_count[engine] += 1
                active_cycles[engine].add(cycle)
                component_engine[component][engine] += 1
        for engine, cycles in active_cycles.items():
            active[engine] = len(cycles)

        floors = {engine: (slot_count[engine] + SLOT_LIMITS[engine] - 1) // SLOT_LIMITS[engine] for engine in slot_count}
        res["engine"] = {engine: {"slots": slot_count[engine], "floor": floors[engine], "active": active[engine]} for engine in slot_count}
        res["component"] = {c: dict(v) for c, v in component_engine.items()}
    return res


def print_result(name, variant=None, profile=False):
    r = run_variant(variant, profile=profile)
    print(f"{name}: cycles={r['cycles']} scratch={r['scratch']} ok={r['ok']} n_instrs={r.get('n_instrs')}")
    if profile and r["ok"]:
        for engine, v in sorted(r["engine"].items()):
            print(f"  {engine:6s} slots={v['slots']} floor={v['floor']} active={v['active']}")
        for c, d in sorted(r["component"].items()):
            parts = " ".join(f"{e}:{n}" for e, n in sorted(d.items()))
            print(f"  comp {c:8s} {parts}")
    return r


if __name__ == "__main__":
    print_result("baseline", profile=True)
