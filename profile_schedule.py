"""Analyze where valu has gaps in the baseline schedule."""
from collections import defaultdict, Counter
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))

from frozen_problem import Tree, Input, build_mem_image, N_CORES, VLEN
from perf_takehome import KernelBuilder


def main():
    random.seed(0)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

    # Identify cycles where valu is idle.
    valu_busy = []
    engine_counts_at_cycle = []
    for cycle, step in enumerate(kb.schedule_trace):
        per_engine = defaultdict(int)
        per_engine_comp = defaultdict(lambda: defaultdict(int))
        for component, engine, slot in step:
            per_engine[engine] += 1
            per_engine_comp[engine][component] += 1
        engine_counts_at_cycle.append(per_engine)
        valu_busy.append(per_engine.get("valu", 0) > 0)

    total = len(valu_busy)
    idle = sum(1 for b in valu_busy if not b)
    print(f"total cycles {total}, valu idle {idle}")

    # Histogram of components per cycle
    print("\nFirst 50 cycles:")
    for c in range(50):
        step = kb.schedule_trace[c]
        per = defaultdict(int)
        per_comp = defaultdict(int)
        for component, engine, slot in step:
            per[engine] += 1
            per_comp[component] += 1
        print(f"  c={c}: " + " ".join(f"{e}:{n}" for e, n in sorted(per.items()))
              + " | comp: " + " ".join(f"{c}:{n}" for c, n in sorted(per_comp.items())))

    print("\nLast 30 cycles:")
    for c in range(max(0, total-30), total):
        step = kb.schedule_trace[c]
        per = defaultdict(int)
        per_comp = defaultdict(int)
        for component, engine, slot in step:
            per[engine] += 1
            per_comp[component] += 1
        print(f"  c={c}: " + " ".join(f"{e}:{n}" for e, n in sorted(per.items()))
              + " | comp: " + " ".join(f"{c}:{n}" for c, n in sorted(per_comp.items())))

    # Find chunks where valu is idle
    print("\nValu idle ranges:")
    in_range = False
    start = 0
    for c, b in enumerate(valu_busy):
        if not b and not in_range:
            in_range = True
            start = c
        elif b and in_range:
            print(f"  c={start}..{c-1} (len={c-start})")
            in_range = False
    if in_range:
        print(f"  c={start}..{total-1} (len={total-start})")


if __name__ == "__main__":
    main()
