"""Profile per-cycle valu slot usage."""
from collections import defaultdict
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

    valu_per_cycle = []
    for step in kb.schedule_trace:
        c = 0
        for component, engine, slot in step:
            if engine == "valu":
                c += 1
        valu_per_cycle.append(c)

    # Histogram of valu count per cycle
    hist = defaultdict(int)
    for v in valu_per_cycle:
        hist[v] += 1
    print("Histogram of valu slots per cycle:")
    for k in sorted(hist):
        print(f"  {k} slots: {hist[k]} cycles")
    print(f"  total cycles: {len(valu_per_cycle)}")
    print(f"  total valu slots used: {sum(valu_per_cycle)}")
    print(f"  ideal (6/cycle): {6 * len(valu_per_cycle)}")
    print(f"  unused valu slot-cycles: {6 * len(valu_per_cycle) - sum(valu_per_cycle)}")

    # Bin by chunks
    bins = 12
    chunk = (len(valu_per_cycle) + bins - 1) // bins
    print("\nValu usage in cycle bins:")
    for i in range(bins):
        s = i * chunk
        e = min((i + 1) * chunk, len(valu_per_cycle))
        if s >= e:
            break
        slots = sum(valu_per_cycle[s:e])
        active = sum(1 for v in valu_per_cycle[s:e] if v > 0)
        print(f"  c={s:5d}..{e-1:5d}: {slots} valu slots, {active}/{e-s} active cycles, avg {slots/(e-s):.2f}/cycle")


if __name__ == "__main__":
    main()
