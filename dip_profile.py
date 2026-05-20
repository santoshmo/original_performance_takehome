"""Profile what's running in the valu-underutilized middle of the schedule."""
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

    # For each 60-cycle slice, show engine breakdown by component
    for s in range(0, len(kb.schedule_trace), 60):
        e = min(s + 60, len(kb.schedule_trace))
        per_engine_comp = defaultdict(lambda: defaultdict(int))
        for c in range(s, e):
            for component, engine, slot in kb.schedule_trace[c]:
                per_engine_comp[engine][component] += 1
        parts = []
        for engine in ("load", "valu", "alu", "store", "flow"):
            d = per_engine_comp[engine]
            if d:
                parts.append(f"{engine}=" + ",".join(f"{c}:{n}" for c, n in d.items()))
        print(f"c={s:4d}..{e-1:4d}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
