#!/usr/bin/env python3
"""Profile generated kernels for the performance take-home."""

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
from problem import SLOT_LIMITS  # noqa: E402


def _non_debug_cycles(instrs: list[dict[str, list[tuple]]]) -> int:
    return sum(1 for instr in instrs if any(engine != "debug" for engine in instr))


def _counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(key): value for key, value in counter.items()}


def profile_kernel(kb: KernelBuilder) -> dict[str, Any]:
    """Return a JSON-serializable profile for a built KernelBuilder."""
    total_cycles = _non_debug_cycles(kb.instrs)
    schedule = kb.schedule_trace
    pre_cycles = total_cycles - len(schedule)

    engine_slots: Counter[str] = Counter()
    engine_active: Counter[str] = Counter()
    engine_fill: dict[str, Counter[int]] = defaultdict(Counter)
    component_engine: dict[str, Counter[str]] = defaultdict(Counter)
    component_active: Counter[str] = Counter()
    component_solo: Counter[str] = Counter()
    component_ops: dict[str, Counter[str]] = defaultdict(Counter)
    component_coissue: Counter[str] = Counter()
    engine_coissue: Counter[str] = Counter()

    for cycle in schedule:
        components = {component for component, _engine, _slot in cycle}
        engines = {engine for _component, engine, _slot in cycle}
        component_coissue[",".join(sorted(components))] += 1
        engine_coissue[",".join(sorted(engines))] += 1

        for component in components:
            component_active[component] += 1
        if len(components) == 1:
            component_solo[next(iter(components))] += 1

        per_engine: Counter[str] = Counter()
        for component, engine, slot in cycle:
            engine_slots[engine] += 1
            component_engine[component][engine] += 1
            component_ops[f"{component}:{engine}"][str(slot[0])] += 1
            per_engine[engine] += 1

        for engine, count in per_engine.items():
            engine_active[engine] += 1
            engine_fill[engine][count] += 1

    engine_summary: dict[str, Any] = {}
    for engine, limit in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        slots = engine_slots[engine]
        floor = (slots + limit - 1) // limit if slots else 0
        util = slots / (len(schedule) * limit) if schedule else 0.0
        engine_summary[engine] = {
            "slots": slots,
            "floor": floor,
            "active": engine_active[engine],
            "utilization": util,
            "slack_vs_active": engine_active[engine] - floor,
            "fill_histogram": _counter_to_dict(engine_fill[engine]),
        }

    component_summary: dict[str, Any] = {}
    for component, engines in component_engine.items():
        floors = {
            engine: (count + SLOT_LIMITS[engine] - 1) // SLOT_LIMITS[engine]
            for engine, count in engines.items()
        }
        component_summary[component] = {
            "active": component_active[component],
            "solo": component_solo[component],
            "lower_bound": max(floors.values()) if floors else 0,
            "slots_by_engine": dict(engines),
            "floors_by_engine": floors,
        }

    return {
        "cycles": total_cycles,
        "scheduled_body_cycles": len(schedule),
        "pre_schedule_cycles": pre_cycles,
        "scratch_used": kb.scratch_ptr,
        "scratch_free": 1536 - kb.scratch_ptr,
        "engine_summary": engine_summary,
        "component_summary": component_summary,
        "component_ops": {key: dict(value) for key, value in component_ops.items()},
        "top_component_coissue": component_coissue.most_common(20),
        "top_engine_coissue": engine_coissue.most_common(20),
    }


def build_profile(
    variant: dict[str, Any] | None = None,
    forest_height: int = 10,
    n_nodes: int = 2047,
    batch_size: int = 256,
    rounds: int = 16,
) -> dict[str, Any]:
    kb = KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds, variant=variant)
    return profile_kernel(kb)


def print_human(profile: dict[str, Any]) -> None:
    print(
        "cycles",
        profile["cycles"],
        "body",
        profile["scheduled_body_cycles"],
        "scratch",
        profile["scratch_used"],
        "free",
        profile["scratch_free"],
    )
    print("\nengine summary")
    for engine, data in profile["engine_summary"].items():
        print(
            f"  {engine:5} slots={data['slots']:5} floor={data['floor']:4} "
            f"active={data['active']:4} util={data['utilization']:6.1%}"
        )
    print("\ncomponent summary")
    for component, data in sorted(profile["component_summary"].items()):
        print(
            f"  {component:7} active={data['active']:4} "
            f"lb={data['lower_bound']:4} slots={data['slots_by_engine']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-json", default="{}", help="Variant dictionary as JSON")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args()

    variant = json.loads(args.variant_json)
    profile = build_profile(variant=variant)
    if args.json:
        print(json.dumps(profile, sort_keys=True))
    else:
        print_human(profile)


if __name__ == "__main__":
    main()
