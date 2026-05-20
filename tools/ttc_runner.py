#!/usr/bin/env python3
"""Run local test-time-compute searches for KernelBuilder variants."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import reduce
from itertools import product
import json
from pathlib import Path
import random
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from frozen_problem import (  # noqa: E402
    Input,
    Machine,
    N_CORES,
    Tree,
    build_mem_image,
    reference_kernel2,
)
from perf_takehome import KernelBuilder  # noqa: E402
from tools.profile_kernel import profile_kernel  # noqa: E402


DEFAULT_FOREST_HEIGHT = 10
DEFAULT_ROUNDS = 16
DEFAULT_BATCH_SIZE = 256


@dataclass(frozen=True)
class GridSpec:
    path: tuple[str, ...]
    start: int
    stop: int
    step: int = 1

    @property
    def values(self) -> range:
        return range(self.start, self.stop + 1, self.step)


def parse_int_list(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_grid_spec(raw: str) -> GridSpec:
    # Syntax: key:start:stop[:step], with dotted paths allowed for nested dicts.
    parts = raw.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(f"Invalid grid spec {raw!r}; expected key:start:stop[:step]")
    path = tuple(parts[0].split("."))
    step = int(parts[3]) if len(parts) == 4 else 1
    if step <= 0:
        raise ValueError(f"Grid step must be positive in {raw!r}")
    return GridSpec(path=path, start=int(parts[1]), stop=int(parts[2]), step=step)


def set_path(value: dict[str, Any], path: tuple[str, ...], leaf: Any) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    cursor = result
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"Cannot set nested path {'.'.join(path)} through non-dict")
    cursor[path[-1]] = leaf
    return result


def variant_key(variant: dict[str, Any]) -> str:
    return json.dumps(variant, sort_keys=True, separators=(",", ":"))


def grid_variants(base: dict[str, Any], specs: list[GridSpec]) -> Iterable[dict[str, Any]]:
    if not specs:
        yield base
        return
    for values in product(*(spec.values for spec in specs)):
        variant = base
        for spec, value in zip(specs, values, strict=True):
            variant = set_path(variant, spec.path, value)
        yield variant


def random_variants(
    base: dict[str, Any],
    specs: list[GridSpec],
    count: int,
    seed: int,
) -> Iterable[dict[str, Any]]:
    rng = random.Random(seed)
    for _ in range(count):
        variant = base
        for spec in specs:
            values = list(spec.values)
            variant = set_path(variant, spec.path, rng.choice(values))
        yield variant


def local_variants(
    center: dict[str, Any],
    specs: list[GridSpec],
    radius: int,
) -> Iterable[dict[str, Any]]:
    if not specs:
        yield center
        return
    local_specs = []
    for spec in specs:
        center_value = reduce(lambda obj, key: obj[key], spec.path, center)
        local_specs.append(
            GridSpec(
                path=spec.path,
                start=max(spec.start, int(center_value) - radius),
                stop=min(spec.stop, int(center_value) + radius),
                step=spec.step,
            )
        )
    yield from grid_variants(center, local_specs)


def do_kernel_run(
    variant: dict[str, Any],
    seed: int,
    forest_height: int,
    rounds: int,
    batch_size: int,
) -> tuple[bool, int | None, str | None, KernelBuilder | None]:
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    try:
        kb = KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds, variant=variant)
        machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        for ref_mem in reference_kernel2(mem):
            pass
        inp_values_p = ref_mem[6]
        expected = ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        actual = machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        if actual != expected:
            return False, machine.cycle, "Incorrect output values", kb
        return True, machine.cycle, None, kb
    except Exception as exc:  # noqa: BLE001 - experiments should record failures.
        return False, None, f"{type(exc).__name__}: {exc}", None


def evaluate_variant(
    variant: dict[str, Any],
    seeds: list[int],
    forest_height: int,
    rounds: int,
    batch_size: int,
) -> dict[str, Any]:
    seed_results = []
    first_kb = None
    ok = True
    for seed in seeds:
        passed, cycles, error, kb = do_kernel_run(variant, seed, forest_height, rounds, batch_size)
        seed_results.append(
            {"seed": seed, "ok": passed, "cycles": cycles, "error": error}
        )
        ok = ok and passed
        if first_kb is None and kb is not None:
            first_kb = kb

    profile = profile_kernel(first_kb) if first_kb is not None else None
    cycles = [result["cycles"] for result in seed_results if result["cycles"] is not None]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "ok": ok,
        "cycles": max(cycles) if cycles else None,
        "seed_results": seed_results,
        "profile": profile,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def print_top(results: list[dict[str, Any]], top: int) -> None:
    def sort_key(row: dict[str, Any]) -> tuple[int, int]:
        return (0 if row["ok"] else 1, row["cycles"] if row["cycles"] is not None else 10**12)

    for index, row in enumerate(sorted(results, key=sort_key)[:top], start=1):
        status = "ok" if row["ok"] else "fail"
        print(
            f"{index:3}. {status:4} cycles={row['cycles']} "
            f"scratch={row.get('profile', {}).get('scratch_used') if row.get('profile') else None} "
            f"variant={json.dumps(row['variant'], sort_keys=True)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-json", default="{}", help="Base variant dictionary as JSON")
    parser.add_argument("--grid", action="append", default=[], help="Grid spec key:start:stop[:step]")
    parser.add_argument("--local-center-json", help="Center variant for local search")
    parser.add_argument("--local-radius", type=int, default=1, help="Radius for local search grids")
    parser.add_argument("--random", type=int, default=0, help="Number of random candidates")
    parser.add_argument("--random-seed", type=int, default=0, help="Random search seed")
    parser.add_argument("--seeds", default="123", help="Comma-separated correctness seeds")
    parser.add_argument("--top", type=int, default=20, help="How many top results to print")
    parser.add_argument("--output", default="ttc-results/results.jsonl", help="JSONL output path")
    parser.add_argument("--forest-height", type=int, default=DEFAULT_FOREST_HEIGHT)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    base = json.loads(args.variant_json)
    specs = [parse_grid_spec(raw) for raw in args.grid]
    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required")

    candidates: list[dict[str, Any]] = []
    if args.local_center_json:
        candidates.extend(
            local_variants(json.loads(args.local_center_json), specs, args.local_radius)
        )
    elif args.random:
        candidates.extend(random_variants(base, specs, args.random, args.random_seed))
    else:
        candidates.extend(grid_variants(base, specs))

    seen: set[str] = set()
    unique_candidates = []
    for candidate in candidates:
        key = variant_key(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    results = [
        evaluate_variant(candidate, seeds, args.forest_height, args.rounds, args.batch_size)
        for candidate in unique_candidates
    ]
    write_jsonl(REPO_ROOT / args.output, results)
    print_top(results, args.top)


if __name__ == "__main__":
    main()
