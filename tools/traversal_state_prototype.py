#!/usr/bin/env python3
"""Self-contained prototype for alternate traversal/index state.

This does not generate a submission kernel. It validates state representations
against the reference heap-index semantics and estimates the vector/scalar
operation pressure each representation would imply in the SIMD kernel.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import argparse
import json
import random


FOREST_HEIGHT = 10
ROUNDS = 16
BATCH_SIZE = 256
VLEN = 8
BLOCKS = BATCH_SIZE // VLEN


@dataclass
class Cost:
    valu: Counter[str]
    alu: Counter[str]
    flow: Counter[str]
    load: Counter[str]

    @classmethod
    def empty(cls) -> "Cost":
        return cls(Counter(), Counter(), Counter(), Counter())

    def add(self, engine: str, op: str, count: int = 1) -> None:
        getattr(self, engine)[op] += count

    def totals(self) -> dict[str, int]:
        return {
            "valu": sum(self.valu.values()),
            "alu": sum(self.alu.values()),
            "flow": sum(self.flow.values()),
            "load": sum(self.load.values()),
        }

    def detail(self) -> dict[str, dict[str, int]]:
        return {
            "valu": dict(self.valu),
            "alu": dict(self.alu),
            "flow": dict(self.flow),
            "load": dict(self.load),
        }


def branch_bit_from_hash(value: int) -> int:
    # Reference chooses left child index +1 for even values and right child +2
    # for odd values. As a level-local offset bit, left is 0 and right is 1.
    return value & 1


def reference_indices(branch_bits: list[list[int]]) -> list[list[int]]:
    idx = [0] * BATCH_SIZE
    trace = []
    n_nodes = 2 ** (FOREST_HEIGHT + 1) - 1
    for round_i in range(ROUNDS):
        trace.append(idx[:])
        for lane in range(BATCH_SIZE):
            bit = branch_bits[round_i][lane]
            idx[lane] = 2 * idx[lane] + 1 + bit
            if idx[lane] >= n_nodes:
                idx[lane] = 0
    return trace


def offset_indices(branch_bits: list[list[int]]) -> list[list[int]]:
    """Carry level-local offset instead of absolute heap index."""
    offset = [0] * BATCH_SIZE
    trace = []
    for round_i in range(ROUNDS):
        level = round_i % (FOREST_HEIGHT + 1)
        idx_base = (1 << level) - 1
        trace.append([idx_base + off for off in offset])
        next_level = (round_i + 1) % (FOREST_HEIGHT + 1)
        if next_level == 0:
            offset = [0] * BATCH_SIZE
        else:
            offset = [
                (off << 1) + branch_bits[round_i][lane]
                for lane, off in enumerate(offset)
            ]
    return trace


def bitstream_indices(branch_bits: list[list[int]]) -> list[list[int]]:
    """Carry rolling path bits and materialize only the current level slice."""
    bits = [0] * BATCH_SIZE
    trace = []
    for round_i in range(ROUNDS):
        level = round_i % (FOREST_HEIGHT + 1)
        mask = (1 << level) - 1
        idx_base = (1 << level) - 1
        trace.append([idx_base + (path & mask) for path in bits])
        next_level = (round_i + 1) % (FOREST_HEIGHT + 1)
        if next_level == 0:
            bits = [0] * BATCH_SIZE
        else:
            bits = [
                ((path << 1) | branch_bits[round_i][lane]) & ((1 << next_level) - 1)
                for lane, path in enumerate(bits)
            ]
    return trace


def estimate_current_offset_cost() -> Cost:
    """Approximate traversal/index cost of the current SIMD offset-state kernel.

    This excludes hash, XOR, and final stores. It focuses on state-derived work:
    shallow selects, deep gather addresses/loads, and non-final offset updates.
    Counts are per vector block across all rounds.
    """
    cost = Cost.empty()
    for _block in range(BLOCKS):
        for round_i in range(ROUNDS):
            level = round_i % (FOREST_HEIGHT + 1)
            final_round = round_i == ROUNDS - 1

            if level == 1:
                cost.add("valu", "level1_bit")
                cost.add("flow", "level1_vselect")
            elif level == 2:
                cost.add("valu", "level2_bits", 2)
                cost.add("flow", "level2_vselects", 3)
            elif level == 3:
                cost.add("valu", "level3_bits", 3)
                cost.add("flow", "level3_vselects", 7)
            elif level >= 4:
                cost.add("alu", "deep_gather_addr_lanes", VLEN)
                cost.add("load", "deep_gather_load_lanes", VLEN)

            if not final_round:
                if level == FOREST_HEIGHT:
                    cost.add("valu", "offset_reset")
                else:
                    cost.add("alu", "parity_lanes", VLEN)
                    cost.add("valu", "offset_update")
    return cost


def estimate_bitstream_cost(materialize_shallow_bits: bool) -> Cost:
    """Estimate a path-bitstream design.

    If materialize_shallow_bits is false, the prototype assumes shallow levels
    consume carried selector bits directly, so level 1..3 selection no longer
    pays vector mask ops. This is the optimistic form worth testing in a real
    kernel. The update cost remains one vector shift/or-like op per non-final
    round, plus scalar lane parity extraction from the hash output.
    """
    cost = Cost.empty()
    for _block in range(BLOCKS):
        for round_i in range(ROUNDS):
            level = round_i % (FOREST_HEIGHT + 1)
            final_round = round_i == ROUNDS - 1

            if level == 1:
                if materialize_shallow_bits:
                    cost.add("valu", "level1_bit")
                cost.add("flow", "level1_vselect")
            elif level == 2:
                if materialize_shallow_bits:
                    cost.add("valu", "level2_bits", 2)
                cost.add("flow", "level2_vselects", 3)
            elif level == 3:
                if materialize_shallow_bits:
                    cost.add("valu", "level3_bits", 3)
                cost.add("flow", "level3_vselects", 7)
            elif level >= 4:
                cost.add("alu", "deep_gather_addr_lanes", VLEN)
                cost.add("load", "deep_gather_load_lanes", VLEN)

            if not final_round:
                if level == FOREST_HEIGHT:
                    cost.add("valu", "path_reset")
                else:
                    cost.add("alu", "parity_lanes", VLEN)
                    cost.add("valu", "path_update")
    return cost


def estimate_deferred_materialization_cost(materialize_levels: set[int]) -> Cost:
    """Estimate only materializing vector offsets at requested levels.

    This models a design that carries lane-local branch bits through scalar
    scratch and pays vector materialization only when a deep gather needs a
    packed offset vector. It is optimistic because it ignores the scratch and
    scheduling cost of carrying those scalar bits.
    """
    cost = Cost.empty()
    for _block in range(BLOCKS):
        for round_i in range(ROUNDS):
            level = round_i % (FOREST_HEIGHT + 1)
            final_round = round_i == ROUNDS - 1

            if level in (1, 2, 3):
                cost.add("flow", f"level{level}_vselects", {1: 1, 2: 3, 3: 7}[level])
            elif level >= 4:
                if level in materialize_levels:
                    cost.add("valu", "materialize_offset")
                cost.add("alu", "deep_gather_addr_lanes", VLEN)
                cost.add("load", "deep_gather_load_lanes", VLEN)

            if not final_round:
                if level == FOREST_HEIGHT:
                    cost.add("alu", "scalar_path_reset_lanes", VLEN)
                else:
                    cost.add("alu", "scalar_path_update_lanes", VLEN * 2)
    return cost


def random_branch_bits(seed: int) -> list[list[int]]:
    random.seed(seed)
    return [
        [random.randint(0, 1) for _ in range(BATCH_SIZE)]
        for _ in range(ROUNDS)
    ]


def validate(seed: int) -> None:
    bits = random_branch_bits(seed)
    ref = reference_indices(bits)
    offset = offset_indices(bits)
    bitstream = bitstream_indices(bits)
    assert offset == ref, "offset representation diverged"
    assert bitstream == ref, "bitstream representation diverged"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()

    for seed in range(args.seeds):
        validate(seed)

    rows = {
        "current_offset_estimate": estimate_current_offset_cost(),
        "bitstream_materialized_shallow": estimate_bitstream_cost(
            materialize_shallow_bits=True
        ),
        "bitstream_carried_shallow": estimate_bitstream_cost(
            materialize_shallow_bits=False
        ),
        "deferred_materialize_deep_only": estimate_deferred_materialization_cost(
            set(range(4, FOREST_HEIGHT + 1))
        ),
    }

    for name, cost in rows.items():
        print(
            json.dumps(
                {
                    "name": name,
                    "totals": cost.totals(),
                    "detail": cost.detail(),
                },
                sort_keys=True,
            )
        )
    print(json.dumps({"validated_seeds": args.seeds}, sort_keys=True))


if __name__ == "__main__":
    main()
