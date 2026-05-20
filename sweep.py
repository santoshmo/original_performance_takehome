"""Sweep tuning parameters around the baseline."""
from bench import run_variant


def main():
    base = {}
    r = run_variant(base, profile=True)
    print(f"baseline cycles={r['cycles']} scratch={r['scratch']} alu_active={r['engine']['alu']['active']} valu_active={r['engine']['valu']['active']}")

    rows = []
    # Sweep up and down on each axis.
    for t1 in (28, 30, 32, 26, 24):
        for t3 in (25, 27, 29, 31, 32):
            for t5 in (25, 27, 29, 31, 32):
                for sg in (20, 18, 22, 16, 24):
                    for si in (19, 17, 21, 23):
                        for cs in (14, 12, 16, 10, 8):
                            v = {
                                "scalar_hash_start_by_stage": {1: t1, 3: t3, 5: t5},
                                "scalar_gather_start_group": sg,
                                "scalar_index_start_group": si,
                                "cache_depth3_start_group": cs,
                            }
                            r = run_variant(v, profile=False)
                            if not r["ok"]:
                                continue
                            rows.append((r["cycles"], t1, t3, t5, sg, si, cs, r["scratch"]))
    rows.sort()
    for row in rows[:30]:
        print(f"cycles={row[0]} t1={row[1]} t3={row[2]} t5={row[3]} sg={row[4]} si={row[5]} cs={row[6]} scratch={row[7]}")


if __name__ == "__main__":
    main()
