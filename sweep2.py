"""Finer sweep around the best known thresholds."""
from bench import run_variant


def main():
    rows = []
    for t1 in range(22, 33):
        for t3 in range(21, 33):
            for t5 in range(21, 33):
                for sg in (16, 18, 20, 22):
                    for si in (17, 19, 21, 23):
                        for cs in (12, 14, 16):
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
