import json, time, sys
from solution import Solution
from validate import evaluate

cases = sys.argv[1].split(",") if len(sys.argv) > 1 else ["case0_0", "case2_0", "case5_0", "case9_0"]
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
seeds = [246813579, 12345, 999983, 777, 2024021, 88121]

for c in cases:
    d = json.load(open(f"data/case_in/{c}_tst.json"))
    costs = []
    print(f"--- {c} N={len(d['box_size'])} budget={budget}s ---", flush=True)
    for s in seeds:
        Solution.TIME_BUDGET = budget
        Solution.SEED = s
        out = Solution().solve(d)
        r = evaluate(d, out["box_position"])
        if r["valid"]:
            costs.append(r["cost"])
        print(f"  seed={s:>10d} valid={r['valid']!s:5s} cost={r['cost']:10.1f}", flush=True)
    if costs:
        print(f"  >> best={min(costs):.1f} worst={max(costs):.1f} spread={max(costs)-min(costs):.1f} "
              f"({100*(max(costs)-min(costs))/min(costs):.2f}%)", flush=True)
