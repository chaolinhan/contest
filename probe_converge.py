import json, time, sys
from solution import Solution
from validate import evaluate

cases = sys.argv[1].split(",") if len(sys.argv) > 1 else ["case0_0", "case5_0", "case9_0"]
budgets = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["8", "15", "25", "40"])]

for c in cases:
    d = json.load(open(f"data/case_in/{c}_tst.json"))
    print(f"--- {c} N={len(d['box_size'])} ---", flush=True)
    for b in budgets:
        Solution.TIME_BUDGET = float(b)
        t = time.time(); out = Solution().solve(d); el = time.time() - t
        r = evaluate(d, out["box_position"])
        ok = "OK" if r["valid"] else "BAD"
        print(f"  budget={b:3d} {ok} cost={r['cost']:10.1f} t={el:.1f}s", flush=True)
