import json, time, sys
import solution_p1
from validate import evaluate, get_positions

S = solution_p1.Solution
S.USE_COMPACT = False  # isolate the init-scale effect
S.TIME_BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
cases = sys.argv[1].split(",") if len(sys.argv) > 1 else ["case0_5", "case9_0"]
scales = [400000.0, 250000.0, 150000.0, 80000.0, 40000.0]

for c in cases:
    d = json.load(open(f"data/case_in/{c}_tst.json"))
    print(f"--- {c} N={len(d['box_size'])} ---", flush=True)
    for sc in scales:
        S.INIT_SCALE = sc
        out = S().solve(d)
        r = evaluate(d, get_positions(out))
        print(f"  scale={sc:9.0f} valid={r['valid']!s:5s} cost={r['cost']:10.1f} "
              f"area={r['area']:10.1f} hpwl={r['hpwl']:8.1f}", flush=True)
