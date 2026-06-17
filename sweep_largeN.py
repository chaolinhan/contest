import json, sys, time
import solution_p2
from validate import evaluate, get_positions

case = sys.argv[1]
mults = [float(x) for x in sys.argv[2].split(",")]
budget = float(sys.argv[3]) if len(sys.argv) > 3 else 113.0
d = json.load(open(f"data/case_in/{case}_tst.json"))
solution_p2.Solution.TIME_BUDGET = budget
print(f"{case} N={len(d['box_size'])} budget={budget}", flush=True)
for m in mults:
    solution_p2.Solution.LARGE_N_MULT = m
    t = time.time(); out = solution_p2.Solution().solve(d); el = time.time() - t
    r = evaluate(d, get_positions(out))
    print(f"  LARGE_N_MULT={m:.1f} valid={r['valid']!s:5s} cost={r['cost']:11.1f} "
          f"area={r['area']:10.1f} hpwl={r['hpwl']:8.1f} t={el:.1f}s", flush=True)
