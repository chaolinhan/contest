import json, sys, time
import solution_p2
from validate import evaluate, get_positions

cases = sys.argv[1].split(",")
stalls = [int(x) for x in sys.argv[2].split(",")]
budget = float(sys.argv[3]) if len(sys.argv) > 3 else 113.0
solution_p2.Solution.TIME_BUDGET = budget
solution_p2.Solution.USE_EARLY_ADVANCE = True
for c in cases:
    d = json.load(open(f"data/case_in/{c}_tst.json"))
    print(f"--- {c} N={len(d['box_size'])} ---", flush=True)
    for st in stalls:
        solution_p2.Solution.STALL_LIMIT = st
        t = time.time(); out = solution_p2.Solution().solve(d); el = time.time() - t
        r = evaluate(d, get_positions(out))
        print(f"  STALL={st} valid={r['valid']!s:5s} cost={r['cost']:11.1f} t={el:.1f}s", flush=True)
