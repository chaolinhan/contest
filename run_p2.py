import json, sys, time, glob
import solution_p2
from validate import evaluate, get_positions

mult = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
substr = sys.argv[2] if len(sys.argv) > 2 else "_0_tst"
budget = float(sys.argv[3]) if len(sys.argv) > 3 else 113.0
solution_p2.Solution.RESTART_MULT = mult
solution_p2.Solution.TIME_BUDGET = budget

files = sorted(f for f in glob.glob("data/case_in/*.json") if substr in f)
print(f"solution_p2 RESTART_MULT={mult} budget={budget} substr={substr}", flush=True)
tot = 0.0
nv = 0
for f in files:
    name = f.split("/")[-1].replace("_tst.json", "")
    d = json.load(open(f))
    t = time.time(); out = solution_p2.Solution().solve(d); el = time.time() - t
    r = evaluate(d, get_positions(out))
    flag = "" if r["valid"] else "  <<INVALID"
    print(f"  {name:10s} N={len(d['box_size']):2d} valid={r['valid']!s:5s} cost={r['cost']:11.1f} "
          f"area={r['area']:10.1f} hpwl={r['hpwl']:8.1f} t={el:5.1f}s{flag}", flush=True)
    if r["valid"]:
        nv += 1; tot += r["cost"]
print(f"TOTAL valid={nv}/{len(files)} sum_cost={tot:.1f}", flush=True)
