import json, sys, time
from validate import evaluate, get_positions

spec = sys.argv[1]          # module:Class
cases = sys.argv[2].split(",")
budget = float(sys.argv[3]) if len(sys.argv) > 3 else 113.0
mod_name, _, cls = spec.partition(":")
mod = __import__(mod_name)
Solver = getattr(mod, cls or "Solution")
if hasattr(Solver, "TIME_BUDGET"):
    Solver.TIME_BUDGET = budget
print(f"{spec} budget={budget}", flush=True)
for c in cases:
    d = json.load(open(f"data/case_in/{c}_tst.json"))
    t = time.time(); out = Solver().solve(d); el = time.time() - t
    r = evaluate(d, get_positions(out))
    print(f"  {c:10s} valid={r['valid']!s:5s} cost={r['cost']:11.1f} t={el:.1f}s", flush=True)
