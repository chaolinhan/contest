"""Fair comparison: seed np.random before each solve so the integer legalizer
is deterministic for BOTH solvers (cur_171 doesn't self-seed; solution_p2 does,
which just re-applies the same seed)."""
import json, sys, time
import numpy as np
from validate import evaluate, get_positions

spec = sys.argv[1]
cases = sys.argv[2].split(",")
budget = float(sys.argv[3]) if len(sys.argv) > 3 else 113.0
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 246813579
mod_name, _, cls = spec.partition(":")
mod = __import__(mod_name)
Solver = getattr(mod, cls or "Solution")
if hasattr(Solver, "TIME_BUDGET"):
    Solver.TIME_BUDGET = budget
print(f"{spec} budget={budget} seed={seed}", flush=True)
for c in cases:
    d = json.load(open(f"data/case_in/{c}_tst.json"))
    np.random.seed(seed & 0x7FFFFFFF)
    t = time.time(); out = Solver().solve(d); el = time.time() - t
    r = evaluate(d, get_positions(out))
    print(f"  {c:10s} valid={r['valid']!s:5s} cost={r['cost']:11.1f} t={el:.1f}s", flush=True)
