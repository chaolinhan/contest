"""Benchmark a solver across cases. Reports per-case validity + Cost and totals.

Usage:
    .venv/bin/python benchmark.py <solver_module:Class> [budget] [case_glob_substr]

Examples:
    .venv/bin/python benchmark.py reference_solution:Solution
    .venv/bin/python benchmark.py solution:Solution 30 _0_tst
"""
import json
import sys
import time
import glob

from validate import evaluate, get_positions


def load_solver(spec):
    mod_name, _, cls_name = spec.partition(":")
    cls_name = cls_name or "Solution"
    mod = __import__(mod_name)
    return getattr(mod, cls_name)


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else "solution:Solution"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else None
    substr = sys.argv[3] if len(sys.argv) > 3 else "_0_tst"  # one case per group by default

    Solver = load_solver(spec)
    if budget is not None and hasattr(Solver, "TIME_BUDGET"):
        Solver.TIME_BUDGET = budget

    files = sorted(f for f in glob.glob("data/case_in/*.json") if substr in f)
    total_cost = 0.0
    n_valid = 0
    print(f"solver={spec} budget={budget} cases={len(files)}")
    for f in files:
        name = f.split("/")[-1].replace("_tst.json", "")
        d = json.load(open(f))
        t = time.time()
        out = Solver().solve(d)
        el = time.time() - t
        r = evaluate(d, get_positions(out))
        flag = "" if r["valid"] else "  <<INVALID"
        print(f"  {name:10s} N={len(d['box_size']):2d} valid={r['valid']!s:5s} "
              f"cost={r['cost']:10.1f} area={r['area']:9.1f} hpwl={r['hpwl']:8.1f} t={el:5.1f}s{flag}")
        if not r["valid"]:
            for v in r["violations"][:3]:
                print("       -", v)
        else:
            n_valid += 1
            total_cost += r["cost"]
    print(f"TOTAL valid={n_valid}/{len(files)} sum_cost(valid)={total_cost:.1f}")


if __name__ == "__main__":
    main()
