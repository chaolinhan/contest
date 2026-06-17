"""run_integer_tuning vs run_integer_tuning_incr must be bit-identical.

Integer coords -> overlaps are exact integers; both consume RNG identically;
so the incremental version must reproduce the full version exactly. Uses an
identity basis (each direction moves one box) over real net/repeat structure.
"""
import json, sys
import numpy as np
from test_incremental import build_arrays
from solution_p2 import run_integer_tuning, run_integer_tuning_incr


def run_case(case, iters=40000, seed=7):
    d = json.load(open(f"data/case_in/{case}_tst.json"))
    box, net_mat, rep, b2i, N = build_arrays(d)
    basis = np.eye(N, dtype=np.float64)  # identity: direction j moves box j by 1
    # support arrays for identity basis
    supp_idx = np.arange(N, dtype=np.int32).reshape(N, 1)
    supp_val = np.ones((N, 1), dtype=np.float64)
    supp_len = np.ones(N, dtype=np.int32)
    in_box = np.zeros(N, dtype=np.int8)

    # Near-feasible start (how the solver uses it): a non-overlapping grid, so
    # exploration stays in the low-overlap regime where costs are exact.
    rng = np.random.default_rng(seed)
    cell = float(np.max(box) * 1.3)
    cols = int(np.ceil(np.sqrt(N)))
    X = np.empty(N); Y = np.empty(N)
    for i in range(N):
        X[i] = float((i % cols)) * cell
        Y[i] = float((i // cols)) * cell

    r1 = np.array([123457], dtype=np.uint32)
    r2 = np.array([123457], dtype=np.uint32)
    bX1, bY1, bc1, bo1 = run_integer_tuning(X.copy(), Y.copy(), basis, basis, box, net_mat, rep, b2i, iters, r1)
    bX2, bY2, bc2, bo2 = run_integer_tuning_incr(X.copy(), Y.copy(), basis, basis, box, net_mat, rep, b2i, iters, r2,
                                                 supp_idx, supp_val, supp_len, supp_idx, supp_val, supp_len, in_box)
    same = (np.array_equal(bX1, bX2) and np.array_equal(bY1, bY2)
            and abs(bc1 - bc2) < 1e-6 and bo1 == bo2)
    print(f"{case:10s} full=({bc1:.1f},ov={bo1:.3g}) incr=({bc2:.1f},ov={bo2:.3g}) "
          f"Xeq={np.array_equal(bX1,bX2)} Yeq={np.array_equal(bY1,bY2)}  {'OK' if same else 'FAIL'}")
    return same


if __name__ == "__main__":
    cases = sys.argv[1].split(",") if len(sys.argv) > 1 else ["case0_0", "case5_0", "case9_0", "case4_5"]
    ok = all(run_case(c) for c in cases)
    print("ALL PASS" if ok else "SOME FAILED")
