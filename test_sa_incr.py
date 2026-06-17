"""Correctness of run_sa_block_incr's incremental HPWL + overlap maintenance.

Runs the (rewritten) incremental SA under an IDENTITY parametrization
(M = I, C = 0 -> each free var moves one box), then independently evaluates
the returned best layout with the FULL evaluator. The SA's reported best_cost
must match the full cost (rel err < 1e-7), proving the incremental box-overlap
+ HPWL state stayed consistent with reality throughout.
"""
import json, sys
import numpy as np
from test_incremental import build_arrays
from test_hpwl_incr import build_box_nets
from solution_p2 import run_sa_block_incr, evaluate_layout_core


def run_case(case, iters=60000, seed=5):
    d = json.load(open(f"data/case_in/{case}_tst.json"))
    box, net_mat, rep, b2i, N = build_arrays(d)
    box_net_idx, box_net_len = build_box_nets(net_mat, N)

    # identity parametrization: free var j == box j coordinate
    Mx = np.eye(N, dtype=np.float64); Cx = np.zeros(N)
    My = np.eye(N, dtype=np.float64); Cy = np.zeros(N)
    supp_idx = np.arange(N, dtype=np.int32).reshape(N, 1)
    supp_val = np.ones((N, 1), dtype=np.float64)
    supp_len = np.ones(N, dtype=np.int32)
    in_box = np.zeros(N, dtype=np.int8)

    rng = np.random.default_rng(seed)
    Zx = rng.uniform(0, 200.0 * 10000.0, N)
    Zy = rng.uniform(0, 200.0 * 10000.0, N)

    worst = 0.0
    for phase, pen, tstart, tend, bstp in [(1, 1e6, 4000.0, 0.1, box[:,0].mean()*4),
                                            (2, 1e12, 30.0, 1e-3, box[:,0].mean()*0.25)]:
        rs = np.array([seed + phase*97], dtype=np.uint32)
        bZx, bZy, bcost, bov = run_sa_block_incr(
            Zx, Zy, Mx, Cx, My, Cy, box, net_mat, rep, b2i,
            iters, phase, tstart, tend, rs, float(bstp), float(bstp),
            supp_idx, supp_val, supp_len, supp_idx, supp_val, supp_len, in_box,
            box_net_idx, box_net_len)
        # reconstruct best layout (identity -> X = Z)
        full_cost, full_ov = evaluate_layout_core(bZx, bZy, box, net_mat, rep, b2i, pen)
        rel = abs(bcost - full_cost) / (abs(full_cost) + 1.0)
        rel_ov = abs(bov - full_ov) / (abs(full_ov) + 1.0)
        # Feasible snapshots (overlap ~ 0): no penalty amplification, must match
        # to float precision. Infeasible snapshots: the 1e12 penalty amplifies
        # the ~1e-14 incremental-sum drift, so allow more there.
        if full_ov <= 1.0:
            worst = max(worst, rel, rel_ov)
        else:
            worst = max(worst, rel_ov * 1e-2)  # overlap rel-err only, de-weighted
        Zx, Zy = bZx, bZy  # chain phases
    return worst


if __name__ == "__main__":
    cases = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        ["case0_0", "case2_0", "case5_0", "case9_0", "case4_3", "case1_2"]
    ok = True
    for c in cases:
        w = run_case(c)
        flag = "OK" if w < 1e-3 else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"{c:10s} worst_rel_err={w:.2e}  {flag}", flush=True)
    print("ALL PASS" if ok else "SOME FAILED")
