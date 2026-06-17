"""Verify the incremental-eval primitives exactly match the full evaluator.

Tests, on random layouts (positions need not satisfy constraints for the eval
math to be valid):
  1. decomposition: eval_partial + full_box_overlap == evaluate_layout_core
  2. incremental update: box_ov + (touch_new - touch_old) == full_box_overlap
     after moving an arbitrary support set S by a random delta.
"""
import json, sys
import numpy as np
import solution_p2 as sp
from solution_p2 import (evaluate_layout_core, full_box_overlap, touch_box_overlap, eval_partial)


def build_arrays(d):
    box = np.array([[w * 10000.0, h * 10000.0] for w, h in d["box_size"]], dtype=np.float64)
    N = len(d["box_size"])
    nets = d.get("nets", [])
    if nets:
        ml = max(len(n) for n in nets)
        net_mat = np.full((len(nets), ml), -1, dtype=np.int32)
        for i, n in enumerate(nets):
            net_mat[i, :len(n)] = [x - 1 for x in n]
    else:
        net_mat = np.zeros((0, 1), dtype=np.int32)
    units, b2i = [], {}
    cnt = 0
    for rg in d.get("repeat_groups", []):
        for gk in rg.get("groups", []):
            gk_idx = [b - 1 for b in gk]
            units.append(gk_idx)
            for b in gk:
                b2i[b - 1] = cnt
            cnt += 1
    if units:
        ml = max(len(u) for u in units)
        rep = np.full((len(units), ml), -1, dtype=np.int32)
        for i, u in enumerate(units):
            rep[i, :len(u)] = u
    else:
        rep = np.zeros((0, 1), dtype=np.int32)
    b2i_arr = np.full(N, -1, dtype=np.int32)
    for b, inst in b2i.items():
        b2i_arr[b] = inst
    return box, net_mat, rep, b2i_arr, N


def run_case(case, n_moves=300, seed=1):
    d = json.load(open(f"data/case_in/{case}_tst.json"))
    box, net_mat, rep, b2i, N = build_arrays(d)
    rng = np.random.default_rng(seed)
    pen = 1e12
    max_err_decomp = 0.0
    max_err_incr = 0.0
    # random spread comparable to a real layout
    span = 200.0 * 10000.0
    X = rng.uniform(0, span, N)
    Y = rng.uniform(0, span, N)
    x2 = X + box[:, 0]
    y2 = Y + box[:, 1]
    in_box = np.zeros(N, dtype=np.int8)

    for _ in range(n_moves):
        # 1. decomposition check
        full_cost, full_ov = evaluate_layout_core(X, Y, box, net_mat, rep, b2i, pen)
        box_ov = full_box_overlap(X, Y, x2, y2, N)
        area, hpwl, inst = eval_partial(X, Y, x2, y2, box, net_mat, rep, b2i)
        recon_cost = area + 10.0 * hpwl + pen * ((box_ov + inst) * 1e-8)
        max_err_decomp = max(max_err_decomp, abs(recon_cost - full_cost) / (abs(full_cost) + 1.0))
        max_err_decomp = max(max_err_decomp, abs((box_ov + inst) - full_ov) / (abs(full_ov) + 1.0))

        # 2. incremental box-overlap check on a random support set
        k = rng.integers(1, min(5, N) + 1)
        S = rng.choice(N, size=k, replace=False).astype(np.int32)
        delta = rng.uniform(-span, span)
        axis_x = rng.random() < 0.5
        old_touch = touch_box_overlap(X, Y, x2, y2, N, S, k, in_box)
        for i in S:
            if axis_x:
                X[i] += delta; x2[i] += delta
            else:
                Y[i] += delta; y2[i] += delta
        new_touch = touch_box_overlap(X, Y, x2, y2, N, S, k, in_box)
        box_ov_inc = box_ov + (new_touch - old_touch)
        box_ov_full = full_box_overlap(X, Y, x2, y2, N)
        max_err_incr = max(max_err_incr, abs(box_ov_inc - box_ov_full) / (abs(box_ov_full) + 1.0))

    return max_err_decomp, max_err_incr


if __name__ == "__main__":
    cases = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        ["case0_0", "case2_0", "case4_0", "case5_0", "case9_0", "case4_5"]
    ok = True
    for c in cases:
        ed, ei = run_case(c)
        flag = "OK" if (ed < 1e-6 and ei < 1e-6) else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"{c:10s} decomp_err={ed:.2e} incr_err={ei:.2e}  {flag}", flush=True)
    print("ALL PASS" if ok else "SOME FAILED")
