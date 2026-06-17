"""Verify incremental HPWL exactly matches the full HPWL recompute.

  1. compute_net_spans == eval_partial's hpwl term (raw, *5e-5).
  2. maintained hpwl_raw (via recompute_touched_nets + commit) == full
     recompute after a sequence of arbitrary support-set moves, on INTEGER
     coords (the regime the integer optimizer runs in).
"""
import json, sys
import numpy as np
from test_incremental import build_arrays
from solution_p2 import (compute_net_spans, recompute_touched_nets, commit_touched_nets,
                         eval_partial)


def build_box_nets(net_mat, N):
    """Ragged box->net adjacency. box_net_idx[b] = nets containing box b."""
    lists = [[] for _ in range(N)]
    for ni in range(net_mat.shape[0]):
        for m in range(net_mat.shape[1]):
            b = net_mat[ni, m]
            if b == -1:
                break
            lists[b].append(ni)
    maxn = max((len(l) for l in lists), default=1)
    maxn = max(maxn, 1)
    idx = np.zeros((N, maxn), dtype=np.int32)
    ln = np.zeros(N, dtype=np.int32)
    for b in range(N):
        for t, n in enumerate(lists[b]):
            idx[b, t] = n
        ln[b] = len(lists[b])
    return idx, ln


def run_case(case, n_moves=400, seed=3):
    d = json.load(open(f"data/case_in/{case}_tst.json"))
    box, net_mat, rep, b2i, N = build_arrays(d)
    box_net_idx, box_net_len = build_box_nets(net_mat, N)
    rng = np.random.default_rng(seed)

    # integer-ish spread
    span = 200.0 * 10000.0
    X = np.round(rng.uniform(0, span, N)).astype(np.float64)
    Y = np.round(rng.uniform(0, span, N)).astype(np.float64)
    x2 = X + box[:, 0]
    y2 = Y + box[:, 1]

    # 1. compute_net_spans vs eval_partial hpwl
    _, hpwl_full, _ = eval_partial(X, Y, x2, y2, box, net_mat, rep, b2i)
    hpwl_raw, sx, sy = compute_net_spans(X, Y, box, net_mat)
    err0 = abs(hpwl_raw * 5e-5 - hpwl_full) / (abs(hpwl_full) + 1.0)

    net_mark = np.zeros(net_mat.shape[0], dtype=np.int8)
    touched = np.zeros(net_mat.shape[0], dtype=np.int32)
    new_sx = np.zeros(net_mat.shape[0], dtype=np.float64)
    new_sy = np.zeros(net_mat.shape[0], dtype=np.float64)

    max_err = err0
    for _ in range(n_moves):
        k = int(rng.integers(1, min(5, N) + 1))
        S = rng.choice(N, size=k, replace=False).astype(np.int32)
        axis_x = rng.random() < 0.5
        delta = float(rng.uniform(-span * 0.05, span * 0.05))
        # apply move
        for i in S:
            if axis_x:
                X[i] += delta; x2[i] += delta
            else:
                Y[i] += delta; y2[i] += delta
        # incremental update (positions already moved)
        dlt, nt = recompute_touched_nets(X, Y, box, net_mat, S, k,
                                         box_net_idx, box_net_len, sx, sy,
                                         net_mark, touched, new_sx, new_sy)
        hpwl_raw += dlt
        commit_touched_nets(sx, sy, touched, new_sx, new_sy, nt)
        # ground truth
        _, hpwl_full2, _ = eval_partial(X, Y, x2, y2, box, net_mat, rep, b2i)
        hpwl_raw_gt, _, _ = compute_net_spans(X, Y, box, net_mat)
        rel = abs(hpwl_raw - hpwl_raw_gt) / (abs(hpwl_raw_gt) + 1.0)
        rel2 = abs(hpwl_raw * 5e-5 - hpwl_full2) / (abs(hpwl_full2) + 1.0)
        max_err = max(max_err, rel, rel2)
    return max_err


if __name__ == "__main__":
    cases = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        ["case0_0", "case2_0", "case4_0", "case5_0", "case9_0", "case1_3"]
    ok = True
    for c in cases:
        e = run_case(c)
        flag = "OK" if e < 1e-9 else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"{c:10s} max_rel_err={e:.2e}  {flag}", flush=True)
    print("ALL PASS" if ok else "SOME FAILED")
