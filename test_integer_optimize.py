"""Correctness of integer_optimize (the greedy ILS final stage).

Uses an IDENTITY basis (each direction moves one box), so this validates the
incremental-eval + descent/ILS logic independent of the constraint basis
(constraint preservation is a property of get_integer_basis, already exercised
by the existing integer tuning). Three properties:

  1. Snapshot consistency: best_cost == area_f + 10*hpwl_f and
     best_overlap == box_ov_full + inst_ov, evaluated INDEPENDENTLY on
     best_X/best_Y. Catches any incremental drift in box_ov / hpwl / inst_ov.
  2. Non-worsening: from a feasible start, best is feasible and best_cost <= init.
  3. Never invents overlap: best_overlap <= init_overlap.
"""
import json, sys
import numpy as np
from test_incremental import build_arrays
from test_hpwl_incr import build_box_nets
from solution_p2 import (integer_optimize, full_box_overlap, eval_partial)


def run_case(case, trials=60000, seed=11):
    d = json.load(open(f"data/case_in/{case}_tst.json"))
    box, net_mat, rep, b2i, N = build_arrays(d)
    num_nets = net_mat.shape[0]
    num_inst = rep.shape[0]
    box_net_idx, box_net_len = build_box_nets(net_mat, N)

    # identity basis: direction j moves box j by 1
    basis = np.eye(N, dtype=np.float64)
    supp_idx = np.arange(N, dtype=np.int32).reshape(N, 1)
    supp_val = np.ones((N, 1), dtype=np.float64)
    supp_len = np.ones(N, dtype=np.int32)
    in_box = np.zeros(N, dtype=np.int8)
    net_mark = np.zeros(num_nets, dtype=np.int8)
    touched = np.zeros(num_nets, dtype=np.int32)
    new_sx = np.zeros(num_nets, dtype=np.float64)
    new_sy = np.zeros(num_nets, dtype=np.float64)
    ix1 = np.zeros(num_inst, dtype=np.float64)
    ix2 = np.zeros(num_inst, dtype=np.float64)
    iy1 = np.zeros(num_inst, dtype=np.float64)
    iy2 = np.zeros(num_inst, dtype=np.float64)

    # near-feasible grid start
    cell = float(np.max(box) * 1.3)
    cols = int(np.ceil(np.sqrt(N)))
    X = np.empty(N, dtype=np.float64); Y = np.empty(N, dtype=np.float64)
    for i in range(N):
        X[i] = float(i % cols) * cell
        Y[i] = float(i // cols) * cell

    # init cost/overlap (independent)
    x2 = X + box[:, 0]; y2 = Y + box[:, 1]
    box_ov0 = full_box_overlap(X, Y, x2, y2, N)
    area0, hpwl0, inst0 = eval_partial(X, Y, x2, y2, box, net_mat, rep, b2i)
    init_overlap = box_ov0 + inst0
    init_cost = area0 + 10.0 * hpwl0

    rng = np.array([seed], dtype=np.uint32)
    bX, bY, bc, bo = integer_optimize(
        X, Y, basis, basis, box, net_mat, rep, b2i, trials, rng,
        supp_idx, supp_val, supp_len, supp_idx, supp_val, supp_len, in_box,
        box_net_idx, box_net_len, net_mark, touched, new_sx, new_sy,
        ix1, ix2, iy1, iy2, 10.0, 4, 3, 3)

    # independent recompute on the returned best
    bx2 = bX + box[:, 0]; by2 = bY + box[:, 1]
    box_ov_b = full_box_overlap(bX, bY, bx2, by2, N)
    area_b, hpwl_b, inst_b = eval_partial(bX, bY, bx2, by2, box, net_mat, rep, b2i)
    overlap_b = box_ov_b + inst_b
    cost_b = area_b + 10.0 * hpwl_b

    rel_cost = abs(bc - cost_b) / (abs(cost_b) + 1.0)
    rel_ov = abs(bo - overlap_b) / (abs(overlap_b) + 1.0)
    nonworse = (bc <= init_cost + 1e-6) if init_overlap <= 0.0 else True
    ov_ok = bo <= init_overlap + 1e-3

    ok = rel_cost < 1e-9 and rel_ov < 1e-9 and nonworse and ov_ok
    print(f"{case:10s} init(cost={init_cost:.1f},ov={init_overlap:.3g}) "
          f"-> best(cost={bc:.1f},ov={bo:.3g}) | recompute(cost={cost_b:.1f},ov={overlap_b:.3g}) "
          f"rel_cost={rel_cost:.1e} rel_ov={rel_ov:.1e} "
          f"nonworse={nonworse} ov_ok={ov_ok} {'OK' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    cases = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        ["case0_0", "case2_0", "case5_0", "case9_0", "case1_3", "case4_2"]
    ok = all(run_case(c) for c in cases)
    print("ALL PASS" if ok else "SOME FAILED")
