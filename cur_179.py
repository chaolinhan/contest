import numpy as np
import time
from numba import njit

# ==========================================
# 0. High-performance RNG (xorshift32)
# ==========================================
@njit(fastmath=True)
def next_rand(rng_state):
    x = rng_state[0]
    x = np.uint32(x ^ (x << 13))
    x = np.uint32(x ^ (x >> 17))
    x = np.uint32(x ^ (x << 5))
    rng_state[0] = x
    return float(x) / 4294967296.0

# ==========================================
# 1. Splitted & Optimized Cost Evaluations
# ==========================================
@njit(fastmath=True)
def compute_single_net_hpwl(net_idx, X, Y, box_sizes_float, net_matrix):
    first = net_matrix[net_idx][0]
    if first == -1: return 0.0
    nmin_x = X[first] * 2.0 + box_sizes_float[first, 0]
    nmax_x = nmin_x
    nmin_y = Y[first] * 2.0 + box_sizes_float[first, 1]
    nmax_y = nmin_y
    cnt = 1
    for m in range(1, net_matrix.shape[1]):
        b = net_matrix[net_idx][m]
        if b == -1: break
        cx = X[b] * 2.0 + box_sizes_float[b, 0]
        cy = Y[b] * 2.0 + box_sizes_float[b, 1]
        if cx < nmin_x: nmin_x = cx
        elif cx > nmax_x: nmax_x = cx
        if cy < nmin_y: nmin_y = cy
        elif cy > nmax_y: nmax_y = cy
        cnt += 1
    if cnt > 1:
        return (nmax_x - nmin_x) + (nmax_y - nmin_y)
    return 0.0

@njit(fastmath=True)
def compute_all_hpwl(X, Y, box_sizes_float, net_matrix):
    hpwl = 0.0
    for net_idx in range(net_matrix.shape[0]):
        hpwl += compute_single_net_hpwl(net_idx, X, Y, box_sizes_float, net_matrix)
    return hpwl * 5e-5

@njit(fastmath=True)
def evaluate_overlap_area(X, Y, box_sizes_float, repeat_units, box_to_inst, 
                          x2_arr, y2_arr, inst_x1, inst_x2, inst_y1, inst_y2):
    N = box_sizes_float.shape[0]
    for i in range(N):
        x2_arr[i] = X[i] + box_sizes_float[i, 0]
        y2_arr[i] = Y[i] + box_sizes_float[i, 1]

    overlap_val = 0.0
    # 1.1 Base overlap
    for i in range(N):
        xi1, xi2 = X[i], x2_arr[i]
        yi1, yi2 = Y[i], y2_arr[i]
        for j in range(i + 1, N):
            xj1, xj2 = X[j], x2_arr[j]
            ix1 = xi1 if xi1 > xj1 else xj1
            ix2 = xi2 if xi2 < xj2 else xj2
            ox = ix2 - ix1
            if ox > 0.0:
                yj1, yj2 = Y[j], y2_arr[j]
                iy1 = yi1 if yi1 > yj1 else yj1
                iy2 = yi2 if yi2 < yj2 else yj2
                oy = iy2 - iy1
                if oy > 0.0:
                    overlap_val += ox * oy

    num_inst = repeat_units.shape[0]
    if num_inst > 0:
        for k in range(num_inst):
            r0 = repeat_units[k][0]
            min_x = X[r0]; max_x = x2_arr[r0]
            min_y = Y[r0]; max_y = y2_arr[r0]
            for m in range(1, repeat_units.shape[1]):
                idx = repeat_units[k][m]
                if idx == -1: break
                if X[idx] < min_x: min_x = X[idx]
                if x2_arr[idx] > max_x: max_x = x2_arr[idx]
                if Y[idx] < min_y: min_y = Y[idx]
                if y2_arr[idx] > max_y: max_y = y2_arr[idx]
            inst_x1[k] = min_x
            inst_x2[k] = max_x
            inst_y1[k] = min_y
            inst_y2[k] = max_y

        for i in range(N):
            idx_inst = box_to_inst[i]
            xi1, xi2 = X[i], x2_arr[i]
            yi1, yi2 = Y[i], y2_arr[i]
            for k in range(num_inst):
                if k == idx_inst: continue
                ix1 = xi1 if xi1 > inst_x1[k] else inst_x1[k]
                ix2 = xi2 if xi2 < inst_x2[k] else inst_x2[k]
                ox = ix2 - ix1
                if ox > 0.0:
                    iy1 = yi1 if yi1 > inst_y1[k] else inst_y1[k]
                    iy2 = yi2 if yi2 < inst_y2[k] else inst_y2[k]
                    oy = iy2 - iy1
                    if oy > 0.0:
                        overlap_val += ox * oy

        for k1 in range(num_inst):
            for k2 in range(k1 + 1, num_inst):
                ix1 = inst_x1[k1] if inst_x1[k1] > inst_x1[k2] else inst_x1[k2]
                ix2 = inst_x2[k1] if inst_x2[k1] < inst_x2[k2] else inst_x2[k2]
                ox = ix2 - ix1
                if ox > 0.0:
                    iy1 = inst_y1[k1] if inst_y1[k1] > inst_y1[k2] else inst_y1[k2]
                    iy2 = inst_y2[k1] if inst_y2[k1] < inst_y2[k2] else inst_y2[k2]
                    oy = iy2 - iy1
                    if oy > 0.0:
                        overlap_val += ox * oy * 5.0

    gmin_x, gmax_x = X[0], x2_arr[0]
    gmin_y, gmax_y = Y[0], y2_arr[0]
    for i in range(1, N):
        if X[i] < gmin_x: gmin_x = X[i]
        if x2_arr[i] > gmax_x: gmax_x = x2_arr[i]
        if Y[i] < gmin_y: gmin_y = Y[i]
        if y2_arr[i] > gmax_y: gmax_y = y2_arr[i]
    area_f = (gmax_x - gmin_x) * (gmax_y - gmin_y) * 1e-8

    return area_f, overlap_val

# ==========================================
# 1b. Incremental overlap primitives (optimization #1)
# ==========================================
@njit(fastmath=True)
def full_box_overlap(X, Y, x2, y2, N):
    ov = 0.0
    for i in range(N):
        xi1, xi2, yi1, yi2 = X[i], x2[i], Y[i], y2[i]
        for j in range(i + 1, N):
            ix1 = xi1 if xi1 > X[j] else X[j]
            ix2 = xi2 if xi2 < x2[j] else x2[j]
            ox = ix2 - ix1
            if ox > 0.0:
                iy1 = yi1 if yi1 > Y[j] else Y[j]
                iy2 = yi2 if yi2 < y2[j] else y2[j]
                oy = iy2 - iy1
                if oy > 0.0:
                    ov += ox * oy
    return ov

@njit(fastmath=True)
def touch_box_overlap(X, Y, x2, y2, N, S, k, in_box):
    """Overlap touching support set S (k active boxes in S[:k]), each pair counted
    EXACTLY ONCE: (S vs non-S) + (within-S pairs). With identical S before/after a
    move, box_ov + (new_touch - old_touch) is an EXACT total box-box overlap update."""
    for a in range(k):
        in_box[S[a]] = 1
    ov = 0.0
    for a in range(k):
        i = S[a]
        xi1, xi2, yi1, yi2 = X[i], x2[i], Y[i], y2[i]
        for j in range(N):
            if in_box[j]:
                continue
            ix1 = xi1 if xi1 > X[j] else X[j]
            ix2 = xi2 if xi2 < x2[j] else x2[j]
            ox = ix2 - ix1
            if ox > 0.0:
                iy1 = yi1 if yi1 > Y[j] else Y[j]
                iy2 = yi2 if yi2 < y2[j] else y2[j]
                oy = iy2 - iy1
                if oy > 0.0:
                    ov += ox * oy
    for a in range(k):
        i = S[a]
        xi1, xi2, yi1, yi2 = X[i], x2[i], Y[i], y2[i]
        for b in range(a + 1, k):
            j = S[b]
            ix1 = xi1 if xi1 > X[j] else X[j]
            ix2 = xi2 if xi2 < x2[j] else x2[j]
            ox = ix2 - ix1
            if ox > 0.0:
                iy1 = yi1 if yi1 > Y[j] else Y[j]
                iy2 = yi2 if yi2 < y2[j] else y2[j]
                oy = iy2 - iy1
                if oy > 0.0:
                    ov += ox * oy
    for a in range(k):
        in_box[S[a]] = 0
    return ov

@njit(fastmath=True)
def inst_and_area(X, Y, x2, y2, repeat_units, box_to_inst,
                  inst_x1, inst_x2, inst_y1, inst_y2):
    """Full instance overlap + area (kept full in v1: cheap O(N*num_inst))."""
    N = X.shape[0]
    overlap_val = 0.0
    num_inst = repeat_units.shape[0]
    if num_inst > 0:
        for k in range(num_inst):
            r0 = repeat_units[k][0]
            min_x = X[r0]; max_x = x2[r0]
            min_y = Y[r0]; max_y = y2[r0]
            for m in range(1, repeat_units.shape[1]):
                idx = repeat_units[k][m]
                if idx == -1: break
                if X[idx] < min_x: min_x = X[idx]
                if x2[idx] > max_x: max_x = x2[idx]
                if Y[idx] < min_y: min_y = Y[idx]
                if y2[idx] > max_y: max_y = y2[idx]
            inst_x1[k] = min_x; inst_x2[k] = max_x
            inst_y1[k] = min_y; inst_y2[k] = max_y
        for i in range(N):
            idx_inst = box_to_inst[i]
            xi1, xi2 = X[i], x2[i]
            yi1, yi2 = Y[i], y2[i]
            for k in range(num_inst):
                if k == idx_inst: continue
                ix1 = xi1 if xi1 > inst_x1[k] else inst_x1[k]
                ix2 = xi2 if xi2 < inst_x2[k] else inst_x2[k]
                ox = ix2 - ix1
                if ox > 0.0:
                    iy1 = yi1 if yi1 > inst_y1[k] else inst_y1[k]
                    iy2 = yi2 if yi2 < inst_y2[k] else inst_y2[k]
                    oy = iy2 - iy1
                    if oy > 0.0:
                        overlap_val += ox * oy
        for k1 in range(num_inst):
            for k2 in range(k1 + 1, num_inst):
                ix1 = inst_x1[k1] if inst_x1[k1] > inst_x1[k2] else inst_x1[k2]
                ix2 = inst_x2[k1] if inst_x2[k1] < inst_x2[k2] else inst_x2[k2]
                ox = ix2 - ix1
                if ox > 0.0:
                    iy1 = inst_y1[k1] if inst_y1[k1] > inst_y1[k2] else inst_y1[k2]
                    iy2 = inst_y2[k1] if inst_y2[k1] < inst_y2[k2] else inst_y2[k2]
                    oy = iy2 - iy1
                    if oy > 0.0:
                        overlap_val += ox * oy * 5.0
    gmin_x, gmax_x = X[0], x2[0]
    gmin_y, gmax_y = Y[0], y2[0]
    for i in range(1, N):
        if X[i] < gmin_x: gmin_x = X[i]
        if x2[i] > gmax_x: gmax_x = x2[i]
        if Y[i] < gmin_y: gmin_y = Y[i]
        if y2[i] > gmax_y: gmax_y = y2[i]
    area_f = (gmax_x - gmin_x) * (gmax_y - gmin_y) * 1e-8
    return area_f, overlap_val

# ==========================================
# 2. Stateful & Incremental SA Cores
# ==========================================
@njit
def run_sa_steps_float(Z_x, Z_y, M_x, C_x, M_y, C_y, box_sizes_float, net_matrix,
                       repeat_units, box_to_inst,
                       v2b_x, v2b_y, box_to_nets_mat, box_to_zx, box_to_zy,
                       scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved,
                       phase, T, base_step, rng_state, w_area, w_hpwl, steps):
    
    X = M_x @ Z_x + C_x
    Y = M_y @ Z_y + C_y
    penalty_weight = 1e6 if phase == 1 else 1e12

    area_f, current_overlap = evaluate_overlap_area(X, Y, box_sizes_float, repeat_units, box_to_inst,
                                                    scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2)
    current_hpwl = compute_all_hpwl(X, Y, box_sizes_float, net_matrix)
    current_cost = w_area * area_f + w_hpwl * current_hpwl + penalty_weight * (current_overlap * 1e-8)

    best_Z_x = Z_x.copy()
    best_Z_y = Z_y.copy()
    best_cost = current_cost
    best_overlap = current_overlap

    len_x, len_y = len(Z_x), len(Z_y)
    num_nets = net_matrix.shape[0]

    for _ in range(steps):
        idx_x, idx_y = -1, -1
        delta = 0.0

        # >>> NET-AWARE MOVE (25% Probability) <<<
        is_net_aware = False
        if num_nets > 0 and next_rand(rng_state) < 0.25:
            net_idx = int(next_rand(rng_state) * num_nets)
            nmin_x, nmax_x = 1e20, -1e20
            nmin_y, nmax_y = 1e20, -1e20
            bcnt = 0
            for m in range(net_matrix.shape[1]):
                b = net_matrix[net_idx][m]
                if b == -1: break
                bcnt += 1
                cx = X[b] + box_sizes_float[b, 0]*0.5
                cy = Y[b] + box_sizes_float[b, 1]*0.5
                if cx < nmin_x: nmin_x = cx
                if cx > nmax_x: nmax_x = cx
                if cy < nmin_y: nmin_y = cy
                if cy > nmax_y: nmax_y = cy
                
            if bcnt > 1:
                center_x = (nmin_x + nmax_x) * 0.5
                center_y = (nmin_y + nmax_y) * 0.5
                rand_m = int(next_rand(rng_state) * bcnt)
                target_b = net_matrix[net_idx][rand_m]
                
                if next_rand(rng_state) < 0.5 and len_x > 0:
                    zx_idx = box_to_zx[target_b]
                    if zx_idx != -1:
                        idx_x = zx_idx
                        dist = center_x - (X[target_b] + box_sizes_float[target_b, 0]*0.5)
                        coeff = M_x[target_b, zx_idx]
                        if abs(coeff) > 1e-5:
                            delta = (dist / coeff) * next_rand(rng_state) * 1.5
                            is_net_aware = True
                elif len_y > 0:
                    zy_idx = box_to_zy[target_b]
                    if zy_idx != -1:
                        idx_y = zy_idx
                        dist = center_y - (Y[target_b] + box_sizes_float[target_b, 1]*0.5)
                        coeff = M_y[target_b, zy_idx]
                        if abs(coeff) > 1e-5:
                            delta = (dist / coeff) * next_rand(rng_state) * 1.5
                            is_net_aware = True

        # >>> RANDOM EXPLORATION <<<
        if not is_net_aware:
            r1 = next_rand(rng_state)
            curr_step = base_step if r1 < 0.7 else base_step * 3.0
            if len_x > 0 and len_y > 0:
                if next_rand(rng_state) < 0.5:
                    idx_x = int(next_rand(rng_state) * len_x)
                    delta = -curr_step + next_rand(rng_state) * (2.0 * curr_step)
                else:
                    idx_y = int(next_rand(rng_state) * len_y)
                    delta = -curr_step + next_rand(rng_state) * (2.0 * curr_step)
            elif len_x > 0:
                idx_x = int(next_rand(rng_state) * len_x)
                delta = -curr_step + next_rand(rng_state) * (2.0 * curr_step)
            elif len_y > 0:
                idx_y = int(next_rand(rng_state) * len_y)
                delta = -curr_step + next_rand(rng_state) * (2.0 * curr_step)
            else: break

        # >>> INCREMENTAL HPWL CALCULATION <<<
        net_moved[:] = False
        old_partial_hpwl = 0.0
        
        moved_boxes = v2b_x[idx_x] if idx_x != -1 else v2b_y[idx_y]
        for i in range(moved_boxes.shape[0]):
            b = moved_boxes[i]
            if b == -1: break
            for j in range(box_to_nets_mat.shape[1]):
                nt = box_to_nets_mat[b, j]
                if nt == -1: break
                net_moved[nt] = True
                
        for nt in range(num_nets):
            if net_moved[nt]:
                old_partial_hpwl += compute_single_net_hpwl(nt, X, Y, box_sizes_float, net_matrix)

        # APPLY DELTA
        if idx_x != -1:
            Z_x[idx_x] += delta
            for i in range(X.shape[0]): X[i] += delta * M_x[i, idx_x]
        else:
            Z_y[idx_y] += delta
            for i in range(Y.shape[0]): Y[i] += delta * M_y[i, idx_y]

        new_partial_hpwl = 0.0
        for nt in range(num_nets):
            if net_moved[nt]:
                new_partial_hpwl += compute_single_net_hpwl(nt, X, Y, box_sizes_float, net_matrix)

        new_hpwl = current_hpwl + (new_partial_hpwl - old_partial_hpwl) * 5e-5
        new_area, new_overlap = evaluate_overlap_area(X, Y, box_sizes_float, repeat_units, box_to_inst,
                                                      scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2)
        new_cost = w_area * new_area + w_hpwl * new_hpwl + penalty_weight * (new_overlap * 1e-8)

        # Accept Logic
        accept = False
        if phase == 1:
            if new_overlap < current_overlap: accept = True
            else:
                dE = (new_cost - current_cost) / T
                if dE < 70.0 and next_rand(rng_state) < np.exp(-dE): accept = True
        else:
            if new_overlap <= 0.0:
                if current_overlap > 0.0 or new_cost < current_cost: accept = True
                else:
                    dE = (new_cost - current_cost) / T
                    if dE < 70.0 and next_rand(rng_state) < np.exp(-dE): accept = True
            else:
                if new_overlap < current_overlap: accept = True
                else:
                    dE = (new_cost - current_cost) / T
                    if dE < 70.0 and next_rand(rng_state) < np.exp(-dE): accept = True

        if accept:
            current_cost, current_overlap, current_hpwl = new_cost, new_overlap, new_hpwl
            if phase == 1:
                if current_overlap < best_overlap or (abs(current_overlap - best_overlap) < 1e-3 and current_cost < best_cost):
                    best_overlap, best_cost = current_overlap, current_cost
                    best_Z_x[:] = Z_x; best_Z_y[:] = Z_y
            else:
                if current_overlap <= 0.0 and (best_overlap > 0.0 or current_cost < best_cost):
                    best_cost, best_overlap = current_cost, current_overlap
                    best_Z_x[:] = Z_x; best_Z_y[:] = Z_y
                elif best_overlap > 0.0 and current_overlap < best_overlap:
                    best_cost, best_overlap = current_cost, current_overlap
                    best_Z_x[:] = Z_x; best_Z_y[:] = Z_y
        else: # Revert
            if idx_x != -1:
                Z_x[idx_x] -= delta
                for i in range(X.shape[0]): X[i] -= delta * M_x[i, idx_x]
            else:
                Z_y[idx_y] -= delta
                for i in range(Y.shape[0]): Y[i] -= delta * M_y[i, idx_y]

    return Z_x, Z_y, best_Z_x, best_Z_y, best_cost, best_overlap

@njit
def run_integer_tuning_steps(X, Y, basis_x, basis_y, box_sizes_float, net_matrix,
                             repeat_units, box_to_inst, b2b_x, b2b_y, box_to_nets_mat,
                             scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved,
                             rng_state, w_area, w_hpwl, T, steps):
    penalty_weight = 1e12
    area_f, current_overlap = evaluate_overlap_area(X, Y, box_sizes_float, repeat_units, box_to_inst,
                                                    scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2)
    current_hpwl = compute_all_hpwl(X, Y, box_sizes_float, net_matrix)
    current_cost = w_area * area_f + w_hpwl * current_hpwl + penalty_weight * (current_overlap * 1e-8)

    best_X = X.copy()
    best_Y = Y.copy()
    best_cost = current_cost
    best_overlap = current_overlap

    len_bx = basis_x.shape[1]
    len_by = basis_y.shape[1]
    num_nets = net_matrix.shape[0]

    for _ in range(steps):
        r_step = next_rand(rng_state)
        mult = 1 if r_step < 0.7 else (2 if r_step < 0.9 else 3)
        if next_rand(rng_state) < 0.5: mult = -mult

        idx_x, idx_y = -1, -1
        if len_bx > 0 and len_by > 0:
            if next_rand(rng_state) < 0.5:
                idx_x = int(next_rand(rng_state) * len_bx)
            else:
                idx_y = int(next_rand(rng_state) * len_by)
        elif len_bx > 0:
            idx_x = int(next_rand(rng_state) * len_bx)
        elif len_by > 0:
            idx_y = int(next_rand(rng_state) * len_by)
        else: break

        # Incremental HPWL Setup
        net_moved[:] = False
        old_partial_hpwl = 0.0
        
        moved_boxes = b2b_x[idx_x] if idx_x != -1 else b2b_y[idx_y]
        for i in range(moved_boxes.shape[0]):
            b = moved_boxes[i]
            if b == -1: break
            for j in range(box_to_nets_mat.shape[1]):
                nt = box_to_nets_mat[b, j]
                if nt == -1: break
                net_moved[nt] = True
                
        for nt in range(num_nets):
            if net_moved[nt]:
                old_partial_hpwl += compute_single_net_hpwl(nt, X, Y, box_sizes_float, net_matrix)

        if idx_x != -1:
            for i in range(X.shape[0]): X[i] += mult * basis_x[i, idx_x]
        else:
            for i in range(Y.shape[0]): Y[i] += mult * basis_y[i, idx_y]

        new_partial_hpwl = 0.0
        for nt in range(num_nets):
            if net_moved[nt]:
                new_partial_hpwl += compute_single_net_hpwl(nt, X, Y, box_sizes_float, net_matrix)
                
        new_hpwl = current_hpwl + (new_partial_hpwl - old_partial_hpwl) * 5e-5
        new_area, new_overlap = evaluate_overlap_area(X, Y, box_sizes_float, repeat_units, box_to_inst,
                                                      scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2)
        new_cost = w_area * new_area + w_hpwl * new_hpwl + penalty_weight * (new_overlap * 1e-8)

        accept = False
        if new_overlap <= 0.0:
            if current_overlap > 0.0 or new_cost < current_cost: accept = True
            else:
                dE = (new_cost - current_cost) / T
                if dE < 70.0 and next_rand(rng_state) < np.exp(-dE): accept = True
        else:
            if new_overlap < current_overlap: accept = True
            else:
                dE = (new_cost - current_cost) / T
                if dE < 70.0 and next_rand(rng_state) < np.exp(-dE): accept = True

        if accept:
            current_cost, current_overlap, current_hpwl = new_cost, new_overlap, new_hpwl
            if current_overlap <= 0.0 and (best_overlap > 0.0 or current_cost < best_cost):
                best_cost, best_overlap = current_cost, current_overlap
                best_X[:] = X; best_Y[:] = Y
            elif best_overlap > 0.0 and current_overlap < best_overlap:
                best_cost, best_overlap = current_cost, current_overlap
                best_X[:] = X; best_Y[:] = Y
        else:
            if idx_x != -1:
                for i in range(X.shape[0]): X[i] -= mult * basis_x[i, idx_x]
            if idx_y != -1:
                for i in range(Y.shape[0]): Y[i] -= mult * basis_y[i, idx_y]

    return X, Y, best_X, best_Y, best_cost, best_overlap

# ==========================================
# 2b. Incremental-overlap SA / tuning cores (optimization #1 + #6)
# Box-box overlap maintained incrementally via touch(); instance overlap + area
# recomputed fully each step (cheap O(N*num_inst)). X/x2 updates are sparse over
# the moved support. Behavior (move selection, accept rule, RNG) is identical to
# the full-eval cores; only the eval math differs -> verified equivalent.
# ==========================================
@njit
def run_sa_steps_float_incr(Z_x, Z_y, M_x, C_x, M_y, C_y, box_sizes_float, net_matrix,
                            repeat_units, box_to_inst,
                            v2b_x_idx, v2b_x_val, v2b_y_idx, v2b_y_val,
                            box_to_nets_mat, box_to_zx, box_to_zy,
                            x2, y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved, in_box,
                            phase, T, base_step, rng_state, w_area, w_hpwl, steps,
                            na_prob, na_mode, na_scr):
    N = box_sizes_float.shape[0]
    X = M_x @ Z_x + C_x
    Y = M_y @ Z_y + C_y
    penalty_weight = 1e6 if phase == 1 else 1e12
    for i in range(N):
        x2[i] = X[i] + box_sizes_float[i, 0]
        y2[i] = Y[i] + box_sizes_float[i, 1]
    box_ov = full_box_overlap(X, Y, x2, y2, N)
    area_f, inst_ov = inst_and_area(X, Y, x2, y2, repeat_units, box_to_inst,
                                    scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2)
    current_overlap = box_ov + inst_ov
    current_hpwl = compute_all_hpwl(X, Y, box_sizes_float, net_matrix)
    current_cost = w_area * area_f + w_hpwl * current_hpwl + penalty_weight * (current_overlap * 1e-8)
    current_box_ov = box_ov

    best_Z_x = Z_x.copy(); best_Z_y = Z_y.copy()
    best_cost = current_cost; best_overlap = current_overlap
    len_x, len_y = len(Z_x), len(Z_y)
    num_nets = net_matrix.shape[0]

    for _ in range(steps):
        idx_x, idx_y = -1, -1
        delta = 0.0

        is_net_aware = False
        if num_nets > 0 and next_rand(rng_state) < na_prob:
            net_idx = int(next_rand(rng_state) * num_nets)
            nmin_x, nmax_x = 1e20, -1e20
            nmin_y, nmax_y = 1e20, -1e20
            bcnt = 0
            for m in range(net_matrix.shape[1]):
                b = net_matrix[net_idx][m]
                if b == -1: break
                bcnt += 1
                cx = X[b] + box_sizes_float[b, 0] * 0.5
                cy = Y[b] + box_sizes_float[b, 1] * 0.5
                if cx < nmin_x: nmin_x = cx
                if cx > nmax_x: nmax_x = cx
                if cy < nmin_y: nmin_y = cy
                if cy > nmax_y: nmax_y = cy
            if bcnt > 1:
                if na_mode == 1:
                    # #3: move the SPAN-DEFINING extreme box toward the axis MEDIAN
                    # (directly shrinks HPWL span, unlike a random interior member).
                    use_x = (next_rand(rng_state) < 0.5)
                    if (use_x and len_x > 0) or ((not use_x) and len_y > 0 and len_x == 0):
                        ax = 0
                    elif len_y > 0:
                        ax = 1
                    else:
                        ax = 0 if len_x > 0 else 1
                    for m in range(bcnt):
                        b = net_matrix[net_idx][m]
                        if ax == 0:
                            na_scr[m] = X[b] + box_sizes_float[b, 0] * 0.5
                        else:
                            na_scr[m] = Y[b] + box_sizes_float[b, 1] * 0.5
                    # insertion sort centers to get median
                    for mm in range(1, bcnt):
                        key = na_scr[mm]; kk = mm - 1
                        while kk >= 0 and na_scr[kk] > key:
                            na_scr[kk + 1] = na_scr[kk]; kk -= 1
                        na_scr[kk + 1] = key
                    median_v = na_scr[bcnt // 2]
                    lo_v = na_scr[0]; hi_v = na_scr[bcnt - 1]
                    pick_hi = next_rand(rng_state) < 0.5
                    target_b = -1
                    for m in range(bcnt):
                        b = net_matrix[net_idx][m]
                        cv = (X[b] + box_sizes_float[b, 0] * 0.5) if ax == 0 else (Y[b] + box_sizes_float[b, 1] * 0.5)
                        if (pick_hi and cv == hi_v) or ((not pick_hi) and cv == lo_v):
                            target_b = b; break
                    if target_b != -1:
                        if ax == 0:
                            zx_idx = box_to_zx[target_b]
                            if zx_idx != -1:
                                idx_x = zx_idx
                                cv = X[target_b] + box_sizes_float[target_b, 0] * 0.5
                                coeff = M_x[target_b, zx_idx]
                                if abs(coeff) > 1e-5 and abs(median_v - cv) > 1e-3:
                                    delta = ((median_v - cv) / coeff) * next_rand(rng_state) * 1.2
                                    is_net_aware = True
                        else:
                            zy_idx = box_to_zy[target_b]
                            if zy_idx != -1:
                                idx_y = zy_idx
                                cv = Y[target_b] + box_sizes_float[target_b, 1] * 0.5
                                coeff = M_y[target_b, zy_idx]
                                if abs(coeff) > 1e-5 and abs(median_v - cv) > 1e-3:
                                    delta = ((median_v - cv) / coeff) * next_rand(rng_state) * 1.2
                                    is_net_aware = True
                else:
                    center_x = (nmin_x + nmax_x) * 0.5
                    center_y = (nmin_y + nmax_y) * 0.5
                    rand_m = int(next_rand(rng_state) * bcnt)
                    target_b = net_matrix[net_idx][rand_m]
                    if next_rand(rng_state) < 0.5 and len_x > 0:
                        zx_idx = box_to_zx[target_b]
                        if zx_idx != -1:
                            idx_x = zx_idx
                            dist = center_x - (X[target_b] + box_sizes_float[target_b, 0] * 0.5)
                            coeff = M_x[target_b, zx_idx]
                            if abs(coeff) > 1e-5:
                                delta = (dist / coeff) * next_rand(rng_state) * 1.5
                                is_net_aware = True
                    elif len_y > 0:
                        zy_idx = box_to_zy[target_b]
                        if zy_idx != -1:
                            idx_y = zy_idx
                            dist = center_y - (Y[target_b] + box_sizes_float[target_b, 1] * 0.5)
                        coeff = M_y[target_b, zy_idx]
                        if abs(coeff) > 1e-5:
                            delta = (dist / coeff) * next_rand(rng_state) * 1.5
                            is_net_aware = True

        if not is_net_aware:
            r1 = next_rand(rng_state)
            curr_step = base_step if r1 < 0.7 else base_step * 3.0
            if len_x > 0 and len_y > 0:
                if next_rand(rng_state) < 0.5:
                    idx_x = int(next_rand(rng_state) * len_x)
                    delta = -curr_step + next_rand(rng_state) * (2.0 * curr_step)
                else:
                    idx_y = int(next_rand(rng_state) * len_y)
                    delta = -curr_step + next_rand(rng_state) * (2.0 * curr_step)
            elif len_x > 0:
                idx_x = int(next_rand(rng_state) * len_x)
                delta = -curr_step + next_rand(rng_state) * (2.0 * curr_step)
            elif len_y > 0:
                idx_y = int(next_rand(rng_state) * len_y)
                delta = -curr_step + next_rand(rng_state) * (2.0 * curr_step)
            else: break

        if idx_x != -1:
            S = v2b_x_idx[idx_x]; Vrow = v2b_x_val[idx_x]
        else:
            S = v2b_y_idx[idx_y]; Vrow = v2b_y_val[idx_y]
        k = 0
        while k < S.shape[0] and S[k] != -1:
            k += 1

        net_moved[:] = False
        old_partial_hpwl = 0.0
        for a in range(k):
            b = S[a]
            for j in range(box_to_nets_mat.shape[1]):
                nt = box_to_nets_mat[b, j]
                if nt == -1: break
                net_moved[nt] = True
        for nt in range(num_nets):
            if net_moved[nt]:
                old_partial_hpwl += compute_single_net_hpwl(nt, X, Y, box_sizes_float, net_matrix)

        old_touch = touch_box_overlap(X, Y, x2, y2, N, S, k, in_box)

        if idx_x != -1:
            Z_x[idx_x] += delta
            for a in range(k):
                i = S[a]; X[i] += delta * Vrow[a]; x2[i] = X[i] + box_sizes_float[i, 0]
        else:
            Z_y[idx_y] += delta
            for a in range(k):
                i = S[a]; Y[i] += delta * Vrow[a]; y2[i] = Y[i] + box_sizes_float[i, 1]

        new_partial_hpwl = 0.0
        for nt in range(num_nets):
            if net_moved[nt]:
                new_partial_hpwl += compute_single_net_hpwl(nt, X, Y, box_sizes_float, net_matrix)
        new_hpwl = current_hpwl + (new_partial_hpwl - old_partial_hpwl) * 5e-5

        new_touch = touch_box_overlap(X, Y, x2, y2, N, S, k, in_box)
        new_box_ov = current_box_ov - old_touch + new_touch
        area_new, inst_ov_new = inst_and_area(X, Y, x2, y2, repeat_units, box_to_inst,
                                              scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2)
        new_overlap = new_box_ov + inst_ov_new
        new_cost = w_area * area_new + w_hpwl * new_hpwl + penalty_weight * (new_overlap * 1e-8)

        accept = False
        if phase == 1:
            if new_overlap < current_overlap: accept = True
            else:
                dE = (new_cost - current_cost) / T
                if dE < 70.0 and next_rand(rng_state) < np.exp(-dE): accept = True
        else:
            if new_overlap <= 0.0:
                if current_overlap > 0.0 or new_cost < current_cost: accept = True
                else:
                    dE = (new_cost - current_cost) / T
                    if dE < 70.0 and next_rand(rng_state) < np.exp(-dE): accept = True
            else:
                if new_overlap < current_overlap: accept = True
                else:
                    dE = (new_cost - current_cost) / T
                    if dE < 70.0 and next_rand(rng_state) < np.exp(-dE): accept = True

        if accept:
            current_cost, current_overlap, current_hpwl = new_cost, new_overlap, new_hpwl
            current_box_ov = new_box_ov
            if phase == 1:
                if current_overlap < best_overlap or (abs(current_overlap - best_overlap) < 1e-3 and current_cost < best_cost):
                    best_overlap, best_cost = current_overlap, current_cost
                    best_Z_x[:] = Z_x; best_Z_y[:] = Z_y
            else:
                if current_overlap <= 0.0 and (best_overlap > 0.0 or current_cost < best_cost):
                    best_cost, best_overlap = current_cost, current_overlap
                    best_Z_x[:] = Z_x; best_Z_y[:] = Z_y
                elif best_overlap > 0.0 and current_overlap < best_overlap:
                    best_cost, best_overlap = current_cost, current_overlap
                    best_Z_x[:] = Z_x; best_Z_y[:] = Z_y
        else:
            if idx_x != -1:
                Z_x[idx_x] -= delta
                for a in range(k):
                    i = S[a]; X[i] -= delta * Vrow[a]; x2[i] = X[i] + box_sizes_float[i, 0]
            else:
                Z_y[idx_y] -= delta
                for a in range(k):
                    i = S[a]; Y[i] -= delta * Vrow[a]; y2[i] = Y[i] + box_sizes_float[i, 1]

    return Z_x, Z_y, best_Z_x, best_Z_y, best_cost, best_overlap

@njit
def run_integer_tuning_steps_incr(X, Y, basis_x, basis_y, box_sizes_float, net_matrix,
                                  repeat_units, box_to_inst,
                                  b2b_x_idx, b2b_x_val, b2b_y_idx, b2b_y_val, box_to_nets_mat,
                                  x2, y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved, in_box,
                                  rng_state, w_area, w_hpwl, T, steps):
    N = box_sizes_float.shape[0]
    penalty_weight = 1e12
    for i in range(N):
        x2[i] = X[i] + box_sizes_float[i, 0]
        y2[i] = Y[i] + box_sizes_float[i, 1]
    box_ov = full_box_overlap(X, Y, x2, y2, N)
    area_f, inst_ov = inst_and_area(X, Y, x2, y2, repeat_units, box_to_inst,
                                    scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2)
    current_overlap = box_ov + inst_ov
    current_hpwl = compute_all_hpwl(X, Y, box_sizes_float, net_matrix)
    current_cost = w_area * area_f + w_hpwl * current_hpwl + penalty_weight * (current_overlap * 1e-8)
    current_box_ov = box_ov

    best_X = X.copy(); best_Y = Y.copy()
    best_cost = current_cost; best_overlap = current_overlap
    len_bx = b2b_x_idx.shape[0]
    len_by = b2b_y_idx.shape[0]
    num_nets = net_matrix.shape[0]

    for _ in range(steps):
        r_step = next_rand(rng_state)
        mult = 1 if r_step < 0.7 else (2 if r_step < 0.9 else 3)
        if next_rand(rng_state) < 0.5: mult = -mult

        idx_x, idx_y = -1, -1
        if len_bx > 0 and len_by > 0:
            if next_rand(rng_state) < 0.5:
                idx_x = int(next_rand(rng_state) * len_bx)
            else:
                idx_y = int(next_rand(rng_state) * len_by)
        elif len_bx > 0:
            idx_x = int(next_rand(rng_state) * len_bx)
        elif len_by > 0:
            idx_y = int(next_rand(rng_state) * len_by)
        else: break

        if idx_x != -1:
            S = b2b_x_idx[idx_x]; Vrow = b2b_x_val[idx_x]
        else:
            S = b2b_y_idx[idx_y]; Vrow = b2b_y_val[idx_y]
        k = 0
        while k < S.shape[0] and S[k] != -1:
            k += 1

        net_moved[:] = False
        old_partial_hpwl = 0.0
        for a in range(k):
            b = S[a]
            for j in range(box_to_nets_mat.shape[1]):
                nt = box_to_nets_mat[b, j]
                if nt == -1: break
                net_moved[nt] = True
        for nt in range(num_nets):
            if net_moved[nt]:
                old_partial_hpwl += compute_single_net_hpwl(nt, X, Y, box_sizes_float, net_matrix)

        old_touch = touch_box_overlap(X, Y, x2, y2, N, S, k, in_box)

        if idx_x != -1:
            for a in range(k):
                i = S[a]; X[i] += mult * Vrow[a]; x2[i] = X[i] + box_sizes_float[i, 0]
        else:
            for a in range(k):
                i = S[a]; Y[i] += mult * Vrow[a]; y2[i] = Y[i] + box_sizes_float[i, 1]

        new_partial_hpwl = 0.0
        for nt in range(num_nets):
            if net_moved[nt]:
                new_partial_hpwl += compute_single_net_hpwl(nt, X, Y, box_sizes_float, net_matrix)
        new_hpwl = current_hpwl + (new_partial_hpwl - old_partial_hpwl) * 5e-5

        new_touch = touch_box_overlap(X, Y, x2, y2, N, S, k, in_box)
        new_box_ov = current_box_ov - old_touch + new_touch
        area_new, inst_ov_new = inst_and_area(X, Y, x2, y2, repeat_units, box_to_inst,
                                              scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2)
        new_overlap = new_box_ov + inst_ov_new
        new_cost = w_area * area_new + w_hpwl * new_hpwl + penalty_weight * (new_overlap * 1e-8)

        accept = False
        if new_overlap <= 0.0:
            if current_overlap > 0.0 or new_cost < current_cost: accept = True
            else:
                dE = (new_cost - current_cost) / T
                if dE < 70.0 and next_rand(rng_state) < np.exp(-dE): accept = True
        else:
            if new_overlap < current_overlap: accept = True
            else:
                dE = (new_cost - current_cost) / T
                if dE < 70.0 and next_rand(rng_state) < np.exp(-dE): accept = True

        if accept:
            current_cost, current_overlap, current_hpwl = new_cost, new_overlap, new_hpwl
            current_box_ov = new_box_ov
            if current_overlap <= 0.0 and (best_overlap > 0.0 or current_cost < best_cost):
                best_cost, best_overlap = current_cost, current_overlap
                best_X[:] = X; best_Y[:] = Y
            elif best_overlap > 0.0 and current_overlap < best_overlap:
                best_cost, best_overlap = current_cost, current_overlap
                best_X[:] = X; best_Y[:] = Y
        else:
            if idx_x != -1:
                for a in range(k):
                    i = S[a]; X[i] -= mult * Vrow[a]; x2[i] = X[i] + box_sizes_float[i, 0]
            if idx_y != -1:
                for a in range(k):
                    i = S[a]; Y[i] -= mult * Vrow[a]; y2[i] = Y[i] + box_sizes_float[i, 1]

    return X, Y, best_X, best_Y, best_cost, best_overlap

# ==========================================
# 3. Constraint parametrization helpers
# ==========================================
def parameterize_system(A, b, V):
    if A.size == 0 or A.shape[0] == 0:
        return np.eye(V), np.zeros(V), V
    E, TotalVars = A.shape
    M = np.hstack([A, b.reshape(-1, 1)]).astype(np.float64)
    pivot_rows, pivot_cols, r = [], [], 0
    used_cols = np.zeros(TotalVars, dtype=bool)
    for c in range(TotalVars):
        if r >= E: break
        if used_cols[c]: continue
        col_slice = M[r:, c]
        max_abs = np.abs(col_slice).max()
        if max_abs < 1e-9: continue
        pivot = int(np.argmax(np.abs(col_slice))) + r
        M[[r, pivot]] = M[[pivot, r]]
        pivot_val = M[r, c]
        M[r] /= pivot_val
        for i in range(E):
            if i != r:
                factor = M[i, c]
                if factor != 0: M[i] -= factor * M[r]
        pivot_rows.append(r); pivot_cols.append(c); used_cols[c] = True; r += 1
    free_cols = [c for c in range(TotalVars) if not used_cols[c]]
    num_free = len(free_cols)
    Param_M = np.zeros((TotalVars, num_free))
    Param_C = np.zeros(TotalVars)
    for idx, c in enumerate(free_cols): Param_M[c, idx] = 1.0
    for r_idx, c in zip(pivot_rows, pivot_cols):
        Param_C[c] = M[r_idx, TotalVars]
        row_data = M[r_idx]
        for idx, f in enumerate(free_cols):
            val = -row_data[f]
            if val != 0: Param_M[c, idx] = val
    return Param_M, Param_C, num_free

def fix_integer_layout(A, b, X_float):
    X = np.round(X_float).astype(np.int64)
    if A.shape[0] == 0: return X.astype(np.float64)
    b_int = np.round(b).astype(np.int64)
    A_int = np.round(A).astype(np.int64)
    for _ in range(50000):
        errs = np.dot(A_int, X) - b_int
        bad_indices = np.where(errs != 0)[0]
        if len(bad_indices) == 0: break
        e_idx = bad_indices[np.random.randint(len(bad_indices))]
        err = errs[e_idx]
        nz = np.where(A_int[e_idx] != 0)[0]
        j = nz[np.random.randint(len(nz))]
        coeff = A_int[e_idx, j]
        delta = -1 if err * coeff > 0 else 1
        X[j] += delta
    return X.astype(np.float64)

def get_integer_basis(M):
    if M.size == 0 or M.shape[1] == 0:
        return np.zeros((M.shape[0], 0))
    num_vars, num_free = M.shape
    basis = np.zeros((num_vars, num_free), dtype=np.float64)
    for j in range(num_free):
        col = M[:, j]
        if np.all(np.abs(col - np.round(col)) < 1e-5): basis[:, j] = np.round(col)
        elif np.all(np.abs(2.0 * col - np.round(2.0 * col)) < 1e-5): basis[:, j] = np.round(2.0 * col)
        elif np.all(np.abs(4.0 * col - np.round(4.0 * col)) < 1e-5): basis[:, j] = np.round(4.0 * col)
        else: basis[:, j] = np.round(8.0 * col)
    return basis

def build_var_to_boxes(M):
    if M.shape[1] == 0: return np.zeros((0, 1), dtype=np.int32)
    num_vars, num_free = M.shape
    var_to_boxes_list = [[] for _ in range(num_free)]
    for v in range(num_free):
        for i in range(num_vars):
            if abs(M[i, v]) > 1e-7:
                var_to_boxes_list[v].append(i)
    max_b = max((len(l) for l in var_to_boxes_list), default=0)
    if max_b == 0: return np.full((num_free, 1), -1, dtype=np.int32)
    var_to_boxes_mat = np.full((num_free, max_b), -1, dtype=np.int32)
    for v, l in enumerate(var_to_boxes_list):
        var_to_boxes_mat[v, :len(l)] = l
    return var_to_boxes_mat

def build_var_to_boxes_vals(M):
    """Like build_var_to_boxes but also returns the coefficient M[i,v] for each
    (var, box) so the X update can be sparse: X[i] += delta * coeff[i]."""
    if M.shape[1] == 0:
        return np.zeros((0, 1), dtype=np.int32), np.zeros((0, 1), dtype=np.float64)
    num_vars, num_free = M.shape
    idx_list = [[] for _ in range(num_free)]
    val_list = [[] for _ in range(num_free)]
    for v in range(num_free):
        for i in range(num_vars):
            c = M[i, v]
            if abs(c) > 1e-7:
                idx_list[v].append(i)
                val_list[v].append(c)
    max_b = max((len(l) for l in idx_list), default=0)
    if max_b == 0:
        return np.full((num_free, 1), -1, dtype=np.int32), np.zeros((num_free, 1), dtype=np.float64)
    idx_mat = np.full((num_free, max_b), -1, dtype=np.int32)
    val_mat = np.zeros((num_free, max_b), dtype=np.float64)
    for v in range(num_free):
        idx_mat[v, :len(idx_list[v])] = idx_list[v]
        val_mat[v, :len(val_list[v])] = val_list[v]
    return idx_mat, val_mat

# ==========================================
# 4. Main Solver Class
# ==========================================
class Solution:
    TIME_BUDGET = 117.0
    SEED = 246813579
    # ---- feature flags (optimization A/B) ----
    USE_INCR_OVERLAP = True    # #1: incremental box-overlap + sparse X update (11.9x throughput on N=81)
    USE_SPARSE_NETLIST = False # #6: net_moved as explicit small list (skip O(num_nets) scan)
    RESTART_MULT = 2.0         # #2: multiplier on restart count (faster eval -> more restarts)
    LARGE_N_MULT = 2.0         # #2: extra multiplier for N>60 (-> ~4x, CLAUDE.md sweet spot)
    NET_AWARE_PROB = 0.25      # #3: net-aware move probability (cur_179 default)
    NET_AWARE_MODE = 0         # #3: 0=random->centroid (kept; 1=extreme->median tested, no clear win)
    FEAS_BACKSTOP = True       # #5: keep best overlap=0 legalized snapshot (never output invalid)
    DIAG = False               # instrumentation off for ship

    def solve(self, data):
        start_time = time.time()
        box_sizes_raw = data["box_size"]
        N = len(box_sizes_raw)
        box_sizes_float = np.array([[w * 10000.0, h * 10000.0] for w, h in box_sizes_raw], dtype=np.float64)

        eqs_x, eqs_y = [], []
        constraints = data.get("constraints", {})
        align = data.get("align", constraints.get("align", {}))
        symmetry_x = data.get("symmetry_x", constraints.get("symmetry_x", []))
        symmetry_y = data.get("symmetry_y", constraints.get("symmetry_y", []))
        repeat_groups = data.get("repeat_groups", constraints.get("repeat_groups", []))

        def process_symmetry_constraint(symmetry_data, axis_idx):
            target = eqs_x if axis_idx == 0 else eqs_y
            perp_target = eqs_y if axis_idx == 0 else eqs_x
            perp_axis_idx = 1 if axis_idx == 0 else 0
            for g in symmetry_data:
                items = []
                for p in g.get("symmetry_pair", []): items.append(('pair', p[0]-1, p[1]-1))
                for s in g.get("self_symmetry", []): items.append(('self', s-1))
                for item in items:
                    if item[0] == 'pair':
                        b1, b2 = item[1], item[2]
                        perp_eq = np.zeros(N)
                        perp_eq[b1] = 1.0; perp_eq[b2] = -1.0
                        perp_val = (box_sizes_float[b2, perp_axis_idx] - box_sizes_float[b1, perp_axis_idx]) / 2.0
                        perp_target.append((perp_eq, perp_val))
                if len(items) <= 1: continue
                first = items[0]
                f_eq = np.zeros(N)
                if first[0] == 'pair':
                    f_b1, f_b2 = first[1], first[2]
                    f_eq[f_b1] = 1.0; f_eq[f_b2] = 1.0
                    f_val = (box_sizes_float[f_b1, axis_idx] + box_sizes_float[f_b2, axis_idx]) / 2.0
                else:
                    f_b = first[1]
                    f_eq[f_b] = 2.0
                    f_val = box_sizes_float[f_b, axis_idx]
                for i in range(1, len(items)):
                    curr = items[i]
                    eq = np.zeros(N)
                    if curr[0] == 'pair':
                        c_b1, c_b2 = curr[1], curr[2]
                        eq[c_b1] = 1.0; eq[c_b2] = 1.0
                        c_val = (box_sizes_float[c_b1, axis_idx] + box_sizes_float[c_b2, axis_idx]) / 2.0
                    else:
                        c_b = curr[1]
                        eq[c_b] = 2.0
                        c_val = box_sizes_float[c_b, axis_idx]
                    target.append((eq - f_eq, f_val - c_val))

        process_symmetry_constraint(symmetry_x, 0)
        process_symmetry_constraint(symmetry_y, 1)

        zero_eq = np.zeros(N)
        for pair_list in align.get("left", []):
            for i in range(len(pair_list)-1):
                b1, b2 = pair_list[i]-1, pair_list[i+1]-1
                eq = zero_eq.copy(); eq[b1], eq[b2] = 1.0, -1.0
                eqs_x.append((eq, 0.0))
        for pair_list in align.get("right", []):
            for i in range(len(pair_list)-1):
                b1, b2 = pair_list[i]-1, pair_list[i+1]-1
                eq = zero_eq.copy(); eq[b1], eq[b2] = 1.0, -1.0
                eqs_x.append((eq, float(box_sizes_float[b2,0] - box_sizes_float[b1,0])))
        for pair_list in align.get("bottom", []):
            for i in range(len(pair_list)-1):
                b1, b2 = pair_list[i]-1, pair_list[i+1]-1
                eq = zero_eq.copy(); eq[b1], eq[b2] = 1.0, -1.0
                eqs_y.append((eq, 0.0))
        for pair_list in align.get("top", []):
            for i in range(len(pair_list)-1):
                b1, b2 = pair_list[i]-1, pair_list[i+1]-1
                eq = zero_eq.copy(); eq[b1], eq[b2] = 1.0, -1.0
                eqs_y.append((eq, float(box_sizes_float[b2,1] - box_sizes_float[b1,1])))

        repeat_units_list = []
        box_to_inst_dict = {}
        inst_counter = 0
        for rg in repeat_groups:
            groups = rg.get("groups", [])
            if not groups: continue
            g0_idx = [b-1 for b in groups[0]]
            for k, gk in enumerate(groups):
                gk_idx = [b-1 for b in gk]
                repeat_units_list.append(gk_idx)
                for b in gk: box_to_inst_dict[b-1] = inst_counter
                inst_counter += 1
            for k in range(1, len(groups)):
                gk_idx = [b-1 for b in groups[k]]
                for m in range(1, min(len(g0_idx), len(gk_idx))):
                    b_km, b_k0, b_0m, b_00 = gk_idx[m], gk_idx[0], g0_idx[m], g0_idx[0]
                    eq_x = zero_eq.copy(); eq_x[b_km], eq_x[b_k0], eq_x[b_0m], eq_x[b_00] = 1.0, -1.0, -1.0, 1.0
                    eqs_x.append((eq_x, 0.0))
                    eq_y = zero_eq.copy(); eq_y[b_km], eq_y[b_k0], eq_y[b_0m], eq_y[b_00] = 1.0, -1.0, -1.0, 1.0
                    eqs_y.append((eq_y, 0.0))

        Ax = np.array([e[0] for e in eqs_x]).astype(np.float64) if eqs_x else np.zeros((0,N))
        bx = np.array([e[1] for e in eqs_x]).astype(np.float64) if eqs_x else np.zeros(0)
        Ay = np.array([e[0] for e in eqs_y]).astype(np.float64) if eqs_y else np.zeros((0,N))
        by = np.array([e[1] for e in eqs_y]).astype(np.float64) if eqs_y else np.zeros(0)
        M_x, C_x, num_free_x = parameterize_system(Ax, bx, N)
        M_y, C_y, num_free_y = parameterize_system(Ay, by, N)

        # Build Net-Aware mappings
        nets = data.get("nets", [])
        if nets:
            max_len = max(len(n) for n in nets)
            net_mat = np.full((len(nets), max_len), -1, dtype=np.int32)
            for i, n in enumerate(nets): net_mat[i,:len(n)] = [x-1 for x in n]
        else:
            net_mat = np.zeros((0,1), dtype=np.int32)

        box_to_nets_list = [[] for _ in range(N)]
        for i, n in enumerate(nets):
            for b in n: box_to_nets_list[b-1].append(i)
        max_nets_per_box = max((len(l) for l in box_to_nets_list), default=0)
        if max_nets_per_box == 0:
            box_to_nets_mat = np.full((N, 1), -1, dtype=np.int32)
        else:
            box_to_nets_mat = np.full((N, max_nets_per_box), -1, dtype=np.int32)
            for i, l in enumerate(box_to_nets_list): box_to_nets_mat[i, :len(l)] = l

        box_to_zx = np.full(N, -1, dtype=np.int32)
        box_to_zy = np.full(N, -1, dtype=np.int32)
        for i in range(N):
            if num_free_x > 0:
                best_j_x = np.argmax(np.abs(M_x[i, :]))
                if abs(M_x[i, best_j_x]) > 1e-5: box_to_zx[i] = best_j_x
            if num_free_y > 0:
                best_j_y = np.argmax(np.abs(M_y[i, :]))
                if abs(M_y[i, best_j_y]) > 1e-5: box_to_zy[i] = best_j_y

        v2b_x = build_var_to_boxes(M_x)
        v2b_y = build_var_to_boxes(M_y)

        if repeat_units_list:
            max_len = max(len(u) for u in repeat_units_list)
            rep_units_arr = np.full((len(repeat_units_list), max_len), -1, dtype=np.int32)
            for i, u in enumerate(repeat_units_list): rep_units_arr[i,:len(u)] = u
        else:
            rep_units_arr = np.zeros((0,1), dtype=np.int32)

        box_to_inst_arr = np.full(N, -1, dtype=np.int32)
        for b, inst in box_to_inst_dict.items(): box_to_inst_arr[b] = inst

        basis_x = get_integer_basis(M_x)
        basis_y = get_integer_basis(M_y)
        b2b_x = build_var_to_boxes(basis_x)
        b2b_y = build_var_to_boxes(basis_y)

        # support-with-coefficients arrays for incremental/sparse cores (#1/#6)
        v2b_x_idx, v2b_x_val = build_var_to_boxes_vals(M_x)
        v2b_y_idx, v2b_y_val = build_var_to_boxes_vals(M_y)
        b2b_x_idx, b2b_x_val = build_var_to_boxes_vals(basis_x)
        b2b_y_idx, b2b_y_val = build_var_to_boxes_vals(basis_y)
        in_box = np.zeros(N, dtype=np.int8)
        use_incr = bool(self.USE_INCR_OVERLAP)

        avg_w = np.mean(box_sizes_float[:, 0]) if N > 0 else 0.0
        avg_h = np.mean(box_sizes_float[:, 1]) if N > 0 else 0.0
        base_step_p1 = float(max(avg_w, avg_h) * 4.0)
        base_step_p2 = float(max(avg_w, avg_h) * 0.25)

        # 内存安全池（消除JIT内部动态分配）
        num_inst = len(repeat_units_list) if repeat_units_list else 0
        scratch_x2 = np.zeros(N, dtype=np.float64)
        scratch_y2 = np.zeros(N, dtype=np.float64)
        scratch_ix1 = np.zeros(num_inst, dtype=np.float64)
        scratch_ix2 = np.zeros(num_inst, dtype=np.float64)
        scratch_iy1 = np.zeros(num_inst, dtype=np.float64)
        scratch_iy2 = np.zeros(num_inst, dtype=np.float64)
        net_moved = np.zeros(len(nets), dtype=np.bool_)
        na_scr = np.zeros(max(net_mat.shape[1], 1), dtype=np.float64)
        na_prob = float(self.NET_AWARE_PROB)
        na_mode = int(self.NET_AWARE_MODE)

        # Fast JIT Warmup (compile only the cores that will actually run)
        cur_Zx_d = np.zeros(num_free_x, dtype=np.float64)
        cur_Zy_d = np.zeros(num_free_y, dtype=np.float64)
        rng_dummy = np.array([self.SEED], dtype=np.uint32)
        if use_incr:
            _ = run_sa_steps_float_incr(cur_Zx_d, cur_Zy_d, M_x, C_x, M_y, C_y, box_sizes_float, net_mat, rep_units_arr, box_to_inst_arr,
                                        v2b_x_idx, v2b_x_val, v2b_y_idx, v2b_y_val, box_to_nets_mat, box_to_zx, box_to_zy,
                                        scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved, in_box,
                                        1, 100.0, base_step_p1, rng_dummy, 1.0, 1.0, 10, na_prob, na_mode, na_scr)
            _ = run_integer_tuning_steps_incr(C_x.copy(), C_y.copy(), basis_x, basis_y, box_sizes_float, net_mat, rep_units_arr, box_to_inst_arr,
                                              b2b_x_idx, b2b_x_val, b2b_y_idx, b2b_y_val, box_to_nets_mat,
                                              scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved, in_box,
                                              rng_dummy, 1.0, 1.0, 20.0, 10)
        else:
            _ = run_sa_steps_float(cur_Zx_d, cur_Zy_d, M_x, C_x, M_y, C_y, box_sizes_float, net_mat, rep_units_arr, box_to_inst_arr,
                                   v2b_x, v2b_y, box_to_nets_mat, box_to_zx, box_to_zy,
                                   scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved,
                                   1, 100.0, base_step_p1, rng_dummy, 1.0, 1.0, 10)
            _ = run_integer_tuning_steps(C_x.copy(), C_y.copy(), basis_x, basis_y, box_sizes_float, net_mat, rep_units_arr, box_to_inst_arr,
                                         b2b_x, b2b_y, box_to_nets_mat,
                                         scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved,
                                         rng_dummy, 1.0, 1.0, 20.0, 10)

        constraint_ratio = (num_free_x + num_free_y) / (2.0 * N) if N > 0 else 1.0
        restart_time_target = 25.0 if constraint_ratio > 0.6 else 15.0
        # optimization #2: lower per-restart target -> more restarts (only pays off
        # when paired with the faster incremental eval, otherwise under-converges)
        restart_time_target = restart_time_target / float(self.RESTART_MULT)
        if N > 60:
            restart_time_target = restart_time_target / float(self.LARGE_N_MULT)
        initial_available = self.TIME_BUDGET - (time.time() - start_time)
        n_restarts = max(1, int(initial_available / restart_time_target))

        scale_N = np.sqrt(N / 20.0) if N > 20 else 1.0
        p1_temp_start = 4000.0 * scale_N if constraint_ratio > 0.6 else 1800.0 * scale_N
        p2_temp_start = 30.0 * scale_N if constraint_ratio > 0.6 else 15.0 * scale_N
        p1_w_area, p1_w_hpwl = 0.1, 2.0
        p2_w_area, p2_w_hpwl = 1.0, 10.0

        s = int(self.SEED) & 0xFFFFFFFF
        seeds = [s]
        for _ in range(n_restarts - 1):
            s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
            if s == 0: s = 2463534242
            seeds.append(s)

        g_cost = np.inf
        g_overlap_best = np.inf
        g_X, g_Y = None, None
        _diag = bool(getattr(self, "DIAG", False))
        _d_p1 = _d_p2 = _d_tune = 0.0
        _d_sa = 0
        _d_nr = 0
        if _diag:
            _d_overhead = time.time() - start_time
        # init so the no-restart fallback (g_X is None) is well-defined
        cur_Zx = np.zeros(num_free_x, dtype=np.float64)
        cur_Zy = np.zeros(num_free_y, dtype=np.float64)

        for r in range(n_restarts):
            current_time = time.time()
            time_left = start_time + self.TIME_BUDGET - current_time
            if time_left < 3.0: break
            if _diag: _d_nr += 1

            current_restart_budget = time_left / (n_restarts - r)
            p1_duration = current_restart_budget * 0.45
            p2_duration = current_restart_budget * 0.45
            tune_duration = current_restart_budget * 0.10

            rng = np.array([seeds[r]], dtype=np.uint32)
            cur_Zx = np.zeros(num_free_x, dtype=np.float64)
            cur_Zy = np.zeros(num_free_y, dtype=np.float64)
            for i in range(num_free_x): cur_Zx[i] = (next_rand(rng) - 0.5) * 400000.0
            for i in range(num_free_y): cur_Zy[i] = (next_rand(rng) - 0.5) * 400000.0

            # ---------- Phase 1 ----------
            p1_start = time.time()
            p1_deadline = p1_start + p1_duration
            log_temp_ratio = np.log(1e-1 / p1_temp_start)
            best_Zx_p1 = cur_Zx.copy()
            best_Zy_p1 = cur_Zy.copy()
            best_cost_p1 = np.inf
            best_ov_p1 = np.inf

            while time.time() < p1_deadline:
                elapsed = time.time() - p1_start
                pct = min(elapsed / p1_duration, 1.0)
                T = p1_temp_start * np.exp(log_temp_ratio * pct)
                step_size = base_step_p1 * (1.0 - 0.95 * pct)
                if step_size < 1.0: step_size = 1.0

                if use_incr:
                    cur_Zx, cur_Zy, bZx, bZy, bc, bov = run_sa_steps_float_incr(
                        cur_Zx, cur_Zy, M_x, C_x, M_y, C_y, box_sizes_float, net_mat, rep_units_arr, box_to_inst_arr,
                        v2b_x_idx, v2b_x_val, v2b_y_idx, v2b_y_val, box_to_nets_mat, box_to_zx, box_to_zy,
                        scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved, in_box,
                        1, T, step_size, rng, p1_w_area, p1_w_hpwl, 200, na_prob, na_mode, na_scr)
                else:
                    cur_Zx, cur_Zy, bZx, bZy, bc, bov = run_sa_steps_float(
                        cur_Zx, cur_Zy, M_x, C_x, M_y, C_y, box_sizes_float, net_mat, rep_units_arr, box_to_inst_arr,
                        v2b_x, v2b_y, box_to_nets_mat, box_to_zx, box_to_zy,
                        scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved,
                        1, T, step_size, rng, p1_w_area, p1_w_hpwl, 200)

                if bov < best_ov_p1 or (abs(bov - best_ov_p1) < 1e-3 and bc < best_cost_p1):
                    best_ov_p1, best_cost_p1 = bov, bc
                    best_Zx_p1[:], best_Zy_p1[:] = bZx, bZy
                if _diag: _d_sa += 1
            cur_Zx, cur_Zy = best_Zx_p1, best_Zy_p1
            if _diag: _d_p1 += time.time() - p1_start

            # ---------- Phase 2 ----------
            p2_start = time.time()
            p2_deadline = p2_start + p2_duration
            log_temp_ratio = np.log(1e-3 / p2_temp_start)
            best_Zx_p2 = cur_Zx.copy()
            best_Zy_p2 = cur_Zy.copy()
            best_cost_p2 = np.inf
            best_ov_p2 = np.inf

            while time.time() < p2_deadline:
                elapsed = time.time() - p2_start
                pct = min(elapsed / p2_duration, 1.0)
                T = p2_temp_start * np.exp(log_temp_ratio * pct)
                step_size = base_step_p2 * (1.0 - 0.95 * pct)
                if step_size < 1.0: step_size = 1.0

                if use_incr:
                    cur_Zx, cur_Zy, bZx, bZy, bc, bov = run_sa_steps_float_incr(
                        cur_Zx, cur_Zy, M_x, C_x, M_y, C_y, box_sizes_float, net_mat, rep_units_arr, box_to_inst_arr,
                        v2b_x_idx, v2b_x_val, v2b_y_idx, v2b_y_val, box_to_nets_mat, box_to_zx, box_to_zy,
                        scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved, in_box,
                        2, T, step_size, rng, p2_w_area, p2_w_hpwl, 200, na_prob, na_mode, na_scr)
                else:
                    cur_Zx, cur_Zy, bZx, bZy, bc, bov = run_sa_steps_float(
                        cur_Zx, cur_Zy, M_x, C_x, M_y, C_y, box_sizes_float, net_mat, rep_units_arr, box_to_inst_arr,
                        v2b_x, v2b_y, box_to_nets_mat, box_to_zx, box_to_zy,
                        scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved,
                        2, T, step_size, rng, p2_w_area, p2_w_hpwl, 200)

                if bov <= 0.0 and (best_ov_p2 > 0.0 or bc < best_cost_p2):
                    best_ov_p2, best_cost_p2 = bov, bc
                    best_Zx_p2[:], best_Zy_p2[:] = bZx, bZy
                elif best_ov_p2 > 0.0 and bov < best_ov_p2:
                    best_ov_p2, best_cost_p2 = bov, bc
                    best_Zx_p2[:], best_Zy_p2[:] = bZx, bZy
                if _diag: _d_sa += 1
            cur_Zx, cur_Zy = best_Zx_p2, best_Zy_p2
            if _diag: _d_p2 += time.time() - p2_start

            X_float = M_x @ cur_Zx + C_x
            Y_float = M_y @ cur_Zy + C_y
            X_int = fix_integer_layout(Ax, bx, X_float)
            Y_int = fix_integer_layout(Ay, by, Y_float)

            # optimization #5 (feasibility backstop): the legalized phase-2 snapshot
            # may already be overlap-free even if integer tuning later fails to clear
            # overlap -> capture it as a feasible candidate so output is never invalid.
            if self.FEAS_BACKSTOP:
                for _bi in range(N):
                    scratch_x2[_bi] = X_int[_bi] + box_sizes_float[_bi, 0]
                    scratch_y2[_bi] = Y_int[_bi] + box_sizes_float[_bi, 1]
                _bo = full_box_overlap(X_int, Y_int, scratch_x2, scratch_y2, N)
                _, _io = inst_and_area(X_int, Y_int, scratch_x2, scratch_y2, rep_units_arr, box_to_inst_arr,
                                       scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2)
                if _bo + _io <= 0.0:
                    _a, _ = inst_and_area(X_int, Y_int, scratch_x2, scratch_y2, rep_units_arr, box_to_inst_arr,
                                          scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2)
                    _h = compute_all_hpwl(X_int, Y_int, box_sizes_float, net_mat)
                    _c = _a + 10.0 * _h
                    if g_overlap_best > 0.0 or _c < g_cost:
                        g_cost = _c; g_X = X_int.copy(); g_Y = Y_int.copy(); g_overlap_best = 0.0

            # ---------- Integer tuning ----------
            tune_start = time.time()
            tune_deadline = tune_start + tune_duration
            best_X_tune, best_Y_tune = X_int.copy(), Y_int.copy()
            best_cost_tune = np.inf
            best_ov_tune = np.inf

            while time.time() < tune_deadline:
                elapsed = time.time() - tune_start
                pct = min(elapsed / tune_duration, 1.0)
                T_tune = 20.0 * (0.0001 ** pct)

                if use_incr:
                    X_int, Y_int, bX, bY, bc, bov = run_integer_tuning_steps_incr(
                        X_int, Y_int, basis_x, basis_y, box_sizes_float, net_mat, rep_units_arr, box_to_inst_arr,
                        b2b_x_idx, b2b_x_val, b2b_y_idx, b2b_y_val, box_to_nets_mat,
                        scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved, in_box,
                        rng, p2_w_area, p2_w_hpwl, T_tune, 200)
                else:
                    X_int, Y_int, bX, bY, bc, bov = run_integer_tuning_steps(
                        X_int, Y_int, basis_x, basis_y, box_sizes_float, net_mat, rep_units_arr, box_to_inst_arr,
                        b2b_x, b2b_y, box_to_nets_mat,
                        scratch_x2, scratch_y2, scratch_ix1, scratch_ix2, scratch_iy1, scratch_iy2, net_moved,
                        rng, p2_w_area, p2_w_hpwl, T_tune, 200)

                if bov <= 0.0 and (best_ov_tune > 0.0 or bc < best_cost_tune):
                    best_ov_tune, best_cost_tune = bov, bc
                    best_X_tune[:], best_Y_tune[:] = bX, bY
                elif best_ov_tune > 0.0 and bov < best_ov_tune:
                    best_ov_tune, best_cost_tune = bov, bc
                    best_X_tune[:], best_Y_tune[:] = bX, bY
            if _diag: _d_tune += time.time() - tune_start

            if best_ov_tune <= 0.0:
                if best_cost_tune < g_cost or g_overlap_best > 0.0:
                    g_cost = best_cost_tune
                    g_X, g_Y = best_X_tune.copy(), best_Y_tune.copy()
                    g_overlap_best = 0.0
            else:
                if g_X is None or (g_overlap_best > 0.0 and best_ov_tune < g_overlap_best):
                    g_cost = best_cost_tune
                    g_X, g_Y = best_X_tune.copy(), best_Y_tune.copy()
                    g_overlap_best = best_ov_tune

        if g_X is None:
            g_X, g_Y = M_x @ cur_Zx + C_x, M_y @ cur_Zy + C_y
            g_X, g_Y = fix_integer_layout(Ax, bx, g_X), fix_integer_layout(Ay, by, g_Y)

        positions = [[round(g_X[i] / 10000.0, 4), round(g_Y[i] / 10000.0, 4)] for i in range(N)]
        if _diag:
            _sa_t = _d_p1 + _d_p2
            import sys as _sys
            print("[diag] N=%d ovh=%.1f restarts=%d p1=%.1f p2=%.1f tune=%.1f "
                  "sa_calls=%d steps/s=%.0f" % (
                      N, _d_overhead, _d_nr, _d_p1, _d_p2, _d_tune, _d_sa,
                      _d_sa * 200.0 / max(_sa_t, 1e-9)), file=_sys.stderr, flush=True)
        return {"positions": positions}