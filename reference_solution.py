import numpy as np
import time
from numba import njit

# ==========================================
# 0. 高性能 RNG 引擎
# ==========================================
@njit
def next_rand(rng_state):
    x = rng_state[0]
    x = np.uint32(x ^ (x << 13))
    x = np.uint32(x ^ (x >> 17))
    x = np.uint32(x ^ (x << 5))
    rng_state[0] = x
    return float(x) / 4294967296.0

# ==========================================
# 1. 纯净浮点评估引擎
# ==========================================
@njit(fastmath=True)
def evaluate_layout_float(Z_x, Z_y, M_x, C_x, M_y, C_y, box_sizes_float, net_matrix, repeat_units, rg_bounds, box_to_inst, penalty_weight):
    X = M_x @ Z_x + C_x
    Y = M_y @ Z_y + C_y
    N = box_sizes_float.shape[0]
    x2_arr = np.empty(N, dtype=np.float64)
    y2_arr = np.empty(N, dtype=np.float64)
    for i in range(N):
        x2_arr[i] = X[i] + box_sizes_float[i, 0]
        y2_arr[i] = Y[i] + box_sizes_float[i, 1]

    overlap_val = 0.0
    for i in range(N):
        xi1, xi2 = X[i], x2_arr[i]
        yi1, yi2 = Y[i], y2_arr[i]
        for j in range(i + 1, N):
            xj1, xj2 = X[j], x2_arr[j]
            yj1, yj2 = Y[j], y2_arr[j]
            ix1 = xi1 if xi1 > xj1 else xj1
            ix2 = xi2 if xi2 < xj2 else xj2
            ox = ix2 - ix1
            if ox > 0.0:
                iy1 = yi1 if yi1 > yj1 else yj1
                iy2 = yi2 if yi2 < yj2 else yj2
                oy = iy2 - iy1
                if oy > 0.0:
                    overlap_val += ox * oy

    num_inst = repeat_units.shape[0]
    if num_inst > 0:
        inst_x1 = np.empty(num_inst, dtype=np.float64)
        inst_x2 = np.empty(num_inst, dtype=np.float64)
        inst_y1 = np.empty(num_inst, dtype=np.float64)
        inst_y2 = np.empty(num_inst, dtype=np.float64)
        for k in range(num_inst):
            min_x = X[repeat_units[k][0]]
            max_x = x2_arr[repeat_units[k][0]]
            min_y = Y[repeat_units[k][0]]
            max_y = y2_arr[repeat_units[k][0]]
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
            if box_to_inst[i] != -1: continue
            xi1, xi2 = X[i], x2_arr[i]
            yi1, yi2 = Y[i], y2_arr[i]
            for k in range(num_inst):
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

    hpwl_f = 0.0
    for net_idx in range(net_matrix.shape[0]):
        first = net_matrix[net_idx][0]
        nmin_x = nmax_x = X[first] * 2.0 + box_sizes_float[first, 0]
        nmin_y = nmax_y = Y[first] * 2.0 + box_sizes_float[first, 1]
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
            hpwl_f += (nmax_x - nmin_x) + (nmax_y - nmin_y)
    hpwl_f *= 5e-5
    cost = area_f + 10.0 * hpwl_f + penalty_weight * (overlap_val * 1e-8)
    return cost, overlap_val

@njit
def run_sa_block_float(init_Z_x, init_Z_y, M_x, C_x, M_y, C_y, box_sizes_float, net_matrix, repeat_units, rg_bounds, box_to_inst,
                       iterations, phase, temp_start, temp_end, rng_state):
    Z_x = init_Z_x.copy()
    Z_y = init_Z_y.copy()
    best_Z_x = Z_x.copy()
    best_Z_y = Z_y.copy()
    penalty_weight = 1e6 if phase == 1 else 1e12
    current_cost, current_overlap = evaluate_layout_float(Z_x, Z_y, M_x, C_x, M_y, C_y, box_sizes_float,
                                                          net_matrix, repeat_units, rg_bounds, box_to_inst, penalty_weight)
    best_cost, best_overlap = current_cost, current_overlap
    T = temp_start
    inv_iter = 1.0 / iterations if iterations > 1 else 0.0
    log_alpha = np.log(temp_end / temp_start) * inv_iter if iterations > 1 else np.log(0.99)
    base_step = 80000.0 if phase == 1 else 5000.0
    len_x = len(Z_x)
    len_y = len(Z_y)
    for step in range(iterations):
        idx_x, idx_y = -1, -1
        old_val_x, old_val_y = 0.0, 0.0
        r1 = next_rand(rng_state)
        curr_step = base_step if r1 < 0.7 else base_step * 3.0
        r2 = next_rand(rng_state)
        if r2 < 0.5 and len_x > 0:
            idx_x = int(next_rand(rng_state) * len_x)
            old_val_x = Z_x[idx_x]
            delta = -curr_step + next_rand(rng_state) * (2.0 * curr_step)
            Z_x[idx_x] += delta
        elif len_y > 0:
            idx_y = int(next_rand(rng_state) * len_y)
            old_val_y = Z_y[idx_y]
            delta = -curr_step + next_rand(rng_state) * (2.0 * curr_step)
            Z_y[idx_y] += delta
        else:
            continue
        new_cost, new_overlap = evaluate_layout_float(Z_x, Z_y, M_x, C_x, M_y, C_y, box_sizes_float,
                                                      net_matrix, repeat_units, rg_bounds, box_to_inst, penalty_weight)
        accept = False
        if phase == 1:
            if new_overlap < current_overlap:
                accept = True
            else:
                dE = (new_cost - current_cost) / T
                if dE < 70.0 and next_rand(rng_state) < np.exp(-dE):
                    accept = True
        else:
            if new_overlap <= 0.0:
                if new_cost < current_cost:
                    accept = True
                else:
                    dE = (new_cost - current_cost) / T
                    if dE < 70.0 and next_rand(rng_state) < np.exp(-dE):
                        accept = True
        if accept:
            current_cost, current_overlap = new_cost, new_overlap
            if phase == 1:
                if current_overlap < best_overlap or (abs(current_overlap - best_overlap) < 1e-3 and current_cost < best_cost):
                    best_overlap, best_cost, best_Z_x, best_Z_y = current_overlap, current_cost, Z_x.copy(), Z_y.copy()
            else:
                if current_overlap <= 0.0 and new_cost < best_cost:
                    best_cost, best_overlap, best_Z_x, best_Z_y = new_cost, current_overlap, Z_x.copy(), Z_y.copy()
        else:
            if idx_x != -1: Z_x[idx_x] = old_val_x
            if idx_y != -1: Z_y[idx_y] = old_val_y
        T *= np.exp(log_alpha)
    return best_Z_x, best_Z_y, best_cost, best_overlap

# ==========================================
# 2. 约束参数化层
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
                if factor != 0:
                    M[i] -= factor * M[r]
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

# ==========================================
# 3. 主求解器
# ==========================================
class Solution:
    def __init__(self):
        pass

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
            for g in symmetry_data:
                items = []
                for p in g.get("symmetry_pair", []): items.append(('pair', p[0]-1, p[1]-1))
                for s in g.get("self_symmetry", []): items.append(('self', s-1))
                if len(items) <= 1: continue
                first = items[0]
                f_eq = np.zeros(N)
                if first[0] == 'pair':
                    f_b1, f_b2 = first[1], first[2]
                    f_eq[f_b1] = 2.0; f_eq[f_b2] = 2.0
                    f_val = float(box_sizes_float[f_b1, axis_idx] + box_sizes_float[f_b2, axis_idx])
                else:
                    f_b = first[1]
                    f_eq[f_b] = 4.0
                    f_val = float(2.0 * box_sizes_float[f_b, axis_idx])
                for i in range(1, len(items)):
                    curr = items[i]
                    eq = np.zeros(N)
                    if curr[0] == 'pair':
                        c_b1, c_b2 = curr[1], curr[2]
                        eq[c_b1] = 2.0; eq[c_b2] = 2.0
                        c_val = float(box_sizes_float[c_b1, axis_idx] + box_sizes_float[c_b2, axis_idx])
                    else:
                        c_b = curr[1]
                        eq[c_b] = 4.0
                        c_val = float(2.0 * box_sizes_float[c_b, axis_idx])
                    target.append((eq - f_eq, f_val - c_val))
                for item in items:
                    if item[0] == 'pair':
                        b1, b2 = item[1], item[2]
                        perp_eq = np.zeros(N)
                        perp_eq[b1] = 1.0
                        perp_eq[b2] = -1.0
                        perp_target.append((perp_eq, 0.0))

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
        rg_bounds = []
        inst_counter = 0
        for rg in repeat_groups:
            groups = rg.get("groups", [])
            if not groups: continue
            g0_idx = [b-1 for b in groups[0]]
            M_len = len(g0_idx)
            start = inst_counter
            for k, gk in enumerate(groups):
                gk_idx = [b-1 for b in gk]
                repeat_units_list.append(gk_idx)
                for b in gk: box_to_inst_dict[b-1] = inst_counter
                inst_counter += 1
            end = inst_counter
            if end > start + 1:
                rg_bounds.append((start, end))
            for k in range(1, len(groups)):
                gk_idx = [b-1 for b in groups[k]]
                common = min(M_len, len(gk_idx))
                for m in range(1, common):
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

        nets = data.get("nets", [])
        if nets:
            max_len = max(len(n) for n in nets)
            net_mat = np.full((len(nets), max_len), -1, dtype=np.int32)
            for i, n in enumerate(nets): net_mat[i,:len(n)] = [x-1 for x in n]
        else:
            net_mat = np.zeros((0,1), dtype=np.int32)

        if repeat_units_list:
            max_len = max(len(u) for u in repeat_units_list)
            rep_units_arr = np.full((len(repeat_units_list), max_len), -1, dtype=np.int32)
            for i, u in enumerate(repeat_units_list): rep_units_arr[i,:len(u)] = u
        else:
            rep_units_arr = np.zeros((0,1), dtype=np.int32)
        rg_bounds_arr = np.array(rg_bounds, dtype=np.int32) if rg_bounds else np.zeros((0,2), dtype=np.int32)
        box_to_inst_arr = np.full(N, -1, dtype=np.int32)
        for b, inst in box_to_inst_dict.items(): box_to_inst_arr[b] = inst

        rng = np.array([246813579], dtype=np.uint32)
        Z_x0 = np.zeros(num_free_x, dtype=np.float64)
        Z_y0 = np.zeros(num_free_y, dtype=np.float64)
        for i in range(num_free_x): Z_x0[i] = (next_rand(rng)-0.5)*400000.0
        for i in range(num_free_y): Z_y0[i] = (next_rand(rng)-0.5)*400000.0
        best_Zx, best_Zy = Z_x0.copy(), Z_y0.copy()

        while time.time()-start_time < 35.0:
            best_Zx, best_Zy, _, ov = run_sa_block_float(
                best_Zx, best_Zy, M_x, C_x, M_y, C_y, box_sizes_float,
                net_mat, rep_units_arr, rg_bounds_arr, box_to_inst_arr,
                220000, 1, 4000.0, 1e-1, rng)
            if ov <= 0.0: break
        while time.time()-start_time < 111.0:
            best_Zx, best_Zy, _, ov = run_sa_block_float(
                best_Zx, best_Zy, M_x, C_x, M_y, C_y, box_sizes_float,
                net_mat, rep_units_arr, rg_bounds_arr, box_to_inst_arr,
                300000, 2, 30.0, 1e-3, rng)

        Xc = M_x @ best_Zx + C_x
        Yc = M_y @ best_Zy + C_y
        min_xc, min_yc = np.min(Xc), np.min(Yc)
        pos = [[round((Xc[i] - min_xc)/10000.0, 4),
                round((Yc[i] - min_yc)/10000.0, 4)] for i in range(N)]

        w = box_sizes_raw
        for _ in range(3):
            for g in symmetry_x:
                pairs = g.get("symmetry_pair", [])
                selfs = g.get("self_symmetry", [])
                if not pairs and not selfs: continue
                centers = []
                for (a,b) in pairs:
                    centers.append(pos[a-1][0] + w[a-1][0]/2)
                    centers.append(pos[b-1][0] + w[b-1][0]/2)
                for s in selfs:
                    centers.append(pos[s-1][0] + w[s-1][0]/2)
                axis = round(float(np.mean(centers)), 4)
                for (a,b) in pairs:
                    target_sum = round(2*axis - (w[a-1][0] + w[b-1][0])/2, 4)
                    d = target_sum - (pos[a-1][0] + pos[b-1][0])
                    pos[a-1][0] = round(pos[a-1][0] + d/2, 4)
                    pos[b-1][0] = round(target_sum - pos[a-1][0], 4)
                    ymean = round((pos[a-1][1] + pos[b-1][1])/2, 4)
                    pos[a-1][1] = pos[b-1][1] = ymean
                for s in selfs:
                    pos[s-1][0] = round(axis - w[s-1][0]/2, 4)

            for g in symmetry_y:
                pairs = g.get("symmetry_pair", [])
                selfs = g.get("self_symmetry", [])
                if not pairs and not selfs: continue
                centers = []
                for (a,b) in pairs:
                    centers.append(pos[a-1][1] + w[a-1][1]/2)
                    centers.append(pos[b-1][1] + w[b-1][1]/2)
                for s in selfs:
                    centers.append(pos[s-1][1] + w[s-1][1]/2)
                axis = round(float(np.mean(centers)), 4)
                for (a,b) in pairs:
                    target_sum = round(2*axis - (w[a-1][1] + w[b-1][1])/2, 4)
                    d = target_sum - (pos[a-1][1] + pos[b-1][1])
                    pos[a-1][1] = round(pos[a-1][1] + d/2, 4)
                    pos[b-1][1] = round(target_sum - pos[a-1][1], 4)
                    xmean = round((pos[a-1][0] + pos[b-1][0])/2, 4)
                    pos[a-1][0] = pos[b-1][0] = xmean
                for s in selfs:
                    pos[s-1][1] = round(axis - w[s-1][1]/2, 4)

            for pair_list in align.get("left", []):
                if not pair_list: continue
                base = pos[pair_list[0]-1][0]
                for b in pair_list[1:]: pos[b-1][0] = base
            for pair_list in align.get("right", []):
                if not pair_list: continue
                base_idx = pair_list[0]-1
                base = round(pos[base_idx][0] + w[base_idx][0], 4)
                for b in pair_list[1:]: pos[b-1][0] = round(base - w[b-1][0], 4)
            for pair_list in align.get("bottom", []):
                if not pair_list: continue
                base = pos[pair_list[0]-1][1]
                for b in pair_list[1:]: pos[b-1][1] = base
            for pair_list in align.get("top", []):
                if not pair_list: continue
                base_idx = pair_list[0]-1
                base = round(pos[base_idx][1] + w[base_idx][1], 4)
                for b in pair_list[1:]: pos[b-1][1] = round(base - w[b-1][1], 4)

            for rg in repeat_groups:
                groups = rg.get("groups", [])
                if not groups: continue
                g0_idx = [b-1 for b in groups[0]]
                for k in range(1, len(groups)):
                    gk_idx = [b-1 for b in groups[k]]
                    dx = pos[gk_idx[0]][0] - pos[g0_idx[0]][0]
                    dy = pos[gk_idx[0]][1] - pos[g0_idx[0]][1]
                    for m in range(1, min(len(g0_idx), len(gk_idx))):
                        pos[gk_idx[m]][0] = round(pos[g0_idx[m]][0] + dx, 4)
                        pos[gk_idx[m]][1] = round(pos[g0_idx[m]][1] + dy, 4)

        minx = min(p[0] for p in pos)
        miny = min(p[1] for p in pos)
        if minx < 0.0 or miny < 0.0:
            sx = -minx if minx < 0.0 else 0.0
            sy = -miny if miny < 0.0 else 0.0
            for i in range(N):
                pos[i][0] = round(pos[i][0] + sx, 4)
                pos[i][1] = round(pos[i][1] + sy, 4)

        return {"box_position": pos}
