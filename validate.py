"""Local validator + cost scorer matching the contest spec in task.md.

Usage:
    python3 validate.py <case.json> [<solver_module:Class>]
    # default solver: reference_solution:Solution

As a module:
    from validate import evaluate
    res = evaluate(data, positions)   # -> dict with valid/cost/violations
"""
import json
import sys
import math


def _centers(box_size, pos):
    return [(pos[i][0] + box_size[i][0] / 2.0, pos[i][1] + box_size[i][1] / 2.0)
            for i in range(len(box_size))]


def evaluate(data, pos, *, lin_tol=1e-3, eq_tol=2e-3, verbose=False):
    """Validate a layout and compute Cost = Area + 10*HPWL.

    lin_tol: linear overlap below this (per axis) is treated as a touch (OK).
    eq_tol:  symmetry/align/repeat equalities must hold within this.
    Returns dict: {valid, cost, area, hpwl, violations:[...]}.
    """
    box_size = data["box_size"]
    N = len(box_size)
    viol = []

    if len(pos) != N:
        return {"valid": False, "cost": float("inf"), "area": 0, "hpwl": 0,
                "violations": [f"position count {len(pos)} != {N}"]}

    x1 = [pos[i][0] for i in range(N)]
    y1 = [pos[i][1] for i in range(N)]
    x2 = [pos[i][0] + box_size[i][0] for i in range(N)]
    y2 = [pos[i][1] + box_size[i][1] for i in range(N)]

    # ---- 1) non-overlap (box vs box) ----
    n_ov = 0
    for i in range(N):
        for j in range(i + 1, N):
            ox = min(x2[i], x2[j]) - max(x1[i], x1[j])
            oy = min(y2[i], y2[j]) - max(y1[i], y1[j])
            if ox > lin_tol and oy > lin_tol:
                n_ov += 1
                if n_ov <= 5:
                    viol.append(f"overlap boxes {i+1},{j+1}: ox={ox:.4f} oy={oy:.4f}")
    if n_ov > 5:
        viol.append(f"... {n_ov} box-box overlaps total")

    cx = [x1[i] + box_size[i][0] / 2.0 for i in range(N)]
    cy = [y1[i] + box_size[i][1] / 2.0 for i in range(N)]

    constraints = data.get("constraints", {})
    align = data.get("align", constraints.get("align", {})) or {}
    symx = data.get("symmetry_x", constraints.get("symmetry_x", [])) or []
    symy = data.get("symmetry_y", constraints.get("symmetry_y", [])) or []
    repeat_groups = data.get("repeat_groups", constraints.get("repeat_groups", [])) or []

    # ---- 2) symmetry ----
    def check_sym(groups, c_par, c_perp, label):
        for gi, g in enumerate(groups):
            pairs = g.get("symmetry_pair", []) or []
            selfs = g.get("self_symmetry", []) or []
            axes = []
            for (a, b) in pairs:
                axes.append((c_par[a - 1] + c_par[b - 1]) / 2.0)
            for s in selfs:
                axes.append(c_par[s - 1])
            if not axes:
                continue
            axis = sum(axes) / len(axes)
            for (a, b) in pairs:
                if abs((c_par[a - 1] + c_par[b - 1]) / 2.0 - axis) > eq_tol:
                    viol.append(f"{label} grp{gi} pair {a},{b} axis off")
                if abs(c_perp[a - 1] - c_perp[b - 1]) > eq_tol:
                    viol.append(f"{label} grp{gi} pair {a},{b} perp mismatch")
            for s in selfs:
                if abs(c_par[s - 1] - axis) > eq_tol:
                    viol.append(f"{label} grp{gi} self {s} off-axis")

    check_sym(symx, cx, cy, "symX")
    check_sym(symy, cy, cx, "symY")

    # ---- 3) align ----
    def check_align(groups, edge, label):
        for grp in groups or []:
            if len(grp) < 2:
                continue
            vals = [edge(g - 1) for g in grp]
            if max(vals) - min(vals) > eq_tol:
                viol.append(f"align {label} {grp} spread={max(vals)-min(vals):.4f}")

    check_align(align.get("left", []), lambda i: x1[i], "left")
    check_align(align.get("right", []), lambda i: x2[i], "right")
    check_align(align.get("bottom", []), lambda i: y1[i], "bottom")
    check_align(align.get("top", []), lambda i: y2[i], "top")

    # ---- 4) repeat groups: equal relative positions + outline non-overlap ----
    outlines = []
    for rgi, rg in enumerate(repeat_groups):
        groups = rg.get("groups", []) or []
        if not groups:
            continue
        g0 = [b - 1 for b in groups[0]]
        for k in range(1, len(groups)):
            gk = [b - 1 for b in groups[k]]
            common = min(len(g0), len(gk))
            for m in range(1, common):
                rel0 = (pos[g0[m]][0] - pos[g0[0]][0], pos[g0[m]][1] - pos[g0[0]][1])
                relk = (pos[gk[m]][0] - pos[gk[0]][0], pos[gk[m]][1] - pos[gk[0]][1])
                if abs(rel0[0] - relk[0]) > eq_tol or abs(rel0[1] - relk[1]) > eq_tol:
                    viol.append(f"repeat grp{rgi} combo{k} member{m} rel pos mismatch")
        for k, gk_raw in enumerate(groups):
            gk = [b - 1 for b in gk_raw]
            ox1 = min(x1[i] for i in gk); oy1 = min(y1[i] for i in gk)
            ox2 = max(x2[i] for i in gk); oy2 = max(y2[i] for i in gk)
            outlines.append((rgi, k, set(gk), ox1, oy1, ox2, oy2))

    # outline vs non-member box
    for (rgi, k, members, ox1, oy1, ox2, oy2) in outlines:
        for i in range(N):
            if i in members:
                continue
            ovx = min(ox2, x2[i]) - max(ox1, x1[i])
            ovy = min(oy2, y2[i]) - max(oy1, y1[i])
            if ovx > lin_tol and ovy > lin_tol:
                viol.append(f"repeat outline grp{rgi}/{k} overlaps box {i+1}")
                break
    # outline vs outline
    for a in range(len(outlines)):
        for b in range(a + 1, len(outlines)):
            _, _, _, ax1, ay1, ax2, ay2 = outlines[a]
            _, _, _, bx1, by1, bx2, by2 = outlines[b]
            ovx = min(ax2, bx2) - max(ax1, bx1)
            ovy = min(ay2, by2) - max(ay1, by1)
            if ovx > lin_tol and ovy > lin_tol:
                viol.append(f"repeat outline {a} overlaps outline {b}")

    # ---- cost ----
    area = (max(x2) - min(x1)) * (max(y2) - min(y1))
    hpwl = 0.0
    for net in data.get("nets", []) or []:
        if len(net) < 2:
            continue
        xs = [cx[b - 1] for b in net]
        ys = [cy[b - 1] for b in net]
        hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))
    cost = area + 10.0 * hpwl

    return {"valid": len(viol) == 0, "cost": cost, "area": area, "hpwl": hpwl,
            "violations": viol}


def _load_solver(spec):
    mod_name, _, cls_name = spec.partition(":")
    cls_name = cls_name or "Solution"
    mod = __import__(mod_name)
    return getattr(mod, cls_name)


def get_positions(out):
    """Solvers in this repo return either {"box_position": ...} (reference,
    solution.py) or {"positions": ...} (cur_171.py). Accept both."""
    if "box_position" in out:
        return out["box_position"]
    return out["positions"]


if __name__ == "__main__":
    case = sys.argv[1]
    solver_spec = sys.argv[2] if len(sys.argv) > 2 else "reference_solution:Solution"
    data = json.load(open(case))
    Solver = _load_solver(solver_spec)
    import time
    t = time.time()
    out = Solver().solve(data)
    elapsed = time.time() - t
    res = evaluate(data, get_positions(out))
    print(f"case={case}  solver={solver_spec}  time={elapsed:.1f}s")
    print(f"  valid={res['valid']}  cost={res['cost']:.2f}  area={res['area']:.2f}  hpwl={res['hpwl']:.2f}")
    if res["violations"]:
        for v in res["violations"][:15]:
            print("   !", v)
