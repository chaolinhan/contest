# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An **algorithm-contest workspace** for a 2D rectangle-layout problem. The goal is to place N rectangles (N ≈ 20–81, no rotation) so that all constraints hold while minimizing `Cost = Area + 10 * HPWL`. Per-case limit: **120 s, single thread**. Any constraint violation or timeout scores that case **0**. See `task.md` for the full spec (in Chinese).

- `solution_p2.py` — **current best (ship this).** = `cur_171.py` + four stacked changes, each measured:
  1. **Incremental box-overlap eval** in *both* the float SA and the integer fine-tune (overlap maintained per-move over the moved support boxes; area/HPWL/instance recomputed). ~1.5–3× faster, biggest on large N. Integer stage stays exact (int coords → overlaps are exact integers <2^52), so validity is unchanged; verified bit-identical (`test_tuning_equiv.py`).
  2. **More restarts** from the freed time — N-aware multiplier, `LARGE_N_MULT=4` for N>60. (Lesson: a faster stage alone does nothing; restart count is schedule-fixed, so freed time is wasted unless you raise the count.)
  3. **Early-advance restarts for N≤60** (`run` advances to the next seed after `STALL_LIMIT` non-improving phase-2 blocks). Gives each mid/small case as many *fully-converged* restarts as fit. Gated off for N>60 (large N wants equal-length fixed slices; early-advance over-shortens or under-fills them).
  4. **Seeded `np.random`** in `solve` — `fix_integer_layout` used unseeded `np.random`, making every solve (incl. `cur_171`) non-deterministic by a few % run-to-run. Seeding gives reproducible submissions and clean A/B.
  5. **Incremental HPWL** in the float SA (`run_sa_block_incr` now maintains per-net x/y spans + a running `hpwl_raw`, updating only the nets touched by each move via `recompute_touched_nets` — `compute_net_spans`/`commit_touched_nets`/`recompute_touched_nets` are the primitives; `area`/instance-overlap stay full-recompute). Measured **1.1–1.4× faster** SA (biggest on net-heavy case9). Because the N≤60 path is a *time-bounded* restart loop, this speedup auto-translates to **more restarts → best-of-more** (monotone, can't regress); case9 (fixed-slice) gets deeper per-restart convergence. Float HPWL drifts ~1e-4 from the full recompute (penalty-amplified on infeasible snapshots) — fine, the float SA only finds basins and the integer stage is exact. Verified `test_sa_incr.py` (feasible snapshots match to ~1e-13).
  Clean seeded result vs `cur_171`: **N=81 (case9) −4.8…−11.2%** (avg ~−8.8%, biggest on the highest-cost cases); mid/small-N net-positive and now without the fixed-slice regressions (early-advance recovers case4_0 / case2_1 / case5_1). Returns both `positions` and `box_position`. Knobs: `LARGE_N_MULT`, `STALL_LIMIT`, `USE_EARLY_ADVANCE`, `RESTART_MULT`, `TUNE_ITERS`, `USE_GREEDY_POLISH`.
- `cur_171.py` — scored **171** on the hidden set (collaborator + earlier work). Constraint-parametrization + 3-stage-per-restart (float SA phase1/2 → integer legalize → integer fine-tune) + multi-seed restarts. The integer stage (`fix_integer_layout` + `run_integer_tuning`, exact ×10000 integer math) guarantees validity, so the float SA is only a basin-finder.
- `solution_p1.py` — the iteration before `solution_p2` (added incremental eval + compaction probe). Superseded by p2; kept only because `probe_initscale.py` targets it.
- `solution.py` — my earlier multi-seed-restart solver (beat `reference` but is superseded by `cur_171`/`solution_p2`).
- `reference_solution.py` — original baseline (scored 109). Single fixed SA chain.
- `test_incremental.py` — proves the incremental eval matches the full evaluator to ~1e-14 (run before trusting any eval change). Sibling tests added with the HPWL work: `test_hpwl_incr.py` (incremental HPWL == full, ~1e-15), `test_sa_incr.py` (the SA's incremental HPWL+overlap state matches the full evaluator on feasible snapshots), `test_integer_optimize.py` (the greedy-ILS integer stage is non-worsening + snapshot-consistent).
- `validate.py` — local validator + Cost scorer matching `task.md` (overlap, symmetry, align, repeat-group constraints; `Cost = Area + 10*HPWL`). Use as a module (`evaluate(data, positions)`) or CLI.
- `benchmark.py` — run a solver over cases, report per-case validity + Cost and totals: `.venv/bin/python benchmark.py solution:Solution [budget] [case_substr]`.
- `probe_converge.py` / `probe_seeds.py` — experiments that established the two key facts (SA converges in ~8 s; seed variance is 14–25 %). `probe_initscale.py` sweeps `solution_p1.INIT_SCALE` (established the "area is constraint-limited, not search-limited" finding).
- **Runner / knob-sweep scripts** (each targets `solution_p2`, overrides one class attr, runs `solve` + `validate` per case — keep `python -u`):
  - `run_cases.py <module:Class> <c1,c2> [budget]` — generic single-solver runner over named cases.
  - `run_fair.py <module:Class> <c1,c2> [budget] [seed]` — same, but **seeds `np.random` before each solve** so the (otherwise unseeded) integer legalizer produces identical randomness across solvers. **Use this for any A/B between `cur_171` and `solution_p2`** — `solution_p2` self-seeds inside `solve`, so a plain `run_cases.py` comparison is confounded by different legalizer dice.
  - `sweep_largeN.py <case> <m1,m2> [budget]` → `LARGE_N_MULT` · `sweep_stall.py <c1,c2> <s1,s2> [budget]` → `STALL_LIMIT` (+ forces `USE_EARLY_ADVANCE=True`) · `run_p2.py <mult> <substr> [budget]` → `RESTART_MULT` over a case substring. These are how the knob values cited above (e.g. `LARGE_N_MULT=4`) were picked.
- `data/case_in/*.json` — 60 test cases in 6 groups (`case0/1/2/4/5/9`), each group has a distinct constraint profile (see below).
- `task.md` — problem statement, scoring, and the **library allowlist** (numpy, numba, scipy, ortools, mip, networkx, … — only these are available at judging).

There is **no build, lint, or test harness, and no git**. The contest judge imports a `Solution` class and calls `solve(data)`.

**Local env:** numba 0.65 has no wheel for the system Python 3.14, so a `.venv` (Python 3.12, numpy 2.2.6 / numba 0.65.1 / scipy 1.16.3) was created with `uv`. Always run via `.venv/bin/python`. Creating venvs / installing needs the sandbox disabled (cache + network).

## Solver contract

`Solution.solve(data) -> {"box_position": [[x, y], ...]}` where `(x, y)` is each rectangle's **bottom-left corner**, in input order.

Input `data` keys: `box_size` ([w,h] per box), `symmetry_x`, `symmetry_y`, `align`, `repeat_groups`, `nets`, `start_from_1`. **All indices in constraints/nets are 1-based** (`start_from_1: True`); subtract 1 before indexing arrays. HPWL uses rectangle **center** points; Area is the global bounding box.

Quick local check of a case:
```bash
python3 -c "import json; from reference_solution import Solution; print(Solution().solve(json.load(open('data/case_in/case0_0_tst.json'))))"
```
Note: the first call pays numba JIT compilation cost (several seconds); the `solve` time budget is wall-clock and self-managed inside `solve`.

## Architecture (`solution.py`, shared with `reference_solution.py`)

The central idea: **equality constraints are satisfied by construction, overlap is a soft penalty, and simulated annealing (SA) optimizes only the free degrees of freedom.** `solution.py` keeps items 1–3 and 5 byte-identical to the reference (so constraint handling and the validity guarantee are unchanged) and only replaces the SA *driver* (item 4).

1. **Constraint → linear system** (`Solution.solve`, lines ~246–344). Symmetry, align, and repeat-group "same relative position" constraints are all *linear equalities* in box coordinates. They are accumulated as `(equation_row, rhs)` into separate x- and y-axis systems (`eqs_x`, `eqs_y`). Symmetry pairs additionally force equal perpendicular coordinates.

2. **Parametrization** (`parameterize_system`). Runs Gaussian elimination on each axis system to produce `X = M_x @ Z_x + C_x` (and similarly y), where `Z` are the **free variables**. SA then perturbs `Z` only — every sampled layout automatically satisfies all equality constraints. This is why non-overlap is the only "hard" thing left to enforce.

3. **Cost evaluation** (`evaluate_layout_float`, `@njit fastmath`). Computes Area + 10·HPWL plus a large overlap penalty. Overlap includes box–box, box–repeat-instance, and instance–instance (repeat groups are treated as rigid outline boxes). Coordinates are scaled ×10000 internally for float precision.

4. **Two-phase SA** (`run_sa_block_float`, `@njit`). Phase 1 (`penalty_weight 1e6`, large steps) drives **overlap → 0**; phase 2 (`penalty_weight 1e12`, small steps) only accepts overlap-free moves and minimizes Cost. RNG is a numba xorshift (`next_rand`).
   - `reference_solution.py`: one fixed SA chain — phase 1 until ~35 s, then phase 2 until ~111 s.
   - `solution.py`: a **multi-seed restart driver**. The key measured facts: a single chain *converges to within ~0.05 % of its basin optimum in ~8 s* and then stagnates for the remaining ~100 s, while *different seeds reach basins 14–25 % apart in Cost* (and the old hard-coded seed is often a poor one). So the budget is split into N independent restarts (12/10/8/7 by box count, each ≥ ~9 s ≥ convergence time), each with a distinct seed, keeping the **global-best overlap-free** layout. Best-of-N can't regress below a single run and usually wins big. `TIME_BUDGET=110` (not 118) leaves margin: a phase-2 block can overshoot the deadline ~1 s and the judge limit is a hard 120 s.

5. **Final snapping** (lines ~396–477). After SA, positions are re-projected to *exactly* satisfy symmetry/align/repeat constraints (SA's float results drift slightly), then shifted so all coordinates are ≥ 0. Output rounded to 4 decimals. **This pass is what guarantees a non-zero score — never remove it.** (Note: it can in principle re-introduce a hair of overlap, so validity is always checked *post-snap* via `validate.py`.)

## Working notes

- **Numba is load-bearing.** The hot loops (`evaluate_layout_float`, `run_sa_block_float`, `next_rand`) are `@njit`. Keep them numba-compatible (typed numpy arrays, no Python objects, `-1`-padded int32 matrices for ragged structures like `nets`/`repeat_units`). Pre-pad and pass arrays in; don't allocate Python lists inside jitted code.
- **The SA converges fast then wastes time — spend it on restart diversity, not a longer chain.** This is the single biggest lever found so far (`probe_converge.py` / `probe_seeds.py` measured it). More time on one chain buys ~nothing; more independent seeds buys 5–20 %.
- **Per-group tuning matters.** Constraint profiles differ sharply: `case0` is small (20 boxes, no repeat groups); `case9` is largest (81 boxes); `case4/5` are repeat-group-heavy; `case5` is align-heavy. Restart count is scaled by `N` so each restart still reaches convergence.
- **Validity beats optimality.** A layout that violates any constraint or overlaps scores 0. When changing the solver, first confirm overlap clears and constraints hold on every case (run `benchmark.py`, which calls `validate.py` per case) before chasing lower Cost.
- **Benchmarks are timing-sensitive and ~111 s/case.** Output is block-buffered to files — run with `python -u` to see per-case progress. Two single-thread runs can share the 8-core box without skewing timings.
- Stay within the `task.md` library allowlist — code using anything outside it will fail at judging.

## Findings (what's been ruled out / measured)
- **Local compaction does NOT reduce area** (tested: a greedy line-search "slide-to-contact" in the integer basis gives +0.00% on every restart). Reason: `Area` = the global bounding box, the trapped whitespace is *interior*, and sliding any single cluster inward is blocked by a neighbor — only a more compact *global arrangement* shrinks the bbox. Don't re-attempt local/post-hoc compaction.
- **Area is often constraint-limited, not search-limited.** Small/heavily-constrained cases (e.g. case0_5 N=20) hit the *same* area across a 10× range of init spreads → at their structural floor; `cur_171` is already near-optimal there. The recoverable headroom is in **large, search-limited cases (N≈81)** and in **HPWL**.
- **SA converges in ~8 s** then a single chain stagnates; **seed/basin variance is large (14–25%)** → best-of-more-restarts is the reliable lever (this is what `solution_p2` exploits via the eval speedup).

- **`run_integer_tuning` is now incremental too** (done in `solution_p2`). Key lesson: making a stage faster does **nothing on its own** — the restart count is fixed by the schedule, so freed time is just unused. It only pays off paired with a higher restart count (`LARGE_N_MULT`). Integer tuning is bit-exact incremental in the feasibility regime (verified: `test_tuning_equiv.py`).
- **Large N (N=81) is restart-limited**, not compaction-limited: best-of-more restarts found basins ~11% better on the worst-packed case9 variants (area dropped, not just HPWL). Sweet spot ≈ `LARGE_N_MULT` 4 (5+ over-shortens slices → under-convergence on the better-packed variants).
- **The integer stage is already at a local optimum of its basis-move neighborhood** (problem 4, ruled out). A greedy ILS descent (`integer_optimize`, kept but disabled via `USE_GREEDY_POLISH=False`) over the integer basis moves — with overlap-feasible accept on the true objective, net-aware incremental HPWL, and perturbation kicks up to 16× — finds **~0 improvement** on a converged feasible layout, and even strong random perturbation can't escape. The SA/tuning already converge to the neighborhood optimum, so a same-neighborhood descent can't beat it. Headroom is *not* in the integer refinement.
- **Net-directed / large SA moves don't reliably help (problem 2, ruled out for now).** HPWL is 40–62% of cost, but moving boxes toward net-neighbour centroids (least-squares force move) helps only some mid-N cases (case5 area dropped) and **blows up large-N area** (case9 area +10k): HPWL and area are in tension, and large moves scatter the dense large-N packing. Gating to N≤60 and capping the step didn't clear the noise floor.
- **Single-run measurement is dominated by wall-clock variance.** The SA loops are time-bounded (`while time.time()…`), so the #restarts/iterations varies run-to-run even with a fixed `SEED` (different seeds give *identical* results; only timing differs). case9 swings ~2.5% run-to-run, swamping any <2% change. **Consequence:** only trust the *theory* of monotone levers (faster eval → best-of-more on time-bounded paths), not single-run deltas — average ≥3 seeds, or compare distributions, to measure anything smaller.

## Ideas not yet tried (headroom toward ~195)
- **Push mid-N restart multipliers** the same way once their per-restart cost is profiled — currently conservative to avoid the noisy ±2–4% swings.
- **HPWL-directed moves** (move a cluster toward its nets' median) for the 18–46% HPWL portion; **force-directed attraction** to bias toward compact basins on the search-limited large cases.
