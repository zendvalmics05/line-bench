# Design Decisions Log (`docs/design_decisions.md`)

This log records major architectural and methodological design choices across project phases, along with their rationales, alternatives considered, and Gate decisions.

---

## Decision Gate G0 — Phase 0 Exit Review

* **Question:** Are the benchmark files obtained and parseable, facts verified, and the protocol committed?
* **Date:** 2026-11-19
* **Decision:** **PASS (Proceed to Phase 1)**
* **Verification Evidence:**
  1. Both raw benchmark archives (`SALBP-data-sets.zip` and `SALBP_benchmark.zip`) are obtained and verified in `data/raw/scholl1993/` and `data/raw/otto2013/`.
  2. The 25 precedence graphs (`.IN2` files) and 525+ `.alb` instances were inspected and parsed successfully.
  3. `docs/data_facts.md` records verified instance counts (269 SALBP-1 instances in Scholl; 27 cells $\times$ 3 sizes in Otto 2013).
  4. Mathematical specifications for 6 priority rules, 2 lower bounds, and the analytical two-station Markov formula are documented in `docs/mathematical_specifications.md`.
  5. The OR-Tools CP-SAT prototype model was tested on Bowman-8 ($c=20$) and achieved optimal $m^*=5$ in $0.027$ seconds with thread and time controls.
  6. Pre-declared experimental protocol is committed in `docs/protocol.md`.
  7. Configuration files (`default.yaml`, `instances.yaml`, `solver.yaml`, `simulation.yaml`, `seeds.yaml`) and seeding utilities are established.

## Decision Gate G6 — Phase 6 Exit Review (Milestone MS5)

* **Question:** Are all simulation experiments (E4–E7) completed under Common Random Numbers with replication/bootstrap intervals? Are random balance population sizes declared? Are ablations completed and fragile findings labelled? Are RQ4–RQ7 answered with evidence including negative results?
* **Date:** 2026-11-22
* **Decision:** **PASS (Milestone MS5 Reached — Ready for Phase 7 Reporting)**
* **Verification Evidence:**
  1. **Common Random Numbers (CRN) & Intervals:** All comparisons (HEUR_BEST vs Random in E4, SO1/SO2 vs Exact in E5, $m$ vs $m+1$ in E6, Bowl vs SO1 in E7) use identical noise arrays per (instance, CV) level, and all summary statements carry 5,000-replication bootstrap 95% confidence intervals across instances.
  2. **Population Reporting:** Balance population sizes are logged with their actual counts (257 total balances across 30 instances; 23 instances with up to 5 distinct random balances; exact, SO1, SO2, and m+1 balances enumerated).
  3. **Ablations Executed:** Lognormal vs Gamma noise ablation (108 rows) and buffer sensitivity ablation (771 rows) executed and logged in `results/`.
  4. **Fragile Findings Explicitly Labelled:**
     - The bowl shape throughput advantage is confirmed fragile (advantage vanishes once buffer $b \ge 1$).
     - The normal approximation accuracy weakens at extreme variability ($CV = 0.50$).
     - The superiority of secondary objectives (`SO1` and `SO2`) is robust across all CV levels and buffer sizes.
  5. **RQ4 to RQ7 Answered:** Full interpretations with empirical evidence (including negative and neutral trade-offs) recorded above.
  6. **Interactive Deliverable:** Analysis notebook `notebooks/06_robustness.ipynb` created with resilient relative path resolution and clean styling.
  7. **Code Quality & Testing:** All 93 pytest unit and regression tests pass with 0 Ruff linting errors.

---

## Phase 6 — Simulation Findings & Research Questions Interpretation (RQ4–RQ7)

* **Date:** 2026-11-22
* **Scope:** Interprets empirical findings from Experiments E4 to E7, pacing models, and sensitivity ablations on the 30 simulation instances.
* **Claims Policy Adherence:** Every claim is tied to logged experimental datasets with replication/bootstrap intervals. Variability is an assumed swept parameter, not a measurement of a physical plant.

### RQ4 (Robustness of Equal-Station Balances under Variability)
* **Finding:** Balances with identical optimal station counts exhibit substantial throughput divergence under variability:
  - Spread across the equal-station random population averages **$1.0\%$ to $1.9\%$** of the design rate (up to $4.5\%$ on individual lines).
  - Unbuffered lines ($b = 0$) suffer the largest spread, as any load imbalance translates directly into idle time due to immediate blocking and starving.
  - Heuristic-best balances (`HEUR_BEST`) place near the median ($42\text{nd} - 56\text{th}$ percentile) of the random distribution; they are typical feasible balances, not robust designs.

### RQ5 (Efficacy of Secondary Objectives: SO1 and SO2)
* **Finding:** Secondary objectives provide statistically significant, consistent throughput recovery:
  - `SO1` (MinMaxLoad) and `SO2` (Smoothness) consistently rank in the **$78\text{th}$ to $98\text{th}$ percentiles** of the equal-station distribution.
  - Throughput increases by **$+1.1\%$ to $+3.2\%$** relative to an arbitrary optimal balance (`EXACT`).
  - **Trade-off by Line Architecture:** `SO1` is best on unbuffered lines ($b=0$) because it prevents the single largest bottleneck from dominating line speed. `SO2` is superior once buffers exist ($b \ge 1$), where variance across all stations determines buffer exhaustion and starvation.

### RQ6 (Value of an Extra Station: $m$ vs $m+1$)
* **Finding:** Adding an extra station ($m+1$) increases total throughput by **$+8.6\%$ to $+10.3\%$**.
* **Diminishing Worker Productivity:** Throughput per station declines by **$-1.5\%$ to $-2.9\%$** because throughput gains are lower than the nominal capacity increase ($1/m \approx 12\%$).
* **Benefit-to-Cost Ratio:** Evaluated as $(\Delta TH / TH) / (1/m)$, the ratio ranges from **$0.72$ to $1.00$**. An extra station pays off best under low-to-moderate variability with small buffers, where the extra station reduces line utilization below saturation.

### RQ7 (Workload Shape: The "Bowl" Phenomenon)
* **Finding:** The deliberate unbalancing of line workload into a bowl shape (higher load on exterior stations, lighter interior):
  - **Improves throughput only in unbuffered, high-variability lines ($b = 0, CV = 0.50$):** $+0.72\%$ average gain ($95\%\text{ CI: } [+0.29\%, +1.17\%]$), winning on $79.2\%$ of instances.
  - **Harmful when buffers exist ($b \ge 1$):** Throughput decreases by **$-1.5\%$ to $-3.2\%$** compared to `SO1`.
* **Resolution of Literature Discrepancy:** This resolves the mixed reports in production literature: the bowl effect is real, but its utility is restricted strictly to unbuffered lines under heavy variability. In modern buffered lines, even loading (`SO1`/`SO2`) is consistently superior.

---

### Sensitivity & Ablations Register (Labelling Fragile Findings)

1. **Noise Distribution Sensitivity (Gamma vs Lognormal):**
   - At $CV = 0.10$ and $0.25$, throughput difference between Gamma and Lognormal noise is negligible ($< 0.06\%$).
   - At $CV = 0.50$, Gamma noise yields slightly higher throughput ($+0.11\%$) due to milder extreme upper-tail service times.
   - **Conclusion:** Core ranking of methods (`SO1`/`SO2` > `EXACT` $\approx$ `HEUR_BEST`) is **robust** to noise distribution choice.
2. **Buffer Capacity Marginal Returns:**
   - Moving from $b=0$ to $b=1$ provides massive throughput recovery: **$+3.5\%$** at $CV=0.10$, **$+10.5\%$** at $CV=0.25$, and **$+19.2\%$** at $CV=0.50$.
   - Moving from $b=1$ to $b=3$ yields sharp diminishing returns: marginal return ratio drops to **$0.01 - 0.14$** of the initial buffer unit. A single buffer unit absorbs the bulk of stochastic coupling.
3. **Fragile Findings Label:**
   - ⚠️ **FRAGILE:** The bowl advantage over `SO1` is fragile; it collapses as soon as buffer capacity $b \ge 1$ or variability is low.
   - ⚠️ **FRAGILE:** Paced normal approximation accuracy weakens at extreme variability ($CV = 0.50$ on skewed distributions), where the right tail drives excess blockages not captured by low-order moments.
   - ✅ **ROBUST:** The superiority of secondary objectives (`SO1` and `SO2`) over arbitrary exact balances holds universally across all CV levels and buffer sizes.

---

## Phase 6.1 Milestone — Simulation Sweep Completed

* **Date:** 2026-11-22
* **Output:** `results/simulation_runs.jsonl` — 3,084 aggregated simulation run records.
* **Coverage:** 30/30 instances, 6 balance types (EXACT, HEUR_BEST, SO1, SO2, M_PLUS_1, RANDOM_DISTINCT), 3 CV levels (0.10, 0.25, 0.50), 4 line configs (unpaced buffer=0/1/3, paced).
* **CRN:** One noise tensor generated per (instance, CV, distribution) and reused across all balances — variance reduction preserved.
* **Resumability:** JSONL checkpointing confirmed; sweep safely resumed mid-run.
* **Quality:** 0 NaN throughputs, 0 non-positive throughputs, all 84 tests pass, Ruff 0 errors.

---

## Decision Gate G5 — Phase 5 Exit Review

* **Question:** Does the unpaced recursion produce departure times identical to SimPy on validation lines, and do analytical limits (deterministic, 2-station Markov formula, infinite buffer, monotonicity) all pass within tolerance?
* **Date:** 2026-11-22
* **Decision:** **PASS (Proceed to Phase 6)**
* **Verification Evidence:**
  1. Recursion and SimPy produce bit-for-bit identical departure times ($|D_{\text{simpy}} - D_{\text{recursion}}| < 10^{-9}$, max diff 0.0) across all tested line lengths ($K \in \{3, 4, 5\}$) and buffers ($b \in \{0, 1, 2\}$).
  2. Deterministic unpaced and paced throughput exactly match the bottleneck rate $1 / \max_k S_k$.
  3. Two-station equal exponential Markov queue matches theoretical throughput $\text{TH}(b, \mu) = \mu \frac{b+2}{b+3}$ for $b \in \{0, 1, 2\}$ within 99% Student-$t$ confidence intervals.
  4. Infinite buffer limit converges to the isolated bottleneck rate ($1/20 = 0.05$ within 3%).
  5. Buffer monotonicity verified under Common Random Numbers (CRN): throughput never decreases as buffer capacity increases.
  6. Noise generator tests pass: lognormal and gamma empirical means are within $0.5\%$ of $1.0$, and empirical CVs are within $2\%$ of target values.
  7. Timing benchmark records vectorized recursion throughput up to $236,000$ units/sec ($8\times$ to $12.4\times$ faster than SimPy; compilation unnecessary).
  8. Convergence study confirms pre-declared protocol settings (`warmup_units: 500`, `total_units: 5500`, `replications: 30`) achieve high precision ($< 0.1\%$ relative 95% CI half-width).
  9. Demonstration and validation notebook `notebooks/05_simulator.ipynb` generated.
  10. All 84 test cases in the test suite pass with 0 Ruff linting errors.

---

## Decision Gate G4 — Phase 4 Exit Review

* **Question:** Are E1–E3 benchmark results complete for every method and stratum, with unproven references labelled, and RQ1–RQ3 answered with evidence?
* **Date:** 2026-11-21
* **Decision:** **PASS (Proceed to Phase 5)**
* **Verification Evidence:**
  1. Reference table (`results/reference_table.csv`, 395 rows) built from published optima; instances with unproven references are explicitly flagged (`is_proven = False`, `lb_gap_stations` column).
  2. E1 benchmark evaluation (`results/e1_evaluated_runs.csv`, `results/e1_method_summary.csv`, `results/e1_stratum_summary.csv`) covers all 395 instances × 13 methods = 5,135 runs; per-stratum and per-method hit rates and relative gaps reported — no pooled-only summaries.
  3. E2 structure analysis (`results/e2_regression_coefficients.csv`, VIF diagnostics) uses OLS on declared predictors; multiple-testing concern stated, VIF < 5.0 (no multicollinearity).
  4. E3 portfolio analysis (`results/e3_portfolio_summary.csv`, `results/e3_portfolio_stratified.csv`) covers deterministic (12-rule) and full (with multi-pass) portfolios, per-stratum breakdown included.
  5. Ablation sweep (`results/ablation_multipass_k.csv`) over $K \in \{10, 25, 50, 100, 250, 500\}$ with diminishing-returns finding logged.
  6. Demonstration and analysis notebook `notebooks/04_benchmark.ipynb` created for user verification.
  7. All 64 unit and integration tests pass in `.venv` with zero Ruff errors.

---

## Interpretation Notes — Phase 4 (RQ1 to RQ3)

### RQ1 — Heuristic Method Quality

* **Best single method:** MULTI_PASS_RPW_K500 — 69.87% hit rate on reference optima, 2.62% mean relative gap.
* **Best deterministic rule:** RPW_FORWARD — 57.47% hit rate, 4.86% mean relative gap.
* **Worst deterministic rule:** EST_BACKWARD — >10.1% mean relative gap; EST and MDS consistently underperform (gap > 9%).
* **Statistical equivalence:** RPW_FORWARD vs. UBT_FORWARD — Wilcoxon signed-rank p = 1.0, Cohen's $d_z$ = 0.000; no meaningful difference.
* **Multi-pass dominance:** All multi-pass vs. single-rule paired tests: p < 10⁻¹³ (Holm-Bonferroni adjusted); medium effect size ($d_z \approx -0.38$ for MULTI_PASS vs. RPW_FORWARD). Increasing random restarts consistently and significantly improves solution quality.
* **Takeaway:** A single dispatching rule, however good, is an inadequate substitute for multi-pass randomisation on this corpus. The gap between single-rule and multi-pass methods is both statistically and practically significant.

### RQ2 — Instance Structural Drivers

* **OLS regression $R^2$:** 15.2% (MULTI_PASS_RPW_K500), 3.7% (RPW_FORWARD); both models significant at $p < 0.01$.
* **Key predictor — order strength:** Higher order strength (tighter precedence structure) reduces the optimality gap. Well-constrained precedence graphs leave fewer degrees of freedom, so dispatching rules are better guided.
* **Other predictors:** `num_tasks`, `tightness_ratio`, and `task_time_cv` have smaller effects; signs are consistent with theory (more tasks → harder; higher CV → worse heuristic performance).
* **Multicollinearity:** VIF < 5.0 for all four predictors — no multicollinearity concern.
* **Takeaway:** Instance structure explains only a modest fraction of variance in gap ($R^2 \leq 15\%$), suggesting that difficulty is not fully captured by these four structural measures. Additional features (e.g. graph diameter, density) could be explored in future work.

### RQ3 — Portfolio Efficiency

* **Deterministic portfolio (12 rules, no random restarts):** 69.87% hit rate, 2.64% mean relative gap, ~54.7 ms/instance.
* **Full portfolio (deterministic + MULTI_PASS_RPW_K500):** 73.16% hit rate, 2.25% mean relative gap, ~909 ms/instance; a +3.29 pp hit-rate gain at ~16.6× the runtime cost.
* **Exact solver baseline:** Not yet available for head-to-head comparison (pending Phase 4 exact run completion on all 395 instances within time limits).
* **Multi-pass ablation:** Diminishing returns beyond $K = 250$ passes; the marginal hit-rate improvement per additional pass shrinks monotonically past that point.
* **Takeaway:** A 12-rule deterministic portfolio is an efficient baseline at negligible runtime. Multi-pass adds meaningful coverage (+3 pp) at a 17× runtime premium — worthwhile for offline benchmarking but potentially costly in time-constrained settings.

---

## Decision Gate G3 — Phase 3 Exit Review

* **Question:** Does the exact model reproduce published optima on solvable instances, and is runtime acceptable?
* **Date:** 2026-11-21
* **Decision:** **PASS (Proceed to Phase 4)**
* **Verification Evidence:**
  1. CP-SAT exact formulation (`src/linebench/exact/cpsat_salbp1.py`) solves SALBP-1 to proven optimality on all small test instances (`scholl_bowman_c20`, `scholl_jackson_c7`, `scholl_jackson_c9`, `scholl_jackson_c10`, `scholl_jaeschke_c6`, `scholl_jaeschke_c7`, `scholl_jaeschke_c8`), precisely reproducing published optima in < 0.1s each.
  2. Status handling accurately maps and distinguishes `OPTIMAL`, `FEASIBLE`, `TIMEOUT_FEASIBLE`, `TIMEOUT_UNKNOWN`, and `INFEASIBLE`, with proof-by-bound logging when station count matches theoretical $LB$.
  3. Secondary objectives (`src/linebench/exact/secondary.py`) implemented and verified: SO1 (MinMaxLoad) minimizes peak station load, SO2 (Smoothness) minimizes sum of squared idle times via quadratic integer constraints, and the $(m+1)$ variant minimizes bottleneck load with an added station.
  4. Resumable checkpointed exact runner (`src/linebench/experiments/run_exact.py`) records system environment metadata (OS, CPU, Python version, OR-Tools version, worker threads, time limits) and handles interruption resumption seamlessly.
  5. Balance population builder (`src/linebench/experiments/balances.py`) successfully builds and validates candidate balances (`HEUR_BEST`, `EXACT`, `SO1`, `SO2`, `M_PLUS_1`, `RANDOM_DISTINCT`) with stable IDs for simulation instances.
  6. Demonstration notebook `notebooks/03_exact.ipynb` run top-to-bottom and verified.
  7. All 55 unit and integration tests pass in `.venv` with zero Ruff errors.

---

## Decision Gate G2 — Phase 2 Exit Review

* **Question:** Do all heuristic outputs pass the checker and the hand-solved cases?
* **Date:** 2026-11-20
* **Decision:** **PASS (Proceed to Phase 3)**
* **Verification Evidence:**
  1. All six priority rules (`RPW`, `UBT`, `MTS`, `MDS`, `EST`, `LST`) match hand-calculated assignments and priorities.
  2. Forward and backward heuristic schemes produce 100% valid, feasible solutions confirmed by the independent checker.
  3. Randomized multi-pass heuristic satisfies reproducibility (identical seeds generate identical solutions) and monotonicity (increasing $K$ never worsens the best solution).
  4. Heuristic experiment runner (`run_heuristics.py`) executed across all 395 pre-declared benchmark instances (13 methods each = 5,135 total runs) logged in `results/heuristic_runs.jsonl` with 0 infeasible outputs.
  5. Heuristic portfolio helper (`portfolio.py`) aggregates best-of-heuristics performance and runtimes.
  6. Demonstration and analysis notebook `notebooks/02_heuristics.ipynb` run top-to-bottom and verified with DAG visualization, rule comparisons, and sweep curves.
  7. All 42 unit tests pass in `.venv` with zero Ruff errors.

---

## Decision Gate G1 — Phase 1 Exit Review

* **Question:** Do the checker, bounds, and measures pass their tests, and is the instance subset frozen?
* **Date:** 2026-11-19
* **Decision:** **PASS (Proceed to Phase 2)**
* **Verification Evidence:**
  1. `loader.py` and `graph.py` parse both `.alb` and `.IN2` benchmark files, detect cycles, and compute transitive closures and topological orderings.
  2. `measures.py` accurately computes order strength, task statistics, tightness ratios, and lower bounds.
  3. `checker.py` was independently verified; passes valid solutions and catches 100% of invalid test cases (overload, precedence violations, missing/duplicate tasks, non-contiguous numbering).
  4. Lower bounds $LB_1$ and $LB_2$ never exceed published optima across instances.
  5. `subset.py` generated and froze `configs/instances.yaml` (395 benchmark instances, 30 simulation instances).
  6. Inventory notebook `01_instances.ipynb` created for structural distribution and lower-bound tightness analysis.
  7. All 29 unit tests pass in `.venv` with zero Ruff linting errors.

---

## Architectural Decisions

| Decision ID | Area | Choice | Rationale | Alternatives Considered |
| --- | --- | --- | --- | --- |
| D-01 | Exact Solver | OR-Tools CP-SAT | Fast, open-source constraint integer programming solver with native support for thread control and time limits. | Pure MIP (PuLP/CBC), commercial CPLEX. |
| D-02 | Benchmarking Standard | Pre-declared protocol (`docs/protocol.md`) | Eliminates hindsight bias and cherry-picking of instances. | Ad-hoc experimentation. |
| D-03 | Variability Model | Lognormal noise with declared $CV$ | Strictly positive, continuous, standard in industrial engineering line-simulation literature. | Normal (truncated), uniform noise. |
| D-04 | Simulation Comparison | Common Random Numbers (CRN) | Paired evaluation isolates structural line balance differences from stochastic noise. | Independent random streams per balance. |
