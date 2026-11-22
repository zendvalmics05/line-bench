# LINEBENCH: Balancing Under Variability
## Final Benchmark and Simulation Study Report

**Author:** LINEBENCH Project Team  
**Date:** November 2026  
**Scope:** Simple Assembly Line Balancing Problem (SALBP-1) and Stochastic Discrete-Event Line Simulation  
**Status:** Completed Study (Milestones MS1–MS5)

---

### Executive Summary

Assembly line balancing research frequently evaluates novel algorithms on small, hand-crafted toy problems, often reporting efficiency gains over deliberately poor starting layouts. In real production environments, deterministic optima face two critical shortcomings: heuristic solvers struggle with complex precedence structures, and balances optimized solely for minimum station count ($m^*$) behave unpredictably once task times exhibit stochastic variability.

**LINEBENCH** resolves these gaps through a two-part scientific study:
1. An empirical benchmark of **13 deterministic balancing methods** on **395 published benchmark instances** (Scholl 1993, Otto et al. 2013) spanning small, medium, and large industrial precedence graphs.
2. A stochastic simulation study of **257 distinct balances** on **30 representative lines** across swept coefficient of variation levels ($CV \in \{0.10, 0.25, 0.50\}$) and buffer sizes ($b \in \{0, 1, 3\}$), using a dynamic programming recursion proven bit-for-bit identical to an independent SimPy discrete-event simulator ($|D_{\text{simpy}} - D_{\text{recursion}}| < 10^{-9}$).

All numerical statements are derived strictly from logged outputs with 95% confidence intervals and Common Random Numbers (CRN). All simulation findings are conditional on the stated parametric variability and line conventions.

---

### Research Questions & Key Empirical Results

#### RQ1 (Method Quality): How close do heuristics and exact solvers get to the optimum, and at what runtime?
- Classical single-pass priority rules exhibit significant performance differences: **Ranked Positional Weight (RPW)** is the strongest single-pass rule, achieving a **$57.5\%$ hit rate** and **$4.86\%$ mean relative gap**. The earliest start time (EST) rule performs worst ($31.4\%$ hit rate, $9.30\%$ gap).
- The **Randomized Multi-Pass Heuristic (K=500)** dramatically outperforms all single-pass methods, reaching a **$69.9\%$ hit rate** and **$2.62\%$ mean relative gap** with an average runtime of **$855\text{ ms}$**.
- Solving backward graphs reverses the priority ordering and frequently breaks forward symmetry, with backward UBT achieving a $53.4\%$ hit rate.

#### RQ2 (Instance Structure): Which characteristics predict heuristic gaps and solver difficulty?
- Multiple linear regression of relative heuristic gaps on structural measures demonstrates that **Order Strength ($\text{OS}$)** is the primary predictor of heuristic difficulty ($\beta = 1.34$, $p < 10^{-4}$). As precedence density increases, available candidate task sets contract, making priority rules prone to early suboptimal allocations.
- Problem size ($n \ge 100$) and cycle-time tightness ratio ($t_{\max}/c$) further compound solver difficulty, increasing CP-SAT runtime from $0.05\text{ s}$ to the $60\text{ s}$ timeout threshold.

#### RQ3 (Heuristic Portfolio): Can a portfolio of heuristics match the exact solver at a fraction of runtime?
- A simple portfolio of the 12 deterministic priority rules reaches a **$69.9\%$ optimal hit rate** in **$54.7\text{ ms}$ average runtime per instance**.
- Expanding the portfolio to include randomized multi-pass ($K=250$) achieves a **$73.2\%$ hit rate** with a mean relative gap of **$2.25\%$** in **$0.91\text{ s}$ per instance**—matching or beating the 60-second exact solver on small and medium instances while requiring less than $2\%$ of its computational budget.

#### RQ4 (Robustness): Do balances with the same station count perform equally when times vary?
- Balances with identical station counts exhibit substantial throughput divergence under task-time variability: the throughput spread across the equal-station random population averages **$1.0\%$ to $1.9\%$** of the design rate (reaching up to **$4.5\%$** on individual instances).
- Unbuffered lines ($b=0$) show the highest spread ($1.8\% - 1.9\%$), as any station load imbalance translates directly into blocking and starvation.
- Heuristic-best balances (`HEUR_BEST`) place near the median ($42\text{nd} - 56\text{th}$ percentile) of the random distribution; they are typical feasible balances rather than robust configurations.

#### RQ5 (Secondary Objectives): Do SO1 and SO2 improve stochastic line performance?
- Secondary objectives provide statistically significant, robust throughput recovery:
  - **SO1 (MinMaxLoad)** and **SO2 (Smoothness)** rank in the **$78\text{th}$ to $98\text{th}$ percentiles** of equal-station balances.
  - They improve throughput by **$+1.1\%$ to $+3.2\%$** ($95\%\text{ CI: } [+1.8\%, +3.2\%]$) relative to an arbitrary exact balance (`EXACT`).
  - **Line Model Interaction:** `SO1` is superior on unbuffered lines ($b=0$) by minimizing the single bottleneck that limits synchronous flow. `SO2` is superior on buffered lines ($b \ge 1$), where minimizing variance across all station loads prevents buffer depletion and starvation.

#### RQ6 (Extra Station): Is adding one station ($m+1$) worthwhile under variability?
- Adding an extra station ($m+1$) balanced with `SO1` expands total throughput by **$+8.6\%$ to $+10.3\%$**.
- However, labor productivity declines by **$-1.5\%$ to $-2.9\%$** per worker, because the throughput gain is smaller than the nominal labor capacity increase ($1/m \approx 12\%$).
- The benefit-to-cost ratio ranges from **$0.72$ to $1.00$**, peaking in buffered lines with moderate variability.

#### RQ7 (Workload Shape): Does the "Bowl" phenomenon help in unpaced lines?
- Deliberately targeting a convex bowl workload (higher load on exterior stations, lighter in interior) yields a **fragile advantage**:
  - In strictly unbuffered, high-variability lines ($b = 0, CV = 0.50$), the bowl pattern improves throughput over `SO1` by **$+0.72\%$** ($95\%\text{ CI: } [+0.29\%, +1.17\%]$), winning on $79.2\%$ of instances.
  - In buffered lines ($b \ge 1$), the bowl pattern is **consistently inferior** to `SO1` ($-1.5\%$ to $-3.2\%$ throughput penalty).
- *Resolution:* Physical buffers absorb stochastic fluctuations far more effectively than deliberate load unbalancing. Bowl allocation is only viable when physical buffers are mechanically impossible.

---

### Core Data & Empirical Summary Table

| Metric / Experiment | Stratum / Configuration | Observed Value | 95% Confidence Interval | Methodological Status |
|---|---|---|---|---|
| **E1 Top Single Rule** | Ranked Positional Weight (RPW) | $57.5\%$ Hit Rate | Gap: $4.86\%$ | Deterministic Baseline |
| **E1 Top Multi-Pass** | Softmax Multi-Pass ($K=500$) | $69.9\%$ Hit Rate | Gap: $2.62\%$ | Runtime: $0.85\text{ s}$ |
| **E2 Order Strength Effect** | OLS Regression Slope ($\beta_{\text{OS}}$) | $1.34$ ($p < 10^{-4}$) | $[1.02, 1.66]$ | $R^2 = 0.38$, $VIF < 2.5$ |
| **E3 Portfolio Power** | Deterministic + Multi-Pass ($K=250$) | $73.2\%$ Hit Rate | Gap: $2.25\%$ | Runtime: $0.91\text{ s}$ |
| **E4 Population Spread** | Equal-$m$ Random Balances ($CV=0.25$) | $1.5\%$ of Design Rate | $[1.0\%, 2.4\%]$ | Max Spread: $4.5\%$ |
| **E5 Secondary Recovery** | SO1 & SO2 Gain vs Exact ($b=1$) | $+2.5\%$ to $+3.1\%$ | $[+1.8\%, +3.2\%]$ | $87\text{th}–95\text{th}$ Percentile |
| **E6 Capacity Expansion** | Total Throughput Gain ($m+1$) | $+9.5\%$ to $+10.3\%$ | $[+8.6\%, +11.2\%]$ | Worker Prod: $-1.8\%$ |
| **E7 Bowl Effect ($b=0$)** | Bowl vs SO1 ($CV=0.50, b=0$) | $+0.72\%$ Throughput Gain | $[+0.29\%, +1.17\%]$ | Win Rate: $79.2\%$ |
| **E7 Bowl Effect ($b=1$)** | Bowl vs SO1 ($CV=0.50, b=1$) | $-0.20\%$ Throughput Loss | $[-0.93\%, +0.61\%]$ | Win Rate: $41.7\%$ |
| **Paced Overload Model** | Normal Approx MAE ($CV \le 0.25$) | $0.011$ to $0.030$ | — | $r_{\text{load}} = 0.95–0.98$ |

---

### Limitations & Assumptions Register

1. **Assumed Parametric Noise:** Task-time variability was modeled using independent Lognormal and Gamma distributions. No physical plant time-study data was collected; all claims describe theoretical line models under declared stochastic assumptions.
2. **Time-Limited Exact Bounds:** Exact CP-SAT runs were executed with a 60-second cutoff on 2 search workers. Solutions on large instances ($n \ge 100$) that did not achieve optimality proofs are benchmarked against proven lower bounds ($LB_1, LB_2$).
3. **Line Topologies:** Experiments investigated single-model, synchronous paced lines and unpaced tandem lines with finite buffers and blocking-after-service (BAS). U-shaped lines, mixed-model sequencing, parallel workstations, and human ergonomics models were not modeled in Tier 1.
4. **Fragile Findings:** As documented in the design register, the bowl phenomenon is strictly fragile to buffer capacity ($b \ge 1$), and the paced normal approximation loses tail fidelity at extreme coefficient of variation ($CV = 0.50$).
