# LINEBENCH — Pre-Declared Experimental Protocol (`docs/protocol.md`)

**Protocol Version:** `protocol-v1`  
**Date:** 2026-11-19  
**Status:** Frozen prior to benchmarking and simulation experiments.

---

## 1. Instance Selection Design

### 1.1 Stratification Architecture
LINEBENCH evaluates methods across two primary strata:
1. **Classical Stratum (Scholl 1993):**
   * All instances with $n \le 100$ tasks (covers 21 of the 25 graphs across small and medium sizes).
   * A stratified subsample of instances from the large graphs ($n > 100$: Arcus 2, Bartholdi, Barthol2, Scholl) selected across cycle time tightness ratios.
2. **Generated Stratum (Otto et al. 2013 - SALBPGen):**
   * Covers Small ($n=20$), Medium ($n=50$), and Large ($n=100$). (Very Large $n=1000$ is reserved for scalability stress-testing).
   * Design cells: $3 \times 3 \times 3 = 27$ cells per size tier:
     - Graph structure: Bottleneck (`BN`), Chain (`CH`), Mixed (`MIXED`).
     - Order Strength ($OS$): 0.2, 0.6, 0.9.
     - Task-time distribution: `peak at bottom`, `bimodal`, `peak in middle`.
   * Sample size: A fixed balanced random sample of 3 instances per cell (81 instances per size tier, totaling 243 generated instances), chosen via pre-declared random seed `42`.

### 1.2 Simulation Subset
* Exactly **30 instances** drawn from the benchmark subset spanning diverse sizes, order strengths, and tightness ratios ($c / t_{\max}$).
* Selection rule: 10 small ($n \le 25$), 10 medium ($30 \le n \le 60$), 10 large ($n \ge 70$).
* Random seed: `2026` for subset generation.

---

## 2. Optimization Methods & Parameter Configuration

### 2.1 Heuristics
* **Priority Rules:** 6 rules evaluated in two directions (Forward and Backward):
  1. `RPW`: Ranked Positional Weight
  2. `UBT`: Maximum Task Time
  3. `MTS`: Maximum Total Successors
  4. `MDS`: Maximum Direct Successors
  5. `EST`: Earliest Station Bound
  6. `LST`: Latest Station Bound
* **Deterministic Tie-Breaking:** Ascending task ID $j$.
* **Randomized Multi-Pass Heuristic:**
  - Passes: $K = 500$ passes per instance.
  - Candidate choice mechanism: Softmax / restricted candidate list (RCL) over top eligible tasks.
  - Seeds: Fixed independent seed sequence initialized with master seed `1001`.

### 2.2 Exact CP-SAT Solver Limits
* **Time Limit:** 60.0 seconds per instance for standard benchmark runs; 300.0 seconds for high-effort calibration.
* **Worker Threads:** Fixed at 2 threads (`num_workers = 2`) for reproducible local benchmarking.
* **Proof Status Classification:**
  - `OPTIMAL`: Solver proved minimal stations within time limit.
  - `FEASIBLE`: Valid solution found, but gap remains unproven.
  - `TIMEOUT / UNKNOWN`: No feasible solution found within limit.

---

## 3. Discrete-Event Simulation Design (Phases 5 & 6)

### 3.1 Variability Assumption (Swept Factor)
* Stochastic noise $\xi_{j, i}$ applied to task $j$ on unit $i$:
  $$\xi_{j, i} \sim \text{Lognormal}(\mu, \sigma^2), \quad \text{with } \mathbb{E}[\xi] = 1, \; \text{Var}(\xi) = CV^2$$
  where $\sigma^2 = \ln(1 + CV^2)$ and $\mu = -\sigma^2 / 2$.
* **CV Levels:** $CV \in \{0.10, 0.25, 0.50\}$ (Low, Medium, High).
* Sensitivity Check: Gamma distribution matching mean 1 and identical $CV$.

### 3.2 Tandem Line & Buffer Configurations
* **Line Types:**
  1. Unpaced tandem line with finite intermediate buffers $b \in \{0, 1, 3\}$ and blocking-after-service (BAS).
  2. Paced synchronous line (cycle length governed by the slowest station per unit).
* **Simulation Scale:**
  - Warm-up period ($W$): 500 units.
  - Measurement period ($N - W$): 5,000 units per replication.
  - Replications: $R = 30$ independent replications per balance configuration.

### 3.3 Paired Comparisons & Common Random Numbers (CRN)
* For comparing balances with identical station count $m$ (e.g., minimum load variance vs. heuristic balance):
  - Every candidate balance experiences the **identical pseudo-random stream** of task variation $\xi_{j, i}$ for each replication $r$.
  - Pairwise difference: $\Delta(r) = \text{TH}_A(r) - \text{TH}_B(r)$.
  - Confidence intervals computed via paired Student's $t$ and 95% bootstrap intervals.

---

## 4. Pre-Declared Metrics & Hypothesis Tests

1. **Deterministic Performance:**
   - Station optimality gap: $(m - m^*) / m^*$.
   - Hit rate: $\mathbb{I}(m = m^*)$.
   - Lower bound quality: $(m^* - LB) / m^*$.
   - CPU runtime (seconds).
2. **Simulation Robustness:**
   - Absolute throughput: units completed per unit time.
   - Relative throughput: $\text{TH} / (1/c)$.
   - Station overload probability: $\hat{P}(S_k > c) = \frac{1}{N-W} \sum_{i=W+1}^N \mathbb{I}(S_{k, i} > c)$.
   - Smoothness index: $SX = \sqrt{\sum_{k=1}^m (c - \text{load}_k)^2}$.

---

## 5. Deviations Log

*Any changes or deviations from this committed protocol must be recorded below with date, reason, and impact analysis.*

| Date | Deviation | Reason | Impact |
| --- | --- | --- | --- |
| *None* | Initial protocol frozen | Baseline setup | None |
