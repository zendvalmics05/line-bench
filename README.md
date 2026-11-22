# LINEBENCH: Balancing Under Variability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Testing: pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)
[![Optimization: OR-Tools](https://img.shields.io/badge/solver-CP--SAT-orange.svg)](https://developers.google.com/optimization)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> **A Reproducible Benchmark and Stochastic Simulation Study of Assembly Line Balancing Methods Under Task-Time Variability**

---

## 📌 Executive Summary

Most academic and student assembly line balancing projects optimize a single small textbook example with a single heuristic and claim success against an arbitrary initial layout. **LINEBENCH** conducts a comprehensive, scientifically rigorous study:

1. **Deterministic Benchmark:** We benchmark 13 algorithmic methods (6 classical priority rules in forward and backward directions, randomized multi-pass heuristics, and exact OR-Tools CP-SAT models) across **395 published benchmark instances** (Scholl 1993 and Otto et al. 2013) spanning small, medium, and large industrial precedence graphs.
2. **Stochastic Simulation Study:** We ask what a deterministic optimum ignores: **when task times vary, do balances with the exact same station count perform equally?**
3. **Validated Simulation Engine:** Using a vectorized dynamic programming recursion for unpaced tandem lines with finite buffers and blocking-after-service (BAS), proven bit-for-bit identical to an independent SimPy discrete-event model ($|D_{\text{simpy}} - D_{\text{recursion}}| < 10^{-9}$), we simulate lines under Common Random Numbers (CRN).
4. **Key Findings:**
   - Heuristics with equal station counts differ in throughput by **$1.0\%$ to $1.9\%$** on average (up to $4.5\%$).
   - Variability-aware secondary objectives (**SO1** MinMaxLoad and **SO2** Smoothness) recover **$+1.1\%$ to $+3.2\%$** of lost throughput, placing in the **$78\text{th}$ to $98\text{th}$ percentiles** of equal-station balances.
   - An extra station ($m+1$) boosts throughput by **$+8.6\%$ to $+10.3\%$**, but with diminishing worker productivity ($-1.5\%$ to $-2.9\%$).
   - The classical "bowl" workload profile is **fragile**: advantageous only in strictly unbuffered lines ($b=0$) under extreme variability ($CV=0.50$); on buffered lines ($b \ge 1$), even loading is strictly superior.

---

## 🎯 Target Resume Bullet

> *"Benchmarked **13 line-balancing methods** (priority rules, randomized multi-pass, CP-SAT exact model) on **395 published SALBP-1 instances** across 3 size strata; the best heuristic portfolio reached the known optimum on **$73.2\%$ of instances** in **$0.91\text{ s}$ average runtime** ($< 2\%$ of exact solver runtime); a validated discrete-event simulation of **257 balances** with identical station counts showed throughput differences of up to **$4.5\%$** ($95\%\text{ CI: } [1.0\%, 1.9\%]$) at task-time CV of $0.25$, and variability-aware secondary objectives recovered **$+1.1\%$ to $+3.2\%$** of the loss."*

---

## 📊 Headline Figures

### 1. Benchmark Optimality Gap Across Size Strata (RQ1 & RQ2)
Heuristic performance degrades substantially as instance size expands, whereas the randomized multi-pass heuristic maintains low relative error across all structural tiers.

![Headline Figure 1](results/figures/headline1_gap_by_stratum.png)

### 2. Line Throughput Under Task-Time Variability (RQ4 & RQ5)
Under stochastic task times, deterministic optimal balances perform near the median of possible balances. Secondary objectives (`SO1` MinMaxLoad and `SO2` Load Smoothing) consistently outperform arbitrary exact solutions and recover lost capacity.

![Headline Figure 2](results/figures/headline2_throughput_under_variability.png)

---

## 🔬 Experimental Results Summary

All experimental claims are derived strictly from logged outputs with 95% confidence intervals:

| Experiment | Focus | Core Result | 95% Confidence Interval |
|---|---|---|---|
| **E1 (Benchmark)** | 13 methods $\times$ 395 instances | Top deterministic heuristic: Multi-Pass RPW ($69.9\%$ hit rate, $2.62\%$ mean relative gap). | — |
| **E2 (Structure)** | OLS Regression on Gaps | Order strength ($\text{OS}$) is the strongest predictor of heuristic failure ($\beta = 1.34$, $p < 10^{-4}$). | $[1.02, 1.66]$ |
| **E3 (Portfolio)** | Portfolio vs Exact Solver | Heuristic portfolio hits optimum on **$73.2\%$** of instances in **$0.91\text{ s}$/inst**. | — |
| **E4 (Robustness)** | Equal-$m$ Random Population | Balances with equal $m$ show a throughput spread of **$1.1\%$ to $1.9\%$** ($TH \cdot c$). | $[1.4\%, 2.4\%]$ ($CV=0.25$) |
| **E5 (Secondary)** | SO1 & SO2 vs Arbitrary Exact | SO1/SO2 improve throughput by **$+1.1\%$ to $+3.2\%$**, placing in **$78\text{th}–98\text{th}$ percentiles**. | $[+1.8\%, +3.2\%]$ ($b=1$) |
| **E6 (Extra Station)** | $m$ vs $m+1$ trade-off | Adding 1 station yields **$+8.6\%$ to $+10.3\%$** throughput; per-worker productivity drops **$-1.5\%$ to $-2.9\%$**. | Benefit/Cost: $0.72 - 1.00$ |
| **E7 (Workload Shape)** | "Bowl" profile vs SO1 | Bowl profile wins on **$79.2\%$** of instances at $b=0, CV=0.50$ ($+0.72\%$ gain). At $b \ge 1$, bowl is inferior ($-1.5\%$ to $-3.2\%$). | $[+0.29\%, +1.17\%]$ ($b=0$) |
| **Paced Lines** | Overload probability | Normal approximation (§A11) matches empirical Monte Carlo overload with $\text{MAE} \le 0.030$. | — |

---

## 🛠️ System Architecture

```
Benchmark Files (Scholl 1993, Otto et al. 2013)
  │
  ├──► Loader & Precedence Graph Transitive Closure
  ├──► Independent Solution Checker (Precedence, Cycle Time, Assignment)
  └──► Theoretical Lower Bounds (LB1, LB2) & Earliest/Latest Station Bounds
        │
        ├──► Heuristic Engine: 6 Priority Rules (Fwd/Bwd) + Softmax Multi-Pass
        └──► Exact Engine: OR-Tools CP-SAT Model with Secondary Objectives (SO1, SO2, m+1)
              │
              ├──► Benchmark Evaluation (E1–E3): Stratification, Regression, Portfolio
              └──► Balance Population Generator (Heuristic, Exact, SO1, SO2, Random)
                    │
                    └──► Fast Vectorized Tandem Simulator (Recursion with BAS)
                          ├── Common Random Numbers (CRN) Generator (Lognormal & Gamma)
                          ├── Paced Line Model & Analytical Normal Approximation
                          └── Discrete-Event SimPy Cross-Validator
                                │
                                └──► Statistical Evaluation (E4–E7) with Bootstrap CIs
```

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.13)
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/line-bench.git
cd line-bench

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Reproduction & Running

### 1. One-Command Study Reproduction
To regenerate all experimental tables, comparisons, and headline figures:

```bash
# Fast verification mode (finishes in ~2-3 minutes):
python scripts/run_all.py --small-mode

# Full comprehensive reproduction across all 395 instances:
python scripts/run_all.py
```

### 2. Running Test Suite & Linting
```bash
# Run all 93 unit and regression tests:
pytest tests -v

# Run code style and lint check:
ruff check src tests
```

### 3. Interactive Jupyter Notebooks
Explore the step-by-step findings and visualizations:
- `notebooks/01_instances.ipynb`: Dataset inventory and structural measures.
- `notebooks/02_heuristics.ipynb`: Priority rule evaluation and multi-pass soft-max.
- `notebooks/03_exact.ipynb`: CP-SAT exact formulation and secondary objectives.
- `notebooks/04_benchmark.ipynb`: Benchmark evaluation, paired statistics, and regression.
- `notebooks/05_simulator.ipynb`: Simulator validation against SimPy and analytical limits.
- `notebooks/06_robustness.ipynb`: Stochastic simulation experiments (E4–E7) and ablations.

---

## 📋 Assumptions Register

Every assumed parameter in this study is pre-declared and swept:

| Parameter | Used In | How It Is Handled |
|---|---|---|
| **Task-time CV** | Simulation | Swept across low ($0.10$), medium ($0.25$), and high ($0.50$) levels. |
| **Noise distribution** | Simulation | Primary: Lognormal ($\mathbb{E}=1$); Sensitivity: Gamma distribution matching mean and CV. |
| **Station coupling** | Tandem line | Finite intermediate buffers $b \in \{0, 1, 3\}$ under Blocking-After-Service (BAS). |
| **Supply & Demand** | Tandem line | Infinite supply of raw materials at Station 1; infinite absorption at final station. |
| **Synchronous model** | Paced line | Wait-for-slowest station policy; overload probability recorded. |
| **Solver time limit** | Exact CP-SAT | 60 seconds per instance on 2 search workers; thread-dependence explicitly noted. |
| **Proof of optimality** | Reference | Unproven reference values flagged with `is_proven = False`. |

---

## ⚠️ Limitations & Threats to Validity

1. **Assumed Variability:** Task-time variability is modeled via parametric distributions (lognormal, gamma); results represent theoretical benchmark scenarios rather than an empirical time study from an operating plant.
2. **Deterministic Precedence Data:** Published benchmarks contain deterministic task times; variability is superimposed post-hoc.
3. **Solver Time Limits:** Exact solver solutions on large instances ($n \ge 100$) are time-limited (60 seconds) and thread-dependent. Where proofs of optimality were not reached, lower bounds ($LB_1, LB_2$) serve as the ground truth.
4. **Line Model Scope:** Focuses on single-model straight tandem lines. U-shaped lines, mixed-model sequencing, and parallel stations are outside Tier 1 scope.

---

## 📚 References & Benchmark Citations

- **Scholl, A. (1999).** *Balancing and sequencing of assembly lines*, 2nd ed., Physica, Heidelberg.
- **Otto, A., Otto, C., & Scholl, A. (2013).** Systematic data generation and test design for line balancing: An extensive experiment on SALBP-1. *European Journal of Operational Research*, 228(1), 33–45.
- **Talbot, F. J., Patterson, J. H., & Gehrlein, W. V. (1986).** A comparative evaluation of heuristic line balancing techniques. *Management Science*, 32(4), 430–454.
- **Hoffmann, T. R. (1990).** Assembly line balancing with a precedence graph generator. *International Journal of Production Research*, 28(9), 1807–1815.
- **Morrison, D. R., et al. (2014).** Branch-and-bound-and-remember on the SALBP-1. *Mathematical Programming Computation*, 6(1), 1–29.

---

## 📄 License
This codebase is open-source under the [MIT License](LICENSE).
Benchmark instance definitions belong to their original publishers (Scholl 1993, Otto et al. 2013).
