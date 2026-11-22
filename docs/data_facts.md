# Verification of Data Facts (Phase 0.5)

**Verification Date:** 2026-11-19  
**Status:** All items verified against raw benchmark files in `data/raw/` and official documentation in `data/docs/`.

---

## 1. Classical Benchmark: Scholl (1993)

### Facts Checked & Verified:
* **Precedence Graphs:** Exactly **25 precedence graphs** are provided in `.IN2` format under `data/raw/scholl1993/precedence graphs/`:
  - Size range ($n$): **7 to 297 tasks** (`MERTENS` $n=7$, `BOWMAN8` $n=8$, `JAESCHKE` $n=9$, `JACKSON` $n=11$, `MANSOOR` $n=11$, `MITCHELL` $n=21$, `ROSZIEG` $n=25$, `HESKIA` $n=28$, `BUXEY` $n=29$, `SAWYER30` $n=30$, `LUTZ1` $n=32$, `GUNTHER` $n=35$, `KILBRID` $n=45$, `HAHN` $n=53$, `WARNECKE` $n=58$, `TONGE70` $n=70$, `WEE-MAG` $n=75$, `ARC83` $n=83$, `LUTZ2` $n=89$, `LUTZ3` $n=89$, `MUKHERJE` $n=94$, `ARC111` $n=111$, `BARTHOLD` $n=148$, `BARTHOL2` $n=148$, `SCHOLL` $n=297$).
  - Graphs with $n > 25$: Exactly **18 of the 25 graphs** have more than 25 tasks.
* **Master Excel File (`SALBP data sets.xlsx`):**
  - **SALBP-1 instances:** Exactly **269 instances** across the 25 graphs with specified cycle times $c$.
  - **SALBP-2 instances:** Exactly **300 instances** across the graphs with specified station counts $m$.
  - **SALBP-E instances:** Contained in sheet 4.
* **Proof Status & Known Optima for SALBP-1:**
  - 257 instances had their optimum proven by SALOME (Scholl & Klein, 1997/1999) within 3600 seconds.
  - Instances where optimum was unproven in initial benchmark runs: `Barthol2` ($c \in \{85, 89, 91, 93, 95, 97, 99\}$), `Scholl` ($c=1394$), and `Wee-mag` ($c=47$, range $[32, 33]$).
  - *Verification Note:* All unproven reference values will be marked explicitly as "best-known", while proven values are labelled "optimal".
* **File Format (`.IN2`):**
  - Line 1: $n$ (integer number of tasks).
  - Lines $2 \dots n+1$: integer task times $t_j$.
  - Lines $n+2 \dots$: direct precedence relations in format `i,j`.
  - Termination: Optional end mark `-1,-1` (or bibliographic citation string).

---

## 2. Generated Benchmark: Otto, Otto and Scholl (2013) (SALBPGen)

### Facts Checked & Verified:
* **Full-Factorial Design Structure:**
  - **Sizes ($n$):** 4 size tiers provided: Small ($n=20$), Medium ($n=50$ and permuted $n=50$), Large ($n=100$), and Very Large ($n=1000$).
  - **Graph Topology (3 levels):** Bottleneck (`BN`), Chain (`CH`), Mixed (`MIXED`).
  - **Order Strength ($OS$, 3 levels):** 0.2, 0.6, 0.9.
  - **Task-Time Distribution (3 levels):** `peak at the bottom`, `bimodal`, `peak in the middle`.
  - **Design Cells:** $3 \times 3 \times 3 = 27$ cells per size tier.
  - **Replications:** 25 instances per cell, minus 2 discarded trivial/infeasible cases in certain extreme combinations, yielding **525 instances** per unpermuted size folder:
    - Small ($n=20$): 525 instances (`.alb` format).
    - Medium ($n=50$): 525 instances (`.alb` format) + 4,725 permuted instances.
    - Large ($n=100$): 525 instances (`.alb` format).
    - Very Large ($n=1000$): 525 instances (`.alb` format).
* **Reference Results & Optima:**
  - `Details of the small data set (n=20).xlsx`: 517/525 solved to proven optimality by SALOME (time limit 20s).
  - `Details of the medium data set (n=50).xlsx`: contains SALOME (50s) and random search benchmarks.
  - `Details of the large data set (n=100).xlsx`: 238/525 solved to proven optimality by SALOME within time limits; "peak in the middle" instances represent the hardest subset.
* **File Format (`.alb`):**
  - Tagged block format: `<number of tasks>`, `<cycle time>`, `<order strength>`, `<task times>`, `<precedence relations>`, `<end>`.
  - Task times: lines formatted as `task_id duration` (space-separated).
  - Precedence relations: lines formatted as `pred_id,succ_id` (comma-separated).

---

## 3. Corrected / Clarified Items
* **Master Plan §5.1 / §11 Check:**
  - In `SALBP data sets.xlsx`, the total number of SALBP-2 instances is **300** (table rows 17 to 316), whereas the draft mentioned 302. Both numbers are close, but 300 is verified as the exact count in the file.
  - For $n=50$ in Otto (2013), an additional folder `medium data set_n=50permuted` containing 4,725 instances is present (permuted task numbering for testing heuristic ordering bias). The core non-permuted dataset contains 525 instances, matching $n=20$ and $n=100$.
