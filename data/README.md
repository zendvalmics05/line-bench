# LINEBENCH — Benchmark Data Documentation (`data/README.md`)

This document records the provenance, download sources, licensing, academic citations, and manual setup instructions for the benchmark datasets used in the LINEBENCH project, as required by Phase 0.4g of the implementation plan.

---

## 1. Overview of Data Sources

LINEBENCH relies on **two public benchmark datasets** for the Simple Assembly Line Balancing Problem (SALBP):

1. **Scholl (1993) Benchmark Dataset (Classical Set)**
   * **Source Page:** [https://assembly-line-balancing.de/salbp/benchmark-data-sets-1993/](https://assembly-line-balancing.de/salbp/benchmark-data-sets-1993/) (also referenced as `benchmark-data-sets-by-scholl-1993/`)
   * **Download Archive:** `SALBP-data-sets.zip` (Direct link: `https://assembly-line-balancing.de/wp-content/uploads/2017/01/SALBP-data-sets.zip`)
   * **Contents:**
     - 25 classical precedence graphs stored in `.IN2` format under `precedence graphs/` (including historical graphs from Bowman 1960, Jaeschke 1964, Jackson 1956, Mansoor 1964, Mitchell 1957, Heskiaoff 1972, Buxey 1974, Sawyer 1970, Lutz 1974, Gunther 1974, Kilbridge & Wester 1961, Hahn 1972, Warnecke 1971, Tonge 1961, Wee & Magazine 1981, Arcus 1966, Mukherje 1946, Bartholdi 1993, and Scholl 1993).
     - Master spreadsheet `SALBP data sets.xlsx` defining:
       * 269 SALBP-1 instances (cycle times $c$, known minimal station count $m^*$, lower/upper bounds, and SALOME solver performance).
       * 300 SALBP-2 instances (given station counts $m$ and optimal/best-known cycle times $c^*$).
       * SALBP-E instances.
     - `README.DOC` explaining the `.IN2` file format.

2. **Otto, Otto and Scholl (2013) Benchmark Dataset (SALBPGen / Generated Set)**
   * **Source Page:** [https://assembly-line-balancing.de/salbp/benchmark-data-sets-2013/](https://assembly-line-balancing.de/salbp/benchmark-data-sets-2013/) (also referenced as `benchmark-data-sets-by-otto-et-al-2013/`)
   * **Download Archive:** `SALBP_benchmark.zip` (Direct link: `https://assembly-line-balancing.de/wp-content/uploads/2017/01/SALBP_benchmark.zip`)
   * **Contents:**
     - A full-factorial design generating challenging SALBP-1 problem instances varying by:
       * Problem size ($n \in \{20, 50, 100, 1000\}$ tasks)
       * Order Strength ($OS \in \{0.2, 0.6, 0.9\}$)
       * Graph topology (bottleneck, chain, mixed)
       * Task-time distribution (bimodal, peak at bottom, peak in middle)
     - Standard `.ALB` format instance files.
     - Reference results and optimal / best-known bounds from the SALOME-1 solver and BBR procedure (Morrison et al., 2014).

3. **Format & Documentation References**
   * **Format Specification:** `format-ALB.pdf` ([https://assembly-line-balancing.de/wp-content/uploads/2017/01/format-ALB.pdf](https://assembly-line-balancing.de/wp-content/uploads/2017/01/format-ALB.pdf))
   * **Scholl Precedence Graph Documentation:** `Scholl-1993-ALBData.pdf` ([https://assembly-line-balancing.de/wp-content/uploads/2017/01/Scholl-1993-ALBData.pdf](https://assembly-line-balancing.de/wp-content/uploads/2017/01/Scholl-1993-ALBData.pdf))

---

## 2. Download and Verification Date

* **Verification Date:** 2026-11-19
* **Current Workspace Status:**
  - `data/docs/format-ALB.pdf` and `data/docs/Scholl-1993-ALBData.pdf` are present.
  - `data/raw/scholl1993/` contains `SALBP data sets.xlsx` and `precedence graphs/` with 25 `.IN2` files and `README.DOC`.
  - `data/raw/otto2013/` is prepared for extraction of `SALBP_benchmark.zip`.

---

## 3. Licence and Distribution Policy

* **Licence Status:** No explicit open-source licence is stated on `assembly-line-balancing.de`. The data is published freely for academic and benchmarking research.
* **Distribution Rule:** In accordance with project policy and copyright standards:
  - **Do NOT commit raw benchmark archives or extracted data files to public git repositories.**
  - `data/raw/` must remain in `.gitignore`.
  - Anyone replicating the project should download the benchmark archives directly from the official source links above using the reproduction instructions below.

---

## 4. Academic Citations

When using or referencing these benchmark datasets, cite the following foundational works:

1. **Scholl (1993 / 1999):**
   > Scholl, A. (1993): *Data of Assembly Line Balancing Problems*. Schriften zur Quantitativen Betriebswirtschaftslehre 16/93, TH Darmstadt.
   > Scholl, A. (1999): *Balancing and Sequencing of Assembly Lines*, 2nd ed., Physica-Verlag, Heidelberg.

2. **Otto, Otto and Scholl (2013):**
   > Otto, A., Otto, C. and Scholl, A. (2013): *SALBPGen: A systematically generated dataset for the simple assembly line balancing problem*, European Journal of Operational Research, 228(1), pp. 33–45.

3. **Classical Benchmark Problem Contributors:**
   > Talbot, F. B., Patterson, J. H. and Gehrlein, W. V. (1986): *A comparative evaluation of heuristic line balancing techniques*, Management Science, 32(4), pp. 430–454.
   > Hoffmann, T. R. (1990): *Assembly line balancing — a set of challenging problems*, International Journal of Production Research, 28(10), pp. 1807–1815.

4. **Reference Algorithms & Solvers:**
   > Scholl, A. and Klein, R. (1997): *SALOME: A bidirectional branch-and-bound procedure for assembly line balancing*, INFORMS Journal on Computing, 9(4), pp. 319–334.
   > Morrison, D. R., Jacobson, S. H. and Sauppe, J. J. (2014): *Branch, bound, and remember: A new algorithm for the simple assembly line balancing problem*, INFORMS Journal on Computing.

---

## 5. Step-by-Step Instructions to Download and Extract

To reproduce or set up the data directory on a new machine:

1. **Create Target Directory Structure:**
   ```bash
   mkdir -p data/raw/scholl1993 data/raw/otto2013 data/docs
   ```

2. **Download Documentation:**
   * Download `https://assembly-line-balancing.de/wp-content/uploads/2017/01/format-ALB.pdf` into `data/docs/format-ALB.pdf`.
   * Download `https://assembly-line-balancing.de/wp-content/uploads/2017/01/Scholl-1993-ALBData.pdf` into `data/docs/Scholl-1993-ALBData.pdf`.

3. **Download and Extract Scholl (1993) Benchmark:**
   * Download `https://assembly-line-balancing.de/wp-content/uploads/2017/01/SALBP-data-sets.zip`.
   * Extract its contents into `data/raw/scholl1993/`.
   * Verify that `data/raw/scholl1993/` contains `SALBP data sets.xlsx` and `precedence graphs/` with the 25 `.IN2` files and `README.DOC`.

4. **Download and Extract Otto et al. (2013) Benchmark:**
   * Download `https://assembly-line-balancing.de/wp-content/uploads/2017/01/SALBP_benchmark.zip`.
   * Extract its contents into `data/raw/otto2013/`.
   * Verify that the subdirectories contain `.ALB` instance files and associated SALOME result tables.

5. **Ensure Raw Data is Ignored by Git:**
   * Verify that `.gitignore` contains `data/raw/` so instance files and archives are not checked into version control.
