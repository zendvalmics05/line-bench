# Mathematical & Heuristic Rule Specifications (Phase 0.6 & 0.7)

This document formalizes the definitions, equations, lower bounds, priority rules, and the analytical validation formula according to standard literature (Scholl 1999; Boysen et al. 2007; Otto et al. 2013).

---

## 1. Six Priority Rules & Two Directions (Phase 0.6)

In a station-oriented constructive heuristic for SALBP-1, station $s=1$ is opened. At each step, a candidate set of eligible tasks $E$ is formed:
$$E = \{ j \notin \text{Assigned} \mid \text{All predecessors of } j \text{ are assigned, and } t_j \le c - \text{load}(s) \}$$
If $E$ is non-empty, the task $j^* \in E$ maximizing the priority function $\pi(j)$ is selected and assigned to station $s$. If $E = \emptyset$, station $s+1$ is opened. The process terminates when all tasks are assigned.

### Priority Rules ($\pi(j)$):
1. **RPW (Ranked Positional Weight - Helgeson & Birnie, 1961):**
   $$\pi_{\text{RPW}}(j) = t_j + \sum_{k \in \text{AllSuccessors}(j)} t_k$$
2. **UBT (Upper Bound on Task Time / Maximum Task Time - Jackson, 1956):**
   $$\pi_{\text{UBT}}(j) = t_j$$
3. **MTS (Maximum Total Successors / Most Followers):**
   $$\pi_{\text{MTS}}(j) = |\text{AllSuccessors}(j)|$$
4. **MDS (Maximum Direct Successors / Immediate Followers):**
   $$\pi_{\text{MDS}}(j) = |\text{ImmediateSuccessors}(j)|$$
5. **EST (Earliest Station Bound - Scholl, 1999):**
   $$E(j) = \left\lceil \frac{t_j + \sum_{i \in \text{AllPredecessors}(j)} t_i}{c} \right\rceil$$
   Tasks with smaller $E(j)$ must be scheduled earlier (or higher urgency). We define priority score:
   $$\pi_{\text{EST}}(j) = -E(j) \quad \text{or minimal } E(j)$$
6. **LST (Latest Station Bound - Scholl, 1999):**
   For an upper bound $m_{\text{ub}}$ on stations (e.g., $m_{\text{ub}} = n$ or heuristic solution):
   $$L(j) = m_{\text{ub}} + 1 - \left\lceil \frac{t_j + \sum_{k \in \text{AllSuccessors}(j)} t_k}{c} \right\rceil$$
   Priority score favors tasks with smaller latest station:
   $$\pi_{\text{LST}}(j) = -L(j) \quad \text{or minimal } L(j)$$

### Directions:
* **Forward Direction:** Build stations from $1$ to $m$ using successors and forward precedence.
* **Backward Direction:** Reverse all precedence arrows ($i \to j$ becomes $j \to i$). Run the station-oriented heuristic on the reversed graph (using predecessors as successors). Reversing the resulting station sequence produces a valid forward assignment.

### Tie-Breaking Rule:
* Deterministic Tie-Breaker: When priorities $\pi(j)$ are equal, break ties by **task index $j$ ascending** (stable and fully reproducible).
* Randomized Multi-Pass: Ties or top-ranked candidates are chosen via random sampling with an explicit pseudo-random seed generator (`numpy.random.Generator`).

---

## 2. Lower Bounds on Stations (Phase 0.6)

For any SALBP-1 instance with task times $t_j$ and cycle time $c$:

1. **Lower Bound 1 (Total Work Content Bound):**
   $$LB_1 = \left\lceil \frac{\sum_{j=1}^n t_j}{c} \right\rceil$$
2. **Lower Bound 2 (Half-Cycle Task Counting Bound):**
   Tasks taking strictly more than $c/2$ can never share a station with each other. Tasks taking exactly $c/2$ can share a station at most in pairs:
   $$LB_2 = |\{j \mid t_j > c/2\}| + \left\lceil \frac{|\{j \mid t_j = c/2\}|}{2} \right\rceil$$
3. **Combined Theoretical Lower Bound:**
   $$LB = \max(LB_1, LB_2)$$
   *Optimality condition:* If any heuristic or exact assignment achieves $m = LB$, the solution is mathematically proven optimal.

---

## 3. Analytical Two-Station Throughput Formula (Phase 0.7)

For validating the discrete-event unpaced simulator (Phase 5), the tandem line model with blocking-after-service has a known exact closed-form solution under exponential processing times.

### Specification:
* Line: $K = 2$ stations in series.
* Service times: Independent and identically exponentially distributed with mean service rate $\mu$ (mean service time $1/\mu$).
* Intermediate buffer capacity: $b$ units (where $b=0$ denotes zero-buffer / direct transfer blocking).
* Supply & demand: Infinite supply before Station 1; infinite space after Station 2.

### Exact Closed-Form Steady-State Throughput:
From classical tandem queueing / Markovian transfer line theory (e.g., Hunt 1956, Gershwin 1994, Buzacott & Shanthikumar 1993):
$$\text{Throughput}(b, \mu) = \mu \cdot \frac{b + 2}{b + 3}$$

* **Zero-buffer ($b=0$):**
  $$\text{Throughput}(0, \mu) = \frac{2}{3} \mu \approx 0.6667 \mu$$
* **Buffer $b=1$:**
  $$\text{Throughput}(1, \mu) = \frac{3}{4} \mu = 0.75 \mu$$
* **Buffer $b=2$:**
  $$\text{Throughput}(2, \mu) = \frac{4}{5} \mu = 0.80 \mu$$

*Use in Project:* The unpaced event recursion engine in Phase 5 will be simulated under these exact parameters, and the empirical throughput will be tested to match $\mu (b+2)/(b+3)$ within confidence intervals.
