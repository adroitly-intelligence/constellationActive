## Planning for Convex-Optimized Frozen Orbit Design

## 1. Objective
Replace the traditional, unstable Broyden-Rosengren root-finding approach for near-equatorial frozen orbits with a **Successive Convex Optimisation (SCO)** framework. The goal is to minimise the **Total Variation (TV) norm** of the *mean-element* eccentricity vector over a mission horizon using a high-fidelity propagator (HPOP or J2+J3).

---

## 2. Problem Formulation

### Why near-equatorial frozen orbits are hard
The J3 frozen eccentricity shrinks as inclination decreases:
$$e_f = -\frac{J_3}{2 J_2} \frac{R_E}{a} \sin(i) \;\longrightarrow\; 0 \quad \text{as } i \to 0$$

The J2 short-period oscillation in eccentricity scales independently of inclination:
$$\delta e_{sp} \sim J_2 \left(\frac{R_E}{a}\right)^2 \sim 10^{-3}$$

At near-equatorial inclinations $\delta e_{sp} \gg e_f$, so the raw osculating trajectory is dominated by short-period "noise" — not the secular drift the optimiser should suppress. Applying the TV-norm directly to osculating elements will drive the solver toward a wrong, noise-minimising solution.

### Coordinate choice
Use the standard eccentricity-vector components, evaluated in the **perigee-rotating frame**:
$$\xi_{rot}(t) = e(t)\cos\!\bigl(\omega(t) - \dot{\omega} t\bigr), \qquad \eta_{rot}(t) = e(t)\sin\!\bigl(\omega(t) - \dot{\omega} t\bigr)$$

where $\dot{\omega}$ is the secular J2 argument-of-perigee precession rate. In this frame the frozen fixed-point is stationary at $(0,\, e_f)$, making the TV-norm objective well-conditioned regardless of the beat period.

> **Note for $i < 5°$:** When the inclination is so low that $\omega$ becomes numerically sensitive near $e \to 0$, switch to equinoctial components $(h, k) = (e\cos(\omega+\Omega),\, e\sin(\omega+\Omega))$ and define the rotating-frame rate as $\mathrm{d}(\omega+\Omega)/\mathrm{d}t$. The LP sub-problem is unchanged.

### Design variables
$$\mathbf{z}_0 = [\xi_0,\; \eta_0]^T = [e_0 \cos\omega_0,\; e_0 \sin\omega_0]^T \quad (\text{osculating, at } t=0 \text{ the rotating frame coincides with the inertial frame})$$

### Objective function — TV-norm on *mean-filtered* trajectory

**Critical modification vs. the original formulation:** apply a one-orbit running-mean filter to the rotating-frame trajectory before computing the TV-norm. This isolates secular/long-period drift from unavoidable short-period oscillations:

$$\bar{\xi}_k = \frac{1}{n_\text{orb}} \sum_{j=k}^{k+n_\text{orb}-1} \xi_{rot}(t_j), \qquad n_\text{orb} = \left\lfloor T_\text{orb}/\Delta t \right\rfloor$$

$$\min_{\mathbf{z}_0} \quad J = \sum_{k=1}^{N_\text{mean}-1} \left( |\bar{\xi}_{k+1} - \bar{\xi}_k| + |\bar{\eta}_{k+1} - \bar{\eta}_k| \right)$$

where $N_\text{mean} = N - n_\text{orb} + 1$ is the number of valid mean-filtered epochs.

### Constraints
$$e_\text{min} \leq \sqrt{\xi_0^2 + \eta_0^2} \leq e_\text{max}, \qquad h_p > h_{p,\min}$$

### Dynamics (propagator interface)
$$\mathbf{z}_{rot}(t_k) = \mathcal{F}\bigl(\mathbf{z}_0,\, t_k;\; \text{force model}\bigr)$$

**Force model options**

| Mode | Force model | Cost per propagation | Use case |
|---|---|---|---|
| Fast | J2 + J3 zonal | < 2 s | Development / iteration tuning |
| High-fidelity | $J_{70\times70}$, SRP, Moon/Sun | 10 – 60 s | Mission-final design |

---

## 3. Implementation — Successive Convex Optimisation Loop

### Phase 1 — Analytical seed
Compute initial $(e_f, \omega_f = 90°)$ from the Coffey-Deprit formula via `frozen_orbit.py`, then convert to initial $(xi_0, \eta_0)$.

### Phase 2 — Perigee-rotating-frame propagation
At each SCO iteration $j$:
1. Propagate from $\mathbf{z}_0^{(j)}$ over $N_\text{orbits} \cdot T_\text{orb}$.
2. Transform to rotating frame at rate $\dot{\omega}$.
3. Apply one-orbit mean filter → $\bar{\mathbf{z}}_k^{(j)}$.
4. Compute TV-norm $J^{(j)}$.

### Phase 3 — Finite-difference sensitivity (STM)
Estimate the $2 \times 2$ sensitivity of the mean-filtered state w.r.t. the initial condition using **4 central-difference propagations**:
$$\mathbf{\Phi}_k^{[:,0]} = \frac{\bar{\mathbf{z}}_k(\xi_0+\delta) - \bar{\mathbf{z}}_k(\xi_0-\delta)}{2\delta}, \qquad \mathbf{\Phi}_k^{[:,1]} = \frac{\bar{\mathbf{z}}_k(\eta_0+\delta) - \bar{\mathbf{z}}_k(\eta_0-\delta)}{2\delta}$$

Typical $\delta = 10^{-6}$ (dimensionless). Sensitivity is computed on the mean-filtered trajectory, so it is not contaminated by short-period jitter.

### Phase 4 — Convex LP sub-problem
Linearise: $\bar{\mathbf{z}}_k^{(j+1)} \approx \bar{\mathbf{z}}_k^{(j)} + \mathbf{\Phi}_k\, \Delta\mathbf{z}_0$

Minimise the TV-norm via a standard **Linear Programme** (auxiliary $t$-variable reformulation):
$$\min_{\Delta\mathbf{z}_0,\,\mathbf{t}} \sum_k (t_k^\xi + t_k^\eta)$$
subject to:
$$\pm\bigl[(\bar{\xi}_{k+1}^{(j)} - \bar{\xi}_k^{(j)}) + (\mathbf{\Phi}_{k+1} - \mathbf{\Phi}_k)^{[0,:]}\,\Delta\mathbf{z}_0\bigr] \leq t_k^\xi$$
$$\pm\bigl[(\bar{\eta}_{k+1}^{(j)} - \bar{\eta}_k^{(j)}) + (\mathbf{\Phi}_{k+1} - \mathbf{\Phi}_k)^{[1,:]}\,\Delta\mathbf{z}_0\bigr] \leq t_k^\eta$$
$$\|\Delta\mathbf{z}_0\|_\infty \leq \delta_\text{trust} \quad \text{(trust region)}$$
$$e_\text{min} - e_\text{cur} \leq \nabla e \cdot \Delta\mathbf{z}_0 \leq e_\text{max} - e_\text{cur} \quad \text{(linearised eccentricity bounds)}$$

Solver: `scipy.optimize.linprog` with the HiGHS backend (no additional dependencies).

### Phase 5 — Trust-region update
1. Apply $\mathbf{z}_0^{(j+1)} = \mathbf{z}_0^{(j)} + \Delta\mathbf{z}_0$ and propagate to get $J^{(j+1)}_\text{actual}$.
2. Compute reduction ratio $\rho = (J^{(j)} - J^{(j+1)}_\text{actual}) / (J^{(j)} - J^{(j+1)}_\text{pred})$.
3. Accept step if $\rho \geq 0.1$; grow trust region if $\rho \geq 0.75$.
4. **Caching:** on an accepted step, reuse the new propagation as the nominal for the next FD computation (saves one propagation). On a rejected step, reuse both the nominal trajectory **and** the Phi matrices and only re-solve the LP with the updated (smaller) $\delta_\text{trust}$ — no extra propagations needed.

### Phase 6 — Convergence
Terminate when $\|\Delta\mathbf{z}_0\| < \varepsilon$ on an accepted step.

---

## 4. Verification Steps
- **Jacobian check:** compare FD Phi against a second FD with $\delta/2$; relative agreement should be $< 10^{-3}$.
- **J2+J3 sanity:** run SCO with the J3-only propagator; the converged $(e_f, \omega_f)$ should match the Coffey-Deprit analytical value to $< 10^{-7}$.
- **Rosengren comparison:** for a mid-inclination orbit the SCO and Rosengren outputs should agree to $< 10^{-8}$.
- **Full perturbations:** enable HPOP; the TV-norm of the SCO result should be lower than that of the analytical J2+J3 seed.
- **Long-term stability:** 1-year HPOP propagation of the optimised state; perigee altitude variation should be bounded.

---

## 5. Visualisation & Plotting
- **Phase-space plot:** $\bar{\xi}_{rot}$ vs. $\bar{\eta}_{rot}$ (mean-filtered rotating frame) — tight cluster vs. drifting spiral.
- **Argument of perigee vs. time:** $\omega$ should show bounded oscillation around $90°$ or $270°$.
- **TV-norm convergence:** $J^{(j)}$ vs. SCO iteration.
- **Eccentricity envelope:** $e(t)$ confirming secular stability.

---

## 6. Optimisation Tips
- **Scaling:** $(\xi_0, \eta_0)$ are already dimensionless and $O(10^{-3}\text{–}10^{-4})$; normalise to $O(1)$ if the LP solver reports poor conditioning.
- **$N_\text{orbits}$ schedule:** start with 5–10 orbits (fast convergence region), switch to 15–30 for final fine-tuning.
- **$\delta_\text{fd}$ choice:** $10^{-6}$ balances truncation and round-off for double-precision eccentricity components; verify with a step-halving check.
- **L1-norm robustness:** the $L_1$ TV-norm is insensitive to isolated large oscillations (J2 short-period spikes) that would inflate an $L_2$ objective.

---

## 7. Computational Cost

### Per-iteration breakdown

| Step | Propagations | Cost (J2+J3) | Cost (HPOP) |
|---|---|---|---|
| Nominal (first iter or reject-cycle start) | 1 | ~2 s | ~30 s |
| FD central differences | 4 | ~8 s | ~120 s |
| Actual-reduction evaluation | 1 | ~2 s | ~30 s |
| LP sub-problem | 0 | < 1 ms | < 1 ms |
| **Total (cold start)** | **6** | **~12 s** | **~180 s** |
| **Total (accepted, with cache)** | **5** | **~10 s** | **~150 s** |
| **Total (rejected, with cache)** | **0** | **< 1 ms** | **< 1 ms** |

> Costs above assume $N_\text{orbits} = 10$, $T_\text{orb} \approx 5800$ s, $\Delta t = 60$ s.  
> HPOP uses $J_{70\times70}$ + Sun/Moon; wall-clock varies with integrator step-size control.

### Recommended settings by use-case

| Use case | `N_orbits` | `dt` [s] | Force model | Typical total time |
|---|---|---|---|---|
| Development / rapid iteration | 5–10 | 60 | J2+J3 | 1–3 min |
| Production design | 15–20 | 60 | J2+J3 then HPOP | 5–15 min |
| Mission verification | 30–90 | 30 | HPOP full | 1–4 hr |

### Mean-filter overhead
The running mean is implemented as a box convolution via `np.convolve(mode='valid')` — $O(N_\text{samples})$ with negligible cost compared to any propagation step.

### LP sub-problem size
$N_\text{mean} \approx N_\text{orbits} \cdot T_\text{orb}/\Delta t - n_\text{orb} + 1 \approx 870$ for the default settings. The LP has $2 + 2 N_\text{mean} \approx 1740$ variables and $\approx 3500$ constraints — solved in $< 5$ ms by HiGHS.
