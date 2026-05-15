# APGI Mathematical Specification

Complete mathematical backbone of APGI (Allostatic Precision-Gated Ignition):

- Signal normalization → precision weighting → threshold competition
- Nonlinear ignition (logistic + hard threshold)
- Dynamic threshold adaptation (energy vs information)
- Multi-scale coupling + stochastic dynamics
- Reservoir, Kuramoto oscillators, and thermodynamic grounding
- Serotonergic threshold modulation, active inference action loop
- Statistical validation via Lorentzian spectral superposition (§12)

**See also:** `docs/APGI-Specs.md` for the complete 55-section specification.

---

## 0. GLOBAL DEFINITIONS

Core variables used everywhere:

| Variable | Meaning |
| -------- | ------- |
| $x$ | sensory input |
| $\hat{x}$ | prediction |
| $\epsilon$ | prediction error |
| $z$ | standardized prediction error |
| $\Pi$ | precision (inverse variance) |
| $\beta$ | somatic bias |
| $\theta_t$ | dynamic threshold |
| $S_t$ | accumulated signal |
| $B_t$ | ignition probability |

---

## 1. SIGNAL PREPROCESSING

### 1.1 Prediction Error

$$\epsilon = x - \hat{x}$$

### 1.2 Running Mean / Variance (windowed)

$$\mu_t = \frac{1}{T} \sum_{i=t-T}^{t} \epsilon_i$$

$$\sigma_t^2 = \frac{1}{T} \sum_{i=t-T}^{t} (\epsilon_i - \mu_t)^2$$

### 1.3 Z-Score Standardization

$$z = \frac{\epsilon - \mu_t}{\sigma_t}$$

Separate channels: $z_e, z_i$

---

## 2. PRECISION SYSTEM

### 2.1 Precision Definition

$$\Pi = \frac{1}{\sigma^2}$$

### 2.2 Effective Interoceptive Precision

$$\Pi_i^{\text{eff}} = \Pi_i^{\text{baseline}} \cdot \exp(\beta \cdot M(c,a))$$

### 2.3 Precision Dynamics (ODE)

$$\frac{d\Pi_\ell}{dt} = -\frac{\Pi_\ell}{\tau_\Pi} + \alpha|\epsilon_\ell| + C_{\text{down}}(\Pi_{\ell+1} - \Pi_\ell) + C_{\text{up}} \cdot \psi(\epsilon_{\ell-1})$$

---

## 3. CORE APGI SIGNAL

### 3.1 Accumulated Signal

$$S_t = \Pi_e |z_e| + \Pi_i^{\text{eff}} |z_i|$$

---

## 4. IGNITION MECHANISM

### 4.1 Logistic Ignition Probability

$$B_t = \frac{1}{1 + \exp(-\alpha(S_t - \theta_t))}$$

### 4.2 Hard Ignition Condition

$$\text{Ignition} = \mathbb{1}(S_t > \theta_t)$$

### 4.3 Margin

$$\Delta_t = S_t - \theta_t$$

---

## 5. CONTINUOUS-TIME DYNAMICS

### 5.1 Signal Dynamics

$$\frac{dS_t}{dt} = -\frac{S_t}{\tau_S} + \Pi_e |z_e| + \beta \cdot \Pi_i |z_i| + \eta_S(t)$$

### 5.2 Threshold Dynamics

$$\frac{d\theta_t}{dt} = \gamma(\theta_0 - \theta_t) + \delta \cdot B_{t-1} - \lambda \cdot \frac{dS_t}{dt}$$

---

## 6. DISCRETE ALLOSTATIC UPDATE

### 6.1 Cost–Value Threshold Update

$$\theta_{t+1} = \theta_t + \eta(C_{\text{metabolic}} - V_{\text{information}})$$

---

## 7. ENERGY / THERMODYNAMIC LAYER

### 7.1 Metabolic Cost

$$C_{\text{metabolic}} = \kappa \cdot (\text{bits erased})$$

### 7.2 Landauer Limit

$$E_{\text{min}} \geq kT \ln(2)$$

---

## 8. STOCHASTIC DIFFERENTIAL EQUATION (SIMULATION CORE)

### 8.1 Euler–Maruyama

$$X_{t+1} = X_t + \mu(X_t, t)dt + \sigma(X_t, t)\sqrt{dt} \cdot \mathcal{N}(0,1)$$

---

## 9. LIQUID NEURAL NETWORK (RESERVOIR)

### 9.1 Reservoir Dynamics

$$\dot{x}(t) = -\frac{x(t)}{\tau(t)} + f(W_{\text{res}} x(t) + W_{\text{in}} u(t))$$

### 9.2 Signal Readout

$$S(t) = x(t)^T x(t)$$

### 9.3 Suprathreshold Amplification

$$\frac{dx}{dt} = -\alpha x + \cdots + A \cdot x \cdot [S - \theta_t]_+$$

Where:

$$[x]_+ = \max(0, x)$$

---

## 10. HIERARCHICAL SYSTEM

### 10.1 Level Count

$$N_{\text{levels}} = \frac{\log(\tau_{\text{max}} / \tau_{\text{min}})}{\log(\text{overlap})}$$

### 10.2 Cross-Level Threshold Modulation

$$\theta_{t,\ell}(t) = \theta_{0,\ell} \cdot [1 + \kappa_{\text{down}} \Pi_{\ell+1} \cos(\phi_{\ell+1})]$$

### 10.3 Bottom-Up Cascade

$$\theta_{t,\ell} \leftarrow \theta_{t,\ell} \cdot [1 - \kappa_{\text{up}} H(S_{\ell-1} - \theta_{\ell-1})]$$

---

## 11. OSCILLATORY / PHASE COUPLING

### 11.1 Phase Signal

$$\phi_\ell(t) = \omega_\ell t + \phi_0$$

### 11.2 Phase Coupling Influence

$$\cos(\phi_{\ell+1})$$

---

## 12. POST-IGNITION RESET

### 12.1 Reset Rule

$$S_t \leftarrow \rho \cdot S_t$$

$$\theta_t \leftarrow \theta_t + \delta$$

---

## 13. STATISTICAL VALIDATION

### 13.1 Power Spectrum (1/f) — Lorentzian Superposition

$$S_\theta(f) = \sum_\ell \frac{\sigma_\ell^2 \tau_\ell^2}{1 + (2\pi f \tau_\ell)^2}$$

Each Lorentzian has corner frequency $f_{c,\ell} = 1/(2\pi\tau_\ell)$.
The spectrum is 1/f between $f_{c,\text{max}}$ and $f_{c,\text{min}}$.
Above $f_{c,\text{max}}$ all terms collapse to white noise — fit only within the transition band.

Goodness-of-fit metric: **log-domain R²** (threshold: R²_log > 0.3).

### 13.2 Hurst Exponent (DFA)

$$H = \frac{\beta_{\text{spec}} + 1}{2}$$

Healthy range for pink noise: $H \in [0.7, 1.0]$ (Mandelbrot/Peng 1994 DFA literature).
Implemented via `dfa_analysis()` in `stats/hurst.py`.

---

## 14. COMPLETE PIPELINE (ORDER OF EXECUTION)

Step-by-step dependency chain:

### 1. Compute prediction error

$$\epsilon = x - \hat{x}$$

### 2. Standardize

$$z = (\epsilon - \mu) / \sigma$$

### 3. Compute precision

$$\Pi = 1 / \sigma^2$$

### 4. Apply somatic bias

$$\Pi_i^{\text{eff}} = \Pi_i \cdot e^{\beta M}$$

### 5. Compute signal

$$S_t = \Pi_e |z_e| + \Pi_i^{\text{eff}} |z_i|$$

### 6. Update threshold

ODE or discrete rule

### 7. Compute ignition

$$B_t = \sigma(\alpha(S_t - \theta_t))$$

### 8. Apply reset if ignition

$$S_t \rightarrow \rho S_t$$

### 9. Update system dynamics (SDE / reservoir / Kuramoto)

### 10. Apply neuromodulatory gains

- ACh → $\Pi_e^{\text{eff}} = g_{\text{ACh}} \cdot \Pi_e$
- NE  → $\Pi_i^{\text{eff}} = g_{\text{NE}} \cdot \Pi_i$ (or threshold offset)
- DA  → $z_i^{\text{eff}} = z_i + \beta_{\text{DA}}$
- 5-HT → $\theta^{\text{eff}} = \theta + \beta_{\text{5HT}}$

### 11. Hierarchical cascade (if enabled)

$$\theta_\ell = \theta_{0,\ell} \cdot [1 + \kappa_{\text{down}} \Pi_{\ell+1} \cos(\phi_{\ell+1})]$$

### 12. Active inference action selection (if enabled)

$$a^* = \operatorname{argmin}_{k} F(a_k) = w_e F_e + w_p F_p + w_m F_m$$

### 13. Maturity assessment (§15)

Evaluates hierarchical architecture score (§8) and statistical validation score (§12)
on recorded history. See `stats/maturity_assessment.py`.
