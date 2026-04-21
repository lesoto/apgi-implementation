# APGI System: Full Mathematical Specification

## 0. Core Variables

At time $t$:

| Variable | Description |
| -------- | ----------- |
| $z_e(t)$ | exteroceptive prediction error |
| $z_i(t)$ | interoceptive prediction error |
| $\Pi_e(t), \Pi_i(t)$ | precisions (inverse variances) |
| $S(t)$ | accumulated (precision-weighted) signal |
| $\theta(t)$ | dynamic ignition threshold |
| $B(t) \in \{0,1\}$ | ignition state |
| $\beta(t)$ | dopaminergic bias term |
| $C(t)$ | metabolic cost |
| $V(t)$ | expected information value |

---

## 1. Prediction Errors

### 1.1 Raw Prediction Errors

$$z_e(t) = x_e(t) - \hat{x}_e(t)$$

$$z_i(t) = x_i(t) - \hat{x}_i(t)$$

### 1.2 Optional Normalization (recommended)

$$\tilde{z}_e(t) = \frac{z_e(t)}{\sigma_e(t) + \epsilon}$$

$$\tilde{z}_i(t) = \frac{z_i(t)}{\sigma_i(t) + \epsilon}$$

---

## 2. Precision (Π) — Core Mechanism

### 2.1 Definition (Inverse Variance)

$$\Pi_e(t) = \frac{1}{\sigma_e^2(t) + \epsilon}$$

$$\Pi_i(t) = \frac{1}{\sigma_i^2(t) + \epsilon}$$

### 2.2 Online Variance Update (EMA)

$$\sigma_e^2(t+1) = (1 - \alpha_e)\sigma_e^2(t) + \alpha_e z_e^2(t)$$

$$\sigma_i^2(t+1) = (1 - \alpha_i)\sigma_i^2(t) + \alpha_i z_i^2(t)$$

### 2.3 Neuromodulatory Gain

#### Acetylcholine (ACh)

$$\Pi_e^{\text{eff}}(t) = g_{\text{ACh}}(t) \cdot \Pi_e(t)$$

#### Norepinephrine (NE)

$$\Pi_i^{\text{eff}}(t) = g_{\text{NE}}(t) \cdot \Pi_i(t)$$

### 2.4 Dopamine Bias (Important correction)

Dopamine should NOT multiply precision globally.

**Instead:**

$$z_i^{\text{eff}}(t) = z_i(t) + \beta(t)$$

**or alternatively:**

$$S(t) = \Pi_e^{\text{eff}}|z_e| + \Pi_i^{\text{eff}}|z_i| + \beta(t)$$

---

## 3. Precision-Weighted Signal Accumulation

### 3.1 Instantaneous Signal

$$S_{\text{inst}}(t) = \Pi_e^{\text{eff}}(t) \cdot |z_e(t)| + \Pi_i^{\text{eff}}(t) \cdot |z_i^{\text{eff}}(t)|$$

### 3.2 Temporal Integration (Leaky Accumulator)

$$S(t+1) = (1 - \lambda)S(t) + \lambda S_{\text{inst}}(t)$$

$\lambda \in (0,1)$: integration rate

### 3.3 Optional Nonlinearity (Stabilization)

$$S(t) \leftarrow \log(1 + S(t))$$

---

## 4. Dynamic Threshold $\theta(t)$

### 4.1 Core Update Rule

$$\theta(t+1) = \theta(t) + \eta[C(t) - V(t)] + \delta \cdot B(t)$$

### 4.2 Metabolic Cost

**Simple version:**

$$C(t) = c_0 + c_1 \cdot S(t)$$

**More realistic:**

$$C(t) = c_1 S(t) + c_2 B(t-1)$$

### 4.3 Information Value

$$V(t) = \mathbb{E}[|z_e(t)| + |z_i(t)|]$$

**Practical approximation:**

$$V(t) = v_1 |z_e(t)| + v_2 |z_i(t)|$$

### 4.4 NE Modulation of Threshold

$$\theta(t) \leftarrow \theta(t) \cdot (1 + \gamma_{\text{NE}} \cdot g_{\text{NE}}(t))$$

---

## 5. Ignition Condition (Phase Transition)

### 5.1 Hard Threshold

$$
B(t) = \begin{cases}
1 & \text{if } S(t) > \theta(t) \\
0 & \text{otherwise}
\end{cases}
$$

### 5.2 Smooth (Recommended for Simulation)

$$P_{\text{ignite}}(t) = \sigma\left(\frac{S(t) - \theta(t)}{\tau}\right)$$

$$B(t) \sim \text{Bernoulli}(P_{\text{ignite}}(t))$$

---

## 6. Refractory Dynamics

### 6.1 Threshold Boost After Ignition

$$\theta(t+1) \leftarrow \theta(t+1) + \delta \cdot B(t)$$

### 6.2 Optional Decay

$$\theta(t+1) = \theta_{\text{base}} + (\theta(t) - \theta_{\text{base}})e^{-\kappa}$$

---

## 7. Multi-Timescale Extension (Core APGI)

This is where your system becomes APGI instead of simple gating.

### 7.1 Timescale Hierarchy

$$\tau_i = \tau_0 \cdot k^i$$

### 7.2 Multi-Scale Errors

$$z_e^{(i)}(t), \quad z_i^{(i)}(t)$$

### 7.3 Multi-Scale Integration

$$\Phi_i(t+1) = \left(1 - \frac{1}{\tau_i}\right)\Phi_i(t) + \frac{1}{\tau_i}z(t)$$

### 7.4 Cross-Scale Aggregation

$$S(t) = \sum_i w_i \cdot \Pi_i(t) \cdot |\Phi_i(t)|$$

### 7.5 Weights

$$w_i = \frac{1}{Z} \cdot k^{-i}$$

---

## 8. Spectral Biomarker (1/f)

### 8.1 Power Spectrum

$$P(f) \propto \frac{1}{f^\beta}$$

### 8.2 Estimation

$$\beta = -\frac{d\log P(f)}{d\log f}$$

---

## 9. Hurst Exponent

### 9.1 Definition

$$H \in (0.5, 1)$$

### 9.2 Approximation

$$H \approx \frac{\beta + 1}{2}$$

---

## 10. Full System Update (One Step)

**Pipeline:**

| Step | Operation |
| ---- | --------- |
| 1 | $z_e, z_i$ |
| 2 | $\sigma^2 \rightarrow \Pi$ |
| 3 | $\Pi_{\text{eff}}$ |
| 4 | $S_{\text{inst}}$ |
| 5 | $S(t)$ |
| 6 | $C(t), V(t)$ |
| 7 | $\theta(t+1)$ |
| 8 | $B(t)$ |

---

## ⚠️ Important Corrections to Your Framework

### 1. Precision must be stable

Clamp it:

$$\Pi \in [\Pi_{\text{min}}, \Pi_{\text{max}}]$$

### 2. Avoid double-counting NE

NE should affect **precision OR threshold** — not both strongly.

### 3. Dopamine ≠ precision

You correctly noted this—keep it as:

- bias
- reward prediction error

### 4. Don't hardcode $\varphi$ yet

Use:

$$k \in [1.3, 2.0]$$

Test $\varphi$ later
