# APGI — Thermodynamic Grounding of Conscious Access

APGI models consciousness as the brain's embodied, metabolically constrained "emergency-broadcast" system: deployed only when local predictive policies fail, the signal is sufficiently reliable, and the local ATP/glucose microdomains can afford the thermodynamic price of making that information globally available. The framework conceptualizes the brain's tracking of reality within four strict biological limits.

---

## 1. Thermodynamic Limits & Metabolic Costs: The "Price" of Reality

Conscious access is metabolically expensive. APGI grounds its computational claims in physical reality via **Landauer's Principle**, which sets the minimum thermodynamic cost of irreversible information erasure:

$$E \geq kT\ln 2 \quad \text{(per bit).}$$

**The $\kappa$ parameter (Landauer overhead).** At 310 K (37 °C), the Landauer minimum is $kT\ln 2 \approx 2.97 \times 10^{-21}$ J/bit $\approx 0.06$ ATP/bit (taking the standard-state $\Delta G^\circ_{\text{ATP}} \approx -30.5$ kJ/mol; the in-vivo value $\Delta G \approx -50$ kJ/mol gives $\approx 0.036$ ATP/bit). Biological neural systems operate far from equilibrium to buy speed and noise-robustness, so APGI defines a falsifiable **dimensionless overhead parameter** $\kappa$ — the factor by which biological bit-commitment exceeds the Landauer floor. APGI's point estimate is $\kappa \approx 1{,}700$, an effective cost of $\approx 100$ ATP/bit (plausible range 10–1,000 ATP/bit).
*Epistemic status: thermodynamic level — conceptually grounded but unmeasured at the hardware scale; an aspirational bridge, not an established quantity.*

**The cost of ignition.** A full frontoparietal–thalamic ignition event involves $\sim 10^{9}$ spikes, costing on the order of $10^{18}$ ATP molecules *(order-of-magnitude estimate; per-event cost unmeasured)*. This should manifest empirically as a **5–10% regional metabolic increment** (CMRglc or BOLD fMRI) in frontoparietal networks during conscious access relative to matched unconscious baselines.

**Falsification.** If PET/fMRI measurements imply an effective per-bit cost **outside 10–10,000 ATP/bit** ($\kappa$ outside $\approx$ 170–170,000) across multiple paradigms, the thermodynamic grounding of the APGI threshold-update rule is falsified. The band is tiered: 10–1,000 ATP/bit is the *prediction*; 1,000–10,000 would force *model revision*; outside 10–10,000 *disconfirms*.

---

## 2. Allostatic Regulation: Predictive Energy Management

The ignition threshold $\theta_t$ is not a static filter; it is an **allostatically regulated control variable**. Unlike homeostasis, which reacts to current depletion, allostasis proactively adjusts set-points against anticipated demand.

**The allostatic ODE.** The threshold adapts continuously as a cost–benefit differential:

$$\frac{d\theta_t}{dt} = \frac{\theta_0 - \theta_t}{\tau_\theta} + \eta_\theta\,\big(C_{\text{metabolic}}(t) - V_{\text{information}}(t)\big),$$

where $\tau_\theta$ is the mean-reversion timescale, $\eta_\theta$ is the cost–benefit gain, $C_{\text{metabolic}}(t)$ is the metabolic cost pressure (rising as the locally available budget $M(t)$ falls under fatigue, caloric restriction, or circadian trough), and $V_{\text{information}}(t)$ is the anticipated value of the signal. Consistent with the canonical neuromodulator mapping, **LC-NE arousal sets the gain $\eta_\theta$ (NE → $\lambda_\theta$)** rather than entering as an independent additive drive: $\eta_\theta$ governs how sharply the threshold tracks the cost–benefit imbalance, while $\tau_\theta$ — a mechanistically distinct quantity — governs recovery speed.

**The topological "why."** To see why ignition is sharp and discontinuous rather than a smooth ramp, model the global activation state $x$ on a **cusp-catastrophe potential**:

$$V(x) = \tfrac{1}{4}x^4 - \tfrac{1}{2}S_t\,x^2 + \theta_t\,x,$$

where $S_t = \Pi\cdot|\varepsilon|$ is the precision-weighted prediction error (the **splitting factor**) and $\theta_t$ is the allostatic threshold (the **normal / control factor**). For a given $S_t$, a high $\theta_t$ leaves $V(x)$ with a single minimum — the unconscious state — so no jump can occur. As $\theta_t$ falls allostatically (metabolic surplus or high anticipated value), the control point $(-S_t,\ \theta_t)$ crosses the cusp's bifurcation set; the landscape becomes bistable and the system jumps discontinuously into the ignited (high-activation) attractor. The surprise $S_t$ sets where that fold lies — larger precision-weighted error lowers the $\theta_t$ required to ignite — so the construction reproduces the gate $S_t > \theta_t$ as a catastrophe rather than a graded threshold. Because the ignition (down-sweep of $\theta_t$) and extinction (up-sweep) folds differ, the model yields the **hysteresis** signature predicted independently in Innovation 4, and supplies a rigorous Thomian account of why "emergency broadcast" is all-or-none.
*Modeling status: analogical normal form — a topological description of the transition, not a claim of literal quartic cortical dynamics.*

**The arbitration rule.** The cost–benefit logic that *sets* $\theta_t$ can be stated as an explicit thermodynamic optimization. Global ignition fires **if and only if** the expected information gain outweighs its thermodynamic price:

$$\Delta I_{\text{surprise}} \cdot \kappa \cdot kT\ln 2 \;<\; V_{\text{information}}(t)\cdot M(t).$$

On the left, $\Delta I_{\text{surprise}}$ (bits) is the expected information committed to broadcast — metabolically penalized by the mandatory erasure/overwriting of pre-existing localized predictive distributions across the frontoparietal workspace — $\kappa$ the dimensionless Landauer overhead, and $kT\ln 2$ the Landauer quantum (J/bit); their product is the effective energetic **cost** of the broadcast (joules). On the right, $V_{\text{information}}(t)$ is the dimensionless **value** of that information and $M(t)$ the locally available metabolic **budget** (joules); their product is the value-weighted energy the microdomain can afford. The allostatic threshold $\theta_t$ is the **dynamic shadow price** of this constraint: the ODE drives $\theta_t$ so that the gate $S_t > \theta_t$ coincides with *cost < value-weighted budget*. By binding Landauer's Principle directly to allostatic gating, the brain avoids thermodynamic bankruptcy — the "emergency-broadcast" system is deployed only when the informational return on investment justifies the local microdomain expenditure.

**Falsification.** If metabolic challenges (glucose depletion, circadian-phase manipulation, caloric restriction) fail to elevate $\theta_t$ after controlling for general arousal, the allostatic claim is falsified and the model reduces to a static-threshold account.

---

## 3. ATP/Glucose Microdomains: Localized Fueling of Ignition

Ignition is not a uniform whole-brain event; it is constrained by localized energy availability. The framework ties threshold computation to the cerebral metabolic rate of glucose (CMRglc) and to local microdomains (e.g., the astrocyte–neuron lactate shuttle supporting intense synaptic firing).

**Central vs. peripheral metabolism.** APGI predicts that $\theta_t$ elevation following a metabolic challenge reflects a **central** mechanism: equivalent peripheral perturbations that do not alter cerebral glucose metabolism (e.g., insulin clamps that do not cross the blood–brain barrier) should **not** elevate $\theta_t$.

**Implication for psychopathology.** Disorders marked by metabolic dysfunction (mitochondrial disease, severe neuroinflammation) should show specific disruptions in ignition capacity — elevated $\theta_t$, reduced P3b amplitude — because the local ATP microdomains required to sustain the near-critical $\rho_{\text{crit}} \approx 1$ regime are compromised.

---

## 4. Neural Avalanches & Self-Organized Criticality: The Dynamics of Efficiency

To maximize computational work per unit energy, the cortex operates **near criticality** — the dynamical signature of a system balanced between order and chaos.

**Spectral radius ($\rho_{\text{crit}} \approx 1$).** The recurrent weight matrix of the cortical liquid network operates in the upper sub-critical regime, $\rho \in [0.7, 0.95]$ (canonical range, Innovation 4), approaching but never reaching $\rho = 1$. This maximizes dynamic range, memory capacity, and sensitivity, letting small precision-weighted signals trigger large processing cascades without runaway excitation; $\rho \geq 1$ violates the echo-state (fading-memory) property and is excluded as biologically implausible.

**Power-law avalanche distributions.** Near criticality, ignition-cascade sizes $s$ follow a scale-free distribution,

$$P(s) \propto s^{-\alpha_{\text{size}}}, \qquad \alpha_{\text{size}} \approx 1.5.$$

**1/f (pink-noise) dynamics.** The superposition of five nested Ornstein–Uhlenbeck regulatory loops (spanning milliseconds to years) generates long-range temporal correlations in threshold fluctuations, yielding a Hurst exponent $H \approx 0.85\text{–}0.95$ (equivalently DFA $\alpha_{\text{DFA}} \approx 0.85\text{–}0.95$). A single OU loop is Markovian, with a Lorentzian spectrum; only the *superposition* of loops with appropriately spaced time constants produces the fractal 1/f structure that gives the threshold a "memory" of past metabolic and informational states, preventing memoryless Markovian drift.

> **Notation.** $\alpha_{\text{size}}$ (avalanche size exponent $\approx 1.5$) and $\alpha_{\text{DFA}}$ (fluctuation exponent $\approx 0.85\text{–}0.95$) are distinct quantities; the subscripts resolve the $\alpha$ collision. $\rho$ denotes the spectral radius and is kept distinct from any branching-ratio measure $\sigma$.

**Falsification.** If cascade-size distributions depart from power-law form (exponential truncation or bimodality), or if the DFA exponent of the threshold time series falls **below 0.55** (indicating Markovian, white-noise dynamics), the self-organized-criticality claim is falsified.

---

## How the Brain Tracks Reality Under These Limits

APGI posits that the brain tracks reality through **cost–benefit arbitration**:

1. **Bottom-up signal.** A prediction error $\varepsilon$ is generated.
2. **Precision weighting ($\Pi$).** Acetylcholine scales the error by its reliability (ACh → $\Pi$), yielding $S_t = \Pi\cdot|\varepsilon|$. Norepinephrine acts separately, setting the gain $\eta_\theta$ on threshold adaptation (NE → $\lambda_\theta$) rather than scaling the error itself.
3. **Allostatic gating ($\theta_t$).** The system tests whether accumulated surprise $S_t$ exceeds the dynamic threshold $\theta_t$ — which is set, via the arbitration rule, by locally available ATP/glucose $M(t)$ and the anticipated value of the information $V_{\text{information}}(t)$.
4. **Critical ignition.** If $S_t > \theta_t$, the network — already poised at $\rho_{\text{crit}} \approx 1$ — crosses the cusp bifurcation set and undergoes a discontinuous phase transition (neural avalanche), triggering a $\sim$5–10% metabolic increment and broadcasting the content globally.
