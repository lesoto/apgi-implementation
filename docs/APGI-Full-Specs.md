# **Allostatic Precision-Gated Ignition Framework**

---

## **1. Core Theoretical Definition**

In the **Allostatic Precision-Gated Ignition (APGI) Framework**, *allostatic* is not a modifier but the **theoretical claim**: the ignition threshold **θₜ** is **actively regulated**—not fixed—by **metabolic state, interoceptive afference, and circadian dynamics**. This distinguishes APGI from every existing **Global Neuronal Workspace (GNW)** variant and from the **Free Energy Principle's (FEP)** underspecified threshold.

**Conscious access is:**

- hierarchical
- recurrent
- precision-weighted
- metabolically constrained
- neuromodulated

**Conscious access** is a **metastable critical-state transition** emerging from **predictive processing dynamics** operating across **coupled cortical and interoceptive systems**.

### **Formal Definition:**

A **transient large-scale metastable attractor transition** produced when **precision-weighted prediction error accumulation**—integrated across **exteroceptive and interoceptive channels**—exceeds an **allosterically regulated energetic–informational threshold θₜ** under **recurrent competitive dynamics** and **phase-coherent synchrony gating**.

---

## **2. Marr's Hierarchy Positioning**

APGI operates at the **intersection of the algorithmic and computational levels**. It specifies:

- **What information the brain must represent**: *precision-weighted prediction errors relative to a dynamically adjusted threshold*.
- **Without displacing implementational descriptions** (e.g., Hodgkin–Huxley, LIF networks) of *how* that computation is physically realized (Marr, 1982).

**Phenomenal consciousness** lies **outside** the framework's explanatory scope. APGI provides a **necessary computational precondition** for empirical progress, **not** a solution to the hard problem.

---

## **3. High-Level Architecture**

The APGI architecture integrates **ten interacting computational components**:


| **Component**                   | **Function**                                                         |
| ------------------------------- | -------------------------------------------------------------------- |
| Hierarchical predictive coding  | Multi-level generative inference across cortical hierarchy           |
| Precision weighting (Π)         | Reliability-sensitive modulation of prediction error gain            |
| Recurrent attractor dynamics    | Ignition amplification via winner-take-all competition               |
| Oscillatory synchrony gate (Γ)  | Long-range binding via phase-amplitude coupling (PAC)                |
| Neuromodulation                 | Dynamic gain, salience, and motivational control (NE, ACh, DA, 5-HT) |
| Allostatic metabolic regulation | Energetic constraint optimization via adaptive **θₜ**                |
| Critical dynamics               | Edge-of-criticality computation near spectral radius **ρ ≈ 1**       |
| Reservoir substrate             | High-dimensional temporal memory via liquid-state dynamics           |
| Active inference loop           | Action-driven future prediction error minimization                   |
| Competitive ignition            | Global workspace selection via inhibitory competition                |


### **Relationship to FEP**

APGI is a **constrained implementation** of **Friston's Free Energy Principle (FEP)** (Friston, 2010), **not a competitor**. The FEP establishes:

- Hierarchical predictive processing with precision weighting.

APGI **adds two empirical commitments** the FEP leaves underspecified:

1. **Conscious access** corresponds specifically to **discrete suprathreshold ignition**, not continuous free-energy minimization.
2. **θₜ is allosterically regulated** rather than a free parameter.

**APGI = FEP + ignition threshold + allostatic regulator.**

---

## **4. Multi-Scale Cortical Hierarchy**

The cortex is modeled as a **hierarchy of levels**:  
**l ∈ {1, 2, …, L}**


| **Level (l)** | **Representative Processing**                                       |
| ------------- | ------------------------------------------------------------------- |
| l = 1         | Sensory feature layers (edge, frequency, somatic signal detection)  |
| l = 2         | Object and perceptual representations (categorical perception)      |
| l = 3         | Situational and narrative models (context, body-schema)             |
| l = L         | Abstract and global self-modeling (autobiographical, metacognitive) |


Each level **l** contains **five functional elements**:

- **xₜ⁽ˡ⁾** — latent state vector
- **f⁽ˡ⁾(·)** — top-down prediction generator
- **εₜ⁽ˡ⁾** — signed prediction error estimator
- **Πₜ⁽ˡ⁾** — dynamic precision estimator
- **W_rec⁽ˡ⁾** — recurrent attractor field

---

## **5. Hierarchical Predictive Coding**

For each level **l**, three quantities are computed at each timestep:

1. **State representation**:
  **xₜ⁽ˡ⁾**
2. **Top-down prediction**:
  **x̂ₜ⁽ˡ⁻¹⁾ = f⁽ˡ⁾(xₜ⁽ˡ⁾)**
3. **Signed prediction error (bottom-up)**:
  **εₜ⁽ˡ⁾ = xₜ⁽ˡ⁻¹⁾ − x̂ₜ⁽ˡ⁻¹⁾**

The **signed formulation** replaces the **unsigned |ε|** of earlier APGI iterations. The **core ignition equation** maintains backward compatibility:  
**Sₜ = Πᵉ·|εᵉ| + Πⁱ·|εⁱ| > θₜ**  
*(valid as the first-order approximation in the symmetric valence case)*

---

## **6. Signed Nonlinear Prediction Error Transformation**

The **unsigned |ε|** formulation treats **rewarding** and **threatening** prediction errors identically. Empirically, **signed dopaminergic prediction errors** in midbrain populations are **asymmetric**:

- **Approach-motivating positive errors** → phasic **DA bursts**
- **Aversive negative errors** → distinct **NE and CRF pathways** (Niv et al., 2012; Schultz et al., 1997).

The **signed transformation** preserves this asymmetry within the **precision-weighted ignition framework**:

**φ(ε) =**

- **α⁺ · tanh(γ⁺ · ε)** if **ε ≥ 0** *(reward / approach signal)*
- **α⁻ · tanh(γ⁻ · ε)** if **ε < 0** *(threat / avoidance signal)*

### **Parameter Bounds and Derivation**

The four parameters (**α⁺, α⁻, γ⁺, γ⁻**) are constrained by **two boundary conditions**:

- **α⁺, α⁻ ∈ [0.5, 2.0]**:
  - Below **0.5**, valence asymmetry is indistinguishable from the symmetric case in standard psychophysical measurement (**Δ < 0.5 SD in HEP amplitude**).
  - Above **2.0**, the gain exceeds empirically observed ceilings for appetitive vs. aversive salience differences (Pessiglione et al., 2006).
- **γ⁺, γ⁻ ∈ [1.0, 5.0]**:
  - Below **1.0**, the **tanh** approximates a linear function, losing the **sigmoidal saturation property** required for phase-transition dynamics.
  - Above **5.0**, the function degenerates to a **step**, eliminating graded precision modulation.

**Compatibility Note:**  
The **symmetric first-order APGI equation** (**Sₜ = Πᵉ·|εᵉ| + Πⁱ·|εⁱ| > θₜ**) is recovered when **α⁺ = α⁻** and **γ⁺ = γ⁻**. All earlier APGI papers remain valid under this symmetric approximation. The **extended signed form** applies specifically to **affective and interoceptive paradigms** where valence asymmetry is operationally relevant.

### **Preserved Biological Properties**

1. **Valence polarity** — approach and avoidance circuits receive differentially weighted signals.
2. **Reward/threat asymmetry** — gain parameters can differ across hemispheres and arousal states.
3. **Excitatory/inhibitory distinction** — positive errors drive ignition; negative errors can suppress it.
4. **Motivational directionality** — dopaminergic and noradrenergic projections are functionally distinguished.

### **Falsification Criterion**

- **α⁺/α⁻ asymmetry ratio** must differ significantly from **1.0** (i.e., the symmetric case) in a **pre-registered paradigm** contrasting appetitive vs. aversive near-threshold stimuli (**n ≥ 40, BF > 10** for asymmetric over symmetric model).
- **γ⁺ and γ⁻** must fall within **[1.0, 5.0]** across healthy participants. Values **systematically outside this range** falsify the parameter derivation.

---

## **7. Temporal Precision Estimation**

Precision is **dynamically inferred** at each level via a **Kalman-like running uncertainty tracker**:

1. **Running uncertainty (Bayesian variance tracking)**:
  **Σₜ⁽ˡ⁾ = β_Σ · Σₜ₋₁⁽ˡ⁾ + (1 − β_Σ) · φ(εₜ⁽ˡ⁾)²**
2. **Precision (inverse variance with stability floor)**:
  **Πₜ⁽ˡ⁾ = 1 / (Σₜ⁽ˡ⁾ + ε_stab)**

The **stability constant ε_stab > 0** prevents division by zero during high-precision states and corresponds biologically to **irreducible noise in synaptic transmission**.

This formulation approximates:

- Bayesian uncertainty tracking
- Kalman-like confidence updating
- Cortical reliability estimation across timescales.

**Acetylcholine modulates Πₜ⁽ˡ⁾ directly via gain scaling**: cholinergic blockade should **reduce precision-weight gain** without altering the running uncertainty accumulation dynamics—a **pharmacologically dissociable prediction**.

---

## **8. Neuromodulatory Gain System**

### **Four-Modulator Dissociation Architecture**

APGI maps **four neuromodulatory systems** to **four mechanistically distinct parameters**. Each mapping is a **testable hypothesis** carrying explicit evidence ratings, not an established fact. The **pharmacological dissociation matrix** (Pillar 8 of the Empirical Credibility Roadmap) provides the experimental infrastructure for testing all four simultaneously.

#### **8.1 Acetylcholine — Sensory Precision (Πₛₑₙₛ)**

**Πₛₑₙₛᵉᶠᶠ = g_ACh(t) · Πₛₑₙₛ**

- **Basal forebrain cholinergic projections** modulate cortical sensory precision and expected-uncertainty encoding (Yu & Dayan, 2005).
- **Function**: Sharpens sensory reliability; suppresses distractors; increases feedforward gain.
- **Prediction**: Cholinergic blockade (**scopolamine**) reduces **Πₛₑₙₛᵉᶠᶠ**, raising detection threshold for exteroceptive stimuli **without equivalently affecting interoceptive precision or θₜ**.

#### **8.2 Norepinephrine — Arousal and Salience Gain**

**Πₛₐₗᵉᶠᶠ = g_NE(t) · Πₛₐₗ**

- The **locus coeruleus–norepinephrine system** modulates global cortical gain and signal-to-noise ratio, governing **threshold adaptation dynamics** (Aston-Jones & Cohen, 2005).
- **Function**: Adaptive arousal; interrupt handling; novelty amplification.
- **Prediction**: NE depletion (**α-methyltyrosine**) flattens threshold adaptation rate **η_θ** without equivalently affecting **Πₛₑₙₛ**, providing the primary pharmacological falsification test for the parameter-specific neuromodulatory mapping.

#### **8.3 Dopamine — Motivational Bias**

**z_intᵉᶠᶠ = z_int + β_DA(t)**

- **Mesolimbic dopamine** encodes reward prediction errors and motivational salience (Schultz et al., 1997).
- **Function**: Expected reward weighting; motivational salience; goal relevance bias.
- **Mechanistic Disambiguation**: **β_DA** acts on the **ignition accumulation variable z_int**, not on **θₜ**. Motivational salience therefore increases ignition probability by **boosting the accumulated signal**, not by **lowering the threshold**.
  - **Testable Prediction**: DA agonists should shift the psychometric detection function **leftward** (same-threshold, higher-gain) rather than **downward** (lower threshold, same gain).

#### **8.4 Serotonin — Patience, Temporal Persistence, and Aversive Prediction Error Suppression**

**θₑᶠᶠ = θ + β_5HT(t)**

- **Theoretical Grounding**: The serotonergic role in threshold modulation is **more contested** than the NE/ACh mappings and is flagged as a **high-uncertainty claim**. The most defensible account (Dayan & Daw, 2008; Crockett et al., 2012) positions serotonin in **aversive prediction error processing** and **temporal discounting**—**patience under uncertainty**.
- **APGI Mapping**: This is mapped to **θₑᶠᶠ** because **patience under uncertainty** is computationally equivalent to maintaining a **higher ignition threshold** against premature commitment to a percept.
- **Function**: Impulse suppression; temporal persistence; uncertainty tolerance.
- **Limitation**: This mapping is **not as strongly established** as the ACh and NE mappings. It should be treated as a **prospective empirical commitment** rather than a settled mechanistic claim. Failure of **5-HT manipulations** to shift **θₑᶠᶠ** in the predicted direction would revise this component of the framework **without affecting the core ignition dynamics**.

### **Four-Way Pharmacological Dissociation Falsification**

- **Atomoxetine** (NE reuptake inhibitor) must **selectively raise Πₛₐₗ** without equivalently shifting **Πₛₑₙₛ or θₜ**.
- **Scopolamine** must reduce **Πₛₑₙₛ** without affecting **θₜ or z_int**.
- A **DA agonist** must shift detection function gain **without shifting threshold**.
- An **SSRI** must shift **θₑᶠᶠ** without equivalently affecting precision parameters.
- **All four predictions must be pre-registered**; failure in any cell of the dissociation matrix constitutes **partial falsification** of the corresponding mapping.

---

## **9. Oscillatory Synchrony Gate**

### **Relationship to PAC Mechanism**

**Phase-Amplitude Coupling (PAC)** as **Hierarchical Precision Weighting** establishes that **theta/alpha phase modulating gamma amplitude** is the cortical mechanism implementing hierarchical precision weighting. The **oscillatory synchrony gate Γ⁽ˡ⁾(t)** defined here is the **operationalization** of that PAC mechanism at the network level: it quantifies the degree to which **inter-regional phase coherence** enables **gamma-band precision signals** to propagate across the hierarchy. These are **not competing formalisms**—**Γ⁽ˡ⁾(t)** measures the network-level gate opened by PAC.

### **Pairwise Phase Coherence**

**G_φ(i, j, t) = cos(φᵢ(t) − φⱼ(t))**

### **Level-Specific Synchrony Index**

**Γ⁽ˡ⁾(t) = (1/N²) · Σᵢ,ⱼ G_φ(i, j, t)**

> **Note**: The phase variable **φ** in this section is a **real-valued oscillatory phase angle (radians)**, not to be confused with **IIT's Φ (integrated information)** or APGI's earlier hierarchical level index notation.

The synchrony gate models **three empirically documented phenomena**:

1. **Gamma synchrony** during conscious access (Engel & Singer, 2001).
2. **Beta feedback coupling** stabilizing prediction (Bastos et al., 2015).
3. **Theta-gamma nesting** implementing the PAC precision mechanism (Jensen & Lisman, 2005).

### **Falsification Criterion**

- **Inter-level Γ⁽ˡ⁾(t) coherence** must increase significantly during conscious access relative to near-miss trials (same stimulus, no conscious report) in a **pre-registered EEG contrast (d > 0.5, n ≥ 30)**.
- If **Γ⁽ˡ⁾(t)** is statistically equivalent between conscious and non-conscious conditions across **two independent datasets**, the oscillatory synchrony gate does **not** contribute independent explanatory variance and should be **absorbed into the PAC precision weighting term**.

---

## **10. Recurrent Competitive Attractor Dynamics**

Each hierarchical level contains **recurrent neural populations** governed by the update equation:

### **Recurrent State Update**

**xₜ₊₁⁽ˡ⁾ = f(  
  W_rec⁽ˡ⁾ · xₜ⁽ˡ⁾

- W_td⁽ˡ⁾ · xₜ⁽ˡ⁺¹⁾   (top-down feedback)
- W_bu⁽ˡ⁾ · xₜ⁽ˡ⁻¹⁾   (bottom-up input)  
  − Iₜ⁽ˡ⁾               (inhibitory stabilization)
- ξₜ                (stochastic neural noise)  
)**

### **Competitive Inhibition**

**Iₜ⁽ˡ⁾ = W_inh⁽ˡ⁾ · Σₖ xₖ⁽ˡ⁾**

### **Biological Correspondence**

- **W_rec** implements **lateral excitatory connectivity**.
- **W_inh** implements **PV+ interneuron-mediated divisive normalization** (Carandini & Heeger, 2012), providing the **gain-control mechanism** that prevents runaway reverberation.

### **Laminar Prediction**

- **Precision signals** are encoded in **superficial-layer feedback projections**.
- **Prediction errors** emerge in **granular input layers**—testable with laminar probes in neurosurgical patients (Bastos et al., 2012).

### **Functional Consequences**

1. **Winner-take-all competition** — only one attractor basin stabilizes per ignition event.
2. **Attractor stabilization** — once crossed, the ignition state is maintained against perturbation.
3. **Selective ignition** — precision-weighted signals, not raw signal amplitude, determine the winner.

---

## **11. Criticality Regulation**

The system operates near the **edge-of-criticality regime**, where computational properties are maximized.

### **Spectral Radius**

**ρ_crit⁽ˡ⁾ = ρ(W_rec⁽ˡ⁾)** *(spectral radius of recurrent weight matrix)*

> **Note**: **κ** is reserved in the AD biomarker document for the **metabolic coupling coefficient (κ ≈ 10⁻¹⁶ mol ATP/bit)** from the Landauer principle. The spectral radius is hereafter designated **ρ_crit⁽ˡ⁾** throughout all APGI documents.

### **Optimal Ignition**

**ρ_crit⁽ˡ⁾ ≈ 1.0**

Operation near **ρ_crit ≈ 1** maximizes **five computational properties** simultaneously:

1. **Dynamic range** — the system can respond to inputs across many orders of magnitude.
2. **Information integration** — signals from distant regions can mutually influence each other.
3. **Metastability** — multiple quasi-stable states coexist, enabling rapid attractor transitions.
4. **Sensitivity** — small precision-weighted signals can trigger ignition near threshold.
5. **Memory capacity** — maximal for reservoir computation (Maass et al., 2002).

### **Three Simultaneous Phase-Transition Signatures**

APGI predicts that ignition near **ρ_crit ≈ 1** produces:

1. **Discontinuity in integrated information (Φ)** at threshold crossing (**Cohen's d > 0.8**).
2. **Elevated susceptibility** — variance of the order parameter exceeding baseline by ratio **> 1.5**.
3. **Critical slowing** — autocorrelation time constant **τ_auto** increasing by **> 20%** as the system approaches **θₜ**.

**All three must co-occur**; failure on any single signature across **two independent datasets** constitutes **partial falsification** requiring revision of the phase-transition claim.

---

## **12. Precision-Weighted Salience Computation**

For each hierarchical level, **instantaneous salience** integrates **precision, signed prediction error, and oscillatory synchrony**:

### **Level-Specific Instantaneous Salience**

**S_inst⁽ˡ⁾(t) = Πₜ⁽ˡ⁾ · φ(εₜ⁽ˡ⁾) · Γ⁽ˡ⁾(t)**

### **Global Salience (Hierarchically Weighted Sum)**

**S_global(t) = Σₗ wₗ · S_inst⁽ˡ⁾(t)**

The **level weights wₗ** are **not free parameters** but are constrained by **two conditions**:

1. **Σwₗ = 1** *(normalization)*.
2. **wₗ** should reflect the **informational value contribution** of each level estimated from prior trials.

In practice, **wₗ** can be **initialized uniformly** and updated via **gradient-free Bayesian optimization** across paradigm runs—a **principled parameter estimation procedure** that prevents post-hoc fitting.

---

## **13. Leaky Metastable Ignition Accumulation**

**S(t+1) = (1 − λ) · S(t) + λ · S_global(t)**

### **Timescale Separation**

- **Surprise accumulates rapidly** (**λ_S ≈ 2–5 s⁻¹**, corresponding to a time constant of **0.2–0.5 s**).
- **Threshold adapts slowly** (**λ_θ ≈ 0.01–0.1 s⁻¹**, time constant **10–100 s**)—a **two-orders-of-magnitude separation** that is recoverable from behavioral data in psychophysical paradigms.

If model fitting yields **overlapping time constant distributions** for the two processes, the **functional separation is not empirically grounded**, and the **ODE formulation requires revision**.

### **Functional Properties**

1. **Temporal evidence integration** — weak signals accumulate across time to reach threshold.
2. **Working-memory persistence** — near-threshold states are maintained within the suprathreshold window.
3. **Metastable buildup** — the system dwells near **θₜ** before committing to ignition.
4. **Hysteresis** — the signal level required to trigger ignition differs from the level sustaining it, a signature of **bistability**.

---

## **14. Metabolic–Energetic Constraint System**

### **Biological Grounding**

No existing consciousness theory **quantitatively links ignition to cellular metabolism**. This is **APGI's most original empirical claim**.

The **metabolic constraint** follows from **Lennie (2003)**:

- Sustained ignition events consume **disproportionate ATP** relative to the brain's energy budget (**~20 W total; ~0.06 ATP/bit** yielding **~1,700× biological overhead** above the Landauer limit).
- Metabolic constraint is therefore **not merely a computational convenience** but a **biologically motivated prediction**.

### **Metabolic Cost Function**

**C(t) =**

- **c₁ · S(t)** *(signal accumulation cost)*
- **+ c₂ · B(t−1)** *(prior-ignition refractory cost)*
- **+ c₃ · A(t)** *(action execution cost)*

### **Informational Value**

**V(t) = Σₗ vₗ · |φ(εₜ⁽ˡ⁾)|**

APGI predicts that **θₜ elevation following metabolic challenge** reflects a **central rather than peripheral mechanism**. **Equivalent metabolic perturbation not affecting cerebral glucose metabolism** should **not elevate θₜ**, requiring **pharmacological dissociation** via **insulin clamp protocols**.

The appropriate neural substrate is **cerebral metabolic rate of glucose (CMRglc)**; candidate mechanisms include:

- **Astrocytic lactate shuttle**
- **Direct neuronal ATP depletion** *(Lennie, 2003; Attwell & Laughlin, 2001 offer competing accounts that APGI does not adjudicate)*.

Stating the mechanism as specifically **"astrocyte-neuron energetics"** without qualification would be **premature**.

---

## **15. Adaptive Threshold Dynamics (Allostatic Regulation)**

The threshold **θₜ** is an **active computational variable**, not a static filter. Its **continuous-time ODE** (from the canonical APGI formulation):

**dθₜ/dt = (θ₀ − θₜ)/τ_θ + η_θ · (C_metabolic − V_information)**

### **Discrete-Time Implementation with Motivational Override**

**θ(t+1) = θ(t)**

- **+ η_θ · [C(t) − V(t)]** *(metabolic–informational cost-benefit term)*
- **+ δ_fatigue · B(t)** *(post-ignition refractory elevation)*
- **− β_DA(t) · Δ_motivation** *(dopaminergic motivational modulation)*

The **motivational modulation term − β_DA(t) · Δ_motivation** is the **same dopaminergic parameter β_DA**. Dopamine acts on **θ(t+1)** via the **threshold ODE**, not independently through **z_int**. This resolves the potential **double-counting ambiguity**:

- The **DA term** defines the biological parameter.
- The **threshold ODE** specifies where it acts mechanistically.
- There is **one DA variable**, **two equations that reference it**, **one causal pathway**.

### **η_θ vs. τ_θ Dissociation**

- **η_θ** is a **gain parameter** on the cost-benefit differential—**large η_θ** produces **sharp threshold responses** to cost-benefit imbalances; **small η_θ** produces **sluggish adaptation** regardless of informational value.
- This is **mechanistically distinct** from **τ_θ**, the **mean-reversion timescale** governing baseline recovery speed.

### **Falsification**

- **Noradrenergic gain manipulation (atomoxetine)** should produce **steeper threshold-crossing functions** (elevated **η_θ**) **without changing baseline recovery speed** (**τ_θ unchanged**)—a dissociation testable in pharmacological paradigms.

### **Allostatic Regulation Falsification**

- **θₜ dynamics** must be **significantly modulated** by metabolic challenge (**glucose depletion, circadian phase manipulation, caloric restriction**) after controlling for arousal, in a **pre-registered study (d > 0.5, n ≥ 30)**.
- **Invariance of θₜ** to metabolic challenge across **all three manipulations** in a well-powered sample **falsifies APGI's allostatic claim** and requires revision to a **static-threshold model**.

---

## **16. Stochastic Ignition Decision**

**Deterministic threshold crossing** cannot account for:

- Perceptual variability
- Spontaneous fluctuations near threshold
- Binocular rivalry dynamics

The **Bernoulli formulation** captures these phenomena while maintaining the **all-or-none ignition phenomenology** that distinguishes APGI from **graded accumulation models**.

### **Ignition Probability**

**P(Bₜ = 1) = σ( [S(t) − θ(t)] / τ_σ )**

### **Ignition Sampling**

**Bₜ ~ Bernoulli( P(Bₜ = 1) )**

The parameter **τ_σ** controls the **sharpness of the stochastic transition**:

- **Small τ_σ** approximates a **deterministic step function**.
- **Large τ_σ** produces **graded probabilistic access**.

### **Key Prediction**

- **β ≥ 10** (steepness parameter in the psychometric function equivalent) indicates **phase-transition dynamics** rather than graded accumulation.

The **full psychometric function**:  
**P(seen) = 1 / (1 + exp(−β · (Π · |ε| − θₜ)))**  
where **β = 1/τ_σ**.

This formulation captures:

- Noisy cortical ignition
- Spontaneous fluctuations producing perceptual rivalry
- Probabilistic access near threshold
- The **all-or-none phenomenology** of conscious access documented by Sergent & Dehaene (2004).

---

## **17. Post-Ignition Refractory Reset**

Following a successful ignition event (**Bₜ = 1**), two reset operations execute:

1. **S(t) ← ρ_S · S(t)** *(partial signal decay, **ρ_S ∈ [0.1, 0.5]**)*
2. **θ(t) ← θ(t) + δ_refractory** *(temporary threshold elevation)*

### **Biological Basis**

The **refractory period** corresponds to the **post-ignition suppression** documented in EEG:

- The **~500 ms attentional blink window** during which a second target fails to reach conscious access (Sergent et al., 2005).
- **δ_refractory** elevates **θₜ** into this window.
- **ρ_S** prevents runaway reverberation by **partially resetting S(t)** without destroying the post-ignition broadcast state.

### **Without Refractory Reset**

The APGI framework predicts **pathological seizure-like ignition cascades**, directly falsifiable via **TMS perturbation studies**.

### **ρ_S Parameter Bounds**

- **ρ_S < 0.1** produces **near-complete signal erasure**, eliminating working-memory maintenance within the ignition window *(incompatible with masking paradigm data)*.
- **ρ_S > 0.5** produces **insufficient reset**, lengthening the effective attentional blink beyond observed **~500 ms**.

These bounds are derived from existing **attentional blink data** (Olivers & Meeter, 2008).

---

## **18. Reservoir Dynamical Substrate**

The **reservoir readout S_res(t)** contributes to **global salience** as a **weighted additive term**, formally integrating **reservoir temporal memory** into the ignition pipeline.

### **Reservoir State Evolution**

**r(t+1) = f( W_res · r(t) + W_in · u(t) + ξₜ )**

### **Reservoir Readout**

**S_res(t) = W_out · r(t)**

### **Integration into Global Salience**

**S_global(t) = Σₗ wₗ · S_inst⁽ˡ⁾(t) + w_res · S_res(t)**

### **Spectral Radius Constraint**

The reservoir operates with **ρ(W_res) ∈ [0.7, 0.95]**, spanning the **upper sub-critical regime** consistent with the **echo state property** (**ρ < 1** required for fading memory). Values **≥ 1.0** produce **non-fading dynamics** incompatible with temporal integration requirements and are excluded as **biologically implausible** (Jaeger & Haas, 2004).

### **Functional Capacities**

The reservoir provides **four functional capacities** absent from hierarchical predictive coding alone:

1. **Temporal memory** — fading memory trace extending integration window beyond membrane time constants.
2. **Nonlinear expansion** — input space mapped to high-dimensional state space enabling complex temporal discriminations.
3. **Metastable trajectories** — transient high-dimensional dynamics supporting working memory during suprathreshold states.
4. **Transient attractor generation** — brief input patterns stabilized into sustained cortical representations.

### **Reservoir Substrate Falsification**

- **LNN architecture** must outperform **standard RNNs and feedforward networks** on a **pre-specified APGI benchmark battery** (temporal integration, bistability, precision-dependent ignition) **without architectural augmentation** *(pre-registered, n-simulation > 100 per architecture)*.
- If **augmented alternatives** match LNN performance, the **intrinsic-advantage claim** is falsified.

---

## **19. Active Inference Action Loop**

### **Integration with FEP**

The **active inference component** implements the **action-selection dimension** of the **Free Energy Principle**: the organism acts to **minimize expected future surprise**, closing the **perception-action loop** that purely afferent accounts of ignition leave open.

### **Policy Selection**

**aₜ = argmin_a E[F(a)]**  
where **F(a) = Expected Free Energy**

### **Actions Feed Back into Ignition Dynamics Through Three Channels**

1. **Sensory consequence** — actions generate future sensory input, modifying **εₜ⁽ˡ⁾** on subsequent trials.
2. **Interoceptive consequence** — motor actions change metabolic state **M**, modulating **θₜ** via allostatic feedback.
3. **Uncertainty reduction** — epistemic actions (eye movements, exploratory behavior) reduce **Σₜ⁽ˡ⁾**, increasing **Πₜ⁽ˡ⁾**.

The organism is computing the **probability of external states** and **actively sampling** to minimize future prediction error. The **ignition threshold θₜ** therefore adapts not only to **current metabolic state** but to **expected future informational value**—a prediction confirmed by **foraging and decision-making paradigms** showing that **uncertainty-reducing actions** are preferentially selected even at metabolic cost (Friston et al., 2015).

---

## **20. Full Ignition Pipeline**

The **complete APGI ignition cycle** executes in **17 ordered steps per timestep**:


| **Step** | **Operation**                                                                             |
| -------- | ----------------------------------------------------------------------------------------- |
| 1        | Sensory and interoceptive input **u(t)** acquired                                         |
| 2        | Reservoir state updated: **r(t+1) = f(W_res · r(t) + W_in · u(t) + ξₜ)**                  |
| 3        | Top-down predictions generated: **x̂ₜ⁽ˡ⁻¹⁾ = f⁽ˡ⁾(xₜ⁽ˡ⁾)** for each level **l**           |
| 4        | Signed prediction errors computed: **εₜ⁽ˡ⁾ = xₜ⁽ˡ⁻¹⁾ − x̂ₜ⁽ˡ⁻¹⁾**                         |
| 5        | Signed nonlinear transform applied: **φ(εₜ⁽ˡ⁾)** via asymmetric **tanh**                  |
| 6        | Running uncertainty updated: **Σₜ⁽ˡ⁾ = β_Σ · Σₜ₋₁⁽ˡ⁾ + (1−β_Σ) · φ(ε)²**                  |
| 7        | Precision computed: **Πₜ⁽ˡ⁾ = 1 / (Σₜ⁽ˡ⁾ + ε_stab)**                                      |
| 8        | Neuromodulatory gain applied: **ACh → Πₛₑₙₛ; NE → Πₛₐₗ; DA → z_int; 5-HT → θ**            |
| 9        | Oscillatory synchrony gate computed: **Γ⁽ˡ⁾(t) = (1/N²) Σᵢ,ⱼ cos(φᵢ−φⱼ)**                 |
| 10       | Level salience computed: **S_inst⁽ˡ⁾(t) = Πₜ⁽ˡ⁾ · φ(εₜ⁽ˡ⁾) · Γ⁽ˡ⁾(t)**                    |
| 11       | Global salience integrated: **S_global(t) = Σₗ wₗ S_inst⁽ˡ⁾ + w_res S_res(t)**            |
| 12       | Leaky accumulation: **S(t+1) = (1−λ) S(t) + λ S_global(t)**                               |
| 13       | Metabolic cost and informational value computed                                           |
| 14       | Adaptive threshold updated via allostatic ODE                                             |
| 15       | Ignition probability: **P(Bₜ=1) = σ([S(t)−θ(t)] / τ_σ)**                                  |
| 16       | Stochastic ignition sampled: **Bₜ ~ Bernoulli(P(Bₜ=1))**                                  |
| 17       | *(if Bₜ=1)* Global broadcasting; active inference action update; refractory reset; repeat |


---

## **21. Biological Mapping**

The following mappings range from **well-established** (predictive coding hierarchy, LC-NE gain control) to **plausible but contested** (metabolic regulation mechanisms) to **prospective hypotheses** (reservoir dynamics in cortical microcircuits). Each entry carries an **implicit evidence rating**; the **neuromodulatory mappings** carry the **highest confidence**; the **metabolic and reservoir mappings** should be treated as **prospective empirical commitments**.


| **APGI Component**                     | **Candidate Neurobiology**                                                                                                                 | **Evidence Rating** |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------- |
| Prediction hierarchy                   | Cortical predictive coding, laminar feedforward/feedback circuits (Bastos et al., 2012)                                                    | HIGH                |
| Precision weighting (Πₛₑₙₛ)            | Pulvinar + basal forebrain ACh gain modulation (Yu & Dayan, 2005)                                                                          | HIGH                |
| Precision weighting (Πₛₐₗ)             | LC-NE system (Aston-Jones & Cohen, 2005)                                                                                                   | HIGH                |
| Recurrent ignition                     | Frontoparietal loops; delayed P3b > 300 ms post-stimulus (Dehaene & Changeux, 2011)                                                        | HIGH                |
| Oscillatory synchrony gate             | Gamma/beta coupling; theta-gamma PAC (Engel & Singer, 2001)                                                                                | MEDIUM-HIGH         |
| Motivational bias (β_DA)               | Mesolimbic dopamine; midbrain RPE neurons (Schultz et al., 1997)                                                                           | HIGH                |
| Patience / threshold offset (β_5HT)    | Dorsal raphe serotonergic projections; temporal discounting (Dayan & Daw, 2008)                                                            | MEDIUM              |
| Metabolic regulation                   | CMRglc; candidate mechanisms: astrocytic lactate shuttle **AND/OR** direct neuronal ATP depletion (Lennie, 2003; Attwell & Laughlin, 2001) | MEDIUM              |
| Reservoir dynamics                     | Recurrent cortical microcircuits; columnar architecture (Maass et al., 2002)                                                               | MEDIUM              |
| Criticality                            | Cortical avalanche dynamics; scale-free neural activity (Beggs & Plenz, 2003)                                                              | MEDIUM-HIGH         |
| Active inference / action minimization | Prefrontal-motor networks; basal ganglia policy selection (Friston et al., 2015)                                                           | MEDIUM              |


---

## **22. Empirical Predictions with Quantitative Benchmarks**

Each prediction below is stated with a **quantitative benchmark** and a **falsification criterion**. Verbal predictions without numerical thresholds are **not included** per APGI's own methodological standard.

### **22.1 Criticality Proximity**

**Prediction**: Conscious ignition peaks at **ρ_crit ≈ 1.0** *(spectral radius near unity)*.

- Neural avalanche size distributions should follow **power-law scaling with exponent −1.5** during conscious access (Beggs & Plenz, 2003).
- Deviations from **ρ_crit ≈ 1** should predict **both degraded conscious access AND degraded computational performance** in simultaneous behavioral tasks.

**Falsification**:

- Avalanche size distributions that are **exponential (sub-critical)** or **flat (super-critical)** during verified conscious access across **N ≥ 20 participants** falsify the criticality claim.

---

### **22.2 Oscillatory Synchrony**

**Prediction**: Ignition requires **increased gamma synchrony, beta feedback stabilization, and long-range phase locking**.

- **Gamma power increase > 30%** from pre-stimulus baseline during suprathreshold vs. subthreshold trials.
- **Beta feedback coherence (top-down: 13–30 Hz)** should increase **before gamma synchrony onset** *(predictive suppression precedes error broadcast)*.
- **Inter-level PAC coherence** must increase significantly: **d > 0.5, n ≥ 30**.

---

### **22.3 Precision Manipulation Effects — Four-Way Pharmacological Dissociation**

**ACh → Πₛₑₙₛ; NE → Πₛₐₗ and η_θ; DA → z_int and threshold via β_DA; 5-HT → θ_eff**

- **Atomoxetine (NE)** should shift detection threshold via **Πₛₐₗ** without altering early sensory ERPs (**N1/P2**)—**null on N1/P2** is a pre-registered positive credibility signal.
- **Propranolol (β-blocker)** should reduce **interoceptive Πⁱ** specifically, without affecting **exteroceptive Πᵉ**—testable by comparing **HEP amplitude vs. N1 amplitude**.
- **Scopolamine (muscarinic antagonist)** and **physostigmine (cholinesterase inhibitor)** should shift precision weighting in **opposite directions** (Yu & Dayan, 2005).

---

### **22.4 Metabolic Suppression**

**Prediction**: Reduced **CMRglc** raises **θₜ**, shortens metastable persistence, impairs conscious integration.

- **Glucose depletion** should elevate detection threshold by **> 15%** from baseline in a well-powered psychophysical study (**n ≥ 40, pre-registered**).
- **Exogenous ketone supplementation (beta-hydroxybutyrate)** should buffer against **θₜ elevation** during caloric restriction—a **striking positive test** of the fuel-source specificity prediction.
- **Peripheral metabolic perturbation not affecting CMRglc** should **NOT elevate θₜ**—this **null prediction** is as diagnostically important as the positive ones.

---

### **22.5 Signed Prediction Error Asymmetry**

**Prediction**: **α⁺/α⁻ ≠ 1** in affective paradigms; gain asymmetry correlates with trait anxiety (**β_som ≈ 0.7**).

- Appetitive near-threshold stimuli should be detected at **higher rates** than matched aversive stimuli when **S_global(t)** is equated across conditions.
- Individual **α⁺/α⁻ asymmetry ratio** should correlate with **interoceptive accuracy scores** (Garfinkel et al., 2015).

---

### **22.6 Reservoir Temporal Complexity**

**Prediction**: Higher reservoir dimensionality improves **temporal integration, conscious capacity, and metastable richness**.

- **Attentional blink duration** should scale predictably with **λ** *(leaky accumulator rate)*.
- **Binocular rivalry alternation rate** should scale with **bistability parameters** near **ρ_crit ≈ 1**.
- **Masking effectiveness** should scale inversely with **temporal integration window τ_S**.
- **All three paradigm predictions** must be fit by a **single parameter set without re-fitting**—a **unified reservoir account**.

---

---

## **23. Relationship to Existing Theories**


| **Theory**                                                                     | **APGI Relationship**                                                                                                                                                                                                                                                        |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Global Workspace Theory** (Baars 1988; Dehaene & Changeux 2011)              | Implements GNW ignition computationally, adds **precision weighting, metabolism, criticality, and probabilistic stochastic sampling** that GNW leaves underspecified. APGI adds the **parameter specification** GNW's architecture leaves open.                              |
| **Free Energy Principle / Active Inference** (Friston 2010)                    | **Constrained implementation**, not a competitor. FEP provides the **predictive processing and precision-weighting machinery**; APGI adds: **(1) discrete ignition threshold, (2) allostatic regulation of θₜ**. **APGI = FEP + ignition threshold + allostatic regulator**. |
| **Integrated Information Theory** (Tononi 2004; Tononi et al. 2016)            | **Orthogonal, not competing**. IIT is **implementational (substrate Φ)**; APGI is **algorithmic (ignition dynamics)**. Productive question: do APGI ignition events correlate with high-Φ states? **Testable** via PyPhi computation on simulated APGI networks.             |
| **Recurrent Processing Theory** (Lamme 2006)                                   | **Strongly compatible**. RPT identifies recurrent processing as sufficient for phenomenal consciousness; APGI specifies the **precision-weighted threshold conditions** under which recurrent dynamics produce global broadcast.                                             |
| **Higher-Order Thought theories** (Rosenthal 2005)                             | **Partially compatible**. APGI's metacognitive **Level ℓ+1** can be interpreted as a higher-order representation; the key difference is that APGI **grounds HOT in measurable inter-level coupling strength**, generating falsifiable EEG predictions.                       |
| **Reservoir / Liquid State Computing** (Maass et al. 2002; Hasani et al. 2021) | Provides the **biological substrate**. LNNs implement **precision-weighted threshold gating and bifurcation dynamics** as intrinsic ODE properties, satisfying APGI's **five biological constraints** without auxiliary mechanisms.                                          |
| **Neural Criticality** (Beggs & Plenz 2003)                                    | **Explicitly integrated**. **ρ_crit ≈ 1** is not a metaphor but a **testable prediction** linking spectral radius to ignition probability and to cortical avalanche statistics.                                                                                              |


---

---

## **24. Computational Advantages Over Existing Frameworks**

### **24.1 Advantages Over Standard GNW**

APGI adds **five components** absent from Dehaene–Changeux GNW:

1. **Precision weighting (Π)** — reliability-sensitive modulation; GNW treats all signals as equally weighted above amplitude threshold.
2. **Allostatic metabolic regulation** — **θₜ adapts**; GNW threshold is static or attention-modulated but not metabolically grounded.
3. **Criticality dynamics** — **ρ_crit ≈ 1** generates testable avalanche statistics; GNW does not specify operating regime.
4. **Oscillatory PAC gate (Γ)** — inter-level synchrony required for ignition; GNW does not specify the binding mechanism.
5. **Probabilistic stochastic ignition** — Bernoulli sampling; GNW predicts all-or-none deterministically, failing to account for trial-by-trial variability at fixed stimulus strength.

---

### **24.2 Advantages Over Predictive Processing Alone**

1. **Explicit ignition transitions** — PP is a continuous minimization process; APGI specifies the **discrete event** constituting conscious access.
2. **Recurrent metastability** — PP accounts lack **bistability dynamics** near **θₜ**.
3. **Global broadcasting mechanism** — PP does not specify the **frontoparietal ignition event**.
4. **Reservoir temporal dynamics** — PP hierarchies operate at **fixed timescales**; reservoir substrate provides **multi-timescale integration** beyond membrane time constants.

---

### **24.3 Advantages Over IIT for Empirical Work**

1. **Operationalizability** — APGI generates **psychometric curves, ERP predictions, pharmacological dissociation matrices** testable with standard neuroscience equipment; **Φ computation** requires full causal architecture knowledge.
2. **Computational implementability** — **17-step pipeline** runs on standard hardware; **PyPhi scales exponentially** with system size.
3. **Biologically grounded dynamics** — APGI parameters map to **measurable biological variables** (CMRglc, HEP amplitude, PCI); IIT makes **no specific metabolic predictions**.
4. **Dynamically testable** — APGI ignition events are **time-stamped and EEG-detectable**; **Φ** is a static substrate property.

---

---

## **25. Computational Complexity**

**Full APGI pipeline per timestep: O(L · N²)**  
where:

- **L** = number of hierarchical levels
- **N** = recurrent population size

### **Reservoir Optimization**

- **Sparse recurrent connectivity** (biological cortex: **~10⁻³ connection probability**) reduces effective complexity to **O(L · N · k)**, where **k** is mean fan-out, typically **k << N**.
- For **neuromorphic implementation** (Intel Loihi, SpiNNaker), **event-driven sparse coding** further reduces to **O(active spikes per timestep)**, achieving **biological energy efficiency**.

### **Full Pipeline Breakdown**


| **Operation**                      | **Complexity**                                        |
| ---------------------------------- | ----------------------------------------------------- |
| Hierarchical prediction generation | O(L · N²)                                             |
| Precision estimation               | O(L · N)                                              |
| Reservoir update                   | O(N_res²) with sparse connectivity → O(N_res · k_res) |
| Neuromodulatory gain application   | O(L · N) — diagonal matrix multiplication             |
| Oscillatory synchrony gate         | O(N²) per level — reducible via spatial clustering    |
| Stochastic ignition sampling       | O(1) per timestep                                     |


---

---

## **26. Minimal Canonical APGI Equation**

The **full ignition dynamics** reduce to a **single summary equation**:

**Bₜ ~ Bernoulli( σ( [Σₗ Πₜ⁽ˡ⁾ · φ(εₜ⁽ˡ⁾) · Γ⁽ˡ⁾(t) + w_res · S_res(t) − θₜ] / τ_σ ) )**

**Subject to four operative constraints:**

1. **Recurrent attractor dynamics** — **xₜ⁺¹⁽ˡ⁾** governed by **W_rec, W_td, W_bu, I_t**.
2. **Allostatic metabolic regulation** — **θₜ** governed by allostatic ODE with **η_θ gain** and **β_DA modulation**.
3. **Criticality regulation** — **ρ_crit⁽ˡ⁾ ≈ 1** maintained for all levels.
4. **Active inference coupling** — actions selected to minimize **E[F(a)]**.

### **Symmetric First-Order Approximation**

**Sₜ = Πᵉ·|εᵉ| + Πⁱ(M,c,a)·|εⁱ| > θₜ**

This is the limit of the canonical equation when:

- **α⁺ = α⁻, γ⁺ = γ⁻** *(symmetric valence)*
- **Γ⁽ˡ⁾ = 1** *(fully synchronized)*
- **w_res = 0** *(reservoir contribution ignored)*

All prior APGI derivations and predictions remain valid under this approximation.

---

---

## **27. Falsification Hierarchy**

Per APGI's **explicit methodological commitment** (Pillar 4, Empirical Credibility Roadmap), falsification criteria are organized into **three tiers**. **Pre-registering this hierarchy** before data collection is the **methodological standard** to which the APGI empirical program holds itself.

### **Tier 1 — Framework-Disconfirming (Core Claim Abandoned)**


| **Domain**                   | **Falsifying Result**                                                                                                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Neural correlates            | **P3b amplitude** shows no relationship to **Π ×                                                                                                                             |
| Allostatic regulation        | **θₜ dynamics** invariant to metabolic challenge (**glucose depletion, circadian phase, caloric restriction**) after controlling for arousal — **central mechanism failed**. |
| Computational benchmark      | APGI-LNN fails **BF > 100** over additive GNW model in pre-registered dataset (**N ≥ 500 trials × N ≥ 30 participants**).                                                    |
| Ignition threshold existence | Psychometric functions across all paradigms are best fit by **graded accumulation models** (no sigmoidal inflection with **β < 5**).                                         |


---

### **Tier 2 — Revision-Requiring (Component Modified, Core Intact)**


| **Component**               | **Revision-Requiring Result**                                                                                                                                          |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Allostatic mechanism        | **θₜ adapts** but is driven by **arousal/attention** rather than metabolic state specifically → revise **C_metabolic** term, retain ODE structure.                     |
| Serotonin mapping           | **5-HT manipulations** fail to shift **θₑᶠᶠ** in predicted direction.                                                                                                  |
| Signed error transform      | **α⁺/α⁻ asymmetry** not detectable in well-powered affective paradigm → revert to **                                                                                   |
| Phase-transition signatures | Only **1 of 3 co-occurrence signatures** (bistability, critical slowing, elevated susceptibility) confirmed → revise bifurcation claim to **graded-transition model**. |
| Reservoir integration       | **w_res contribution** not significantly different from zero → eliminate reservoir from ignition equation; retain as substrate model.                                  |


---

### **Tier 3 — Peripheral (Secondary Predictions)**

Null results on the following leave the **core framework intact**:

- Cross-species **PCI gradient** not significant at current sample sizes.
- **Circadian θₜ modulation** below detection threshold in specific populations.
- **Ultradian ~90-minute oscillations** in threshold not detectable in vigilance paradigms.
- **Cultural/contemplative PAC differences** not reaching pre-specified effect size.

---

---

## **28. Technical Glossary**


| **Term**                            | **Definition**                                                                                                                                                                                                                                     |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Allostatic regulation**           | Active, predictive adjustment of a setpoint variable (here: **θₜ**) in anticipation of future metabolic and informational demands — distinct from **homeostatic regulation**, which corrects deviations after the fact.                            |
| **Attentional blink**               | **~500 ms window** following a first conscious target during which a second target typically fails to reach conscious access; in APGI, modeled as the **refractory period** following **δ_refractory**.                                            |
| **Critical dynamics / criticality** | Operation near the boundary between ordered and chaotic regimes (**ρ_crit ≈ 1**), maximizing **dynamic range, sensitivity, and information integration**.                                                                                          |
| **Expected Free Energy (F(a))**     | Sum of **expected epistemic value** (uncertainty reduction) and **expected pragmatic value** (reward); the quantity minimized by action selection in **active inference** (Friston et al., 2015).                                                  |
| **Ignition**                        | The **discrete suprathreshold transition** in which **precision-weighted prediction error accumulation** triggers **global frontoparietal broadcast** — the event APGI equates with **conscious access**.                                          |
| **Metastability**                   | A dynamical regime in which **multiple quasi-stable states coexist** and the system transitions between them without settling permanently into any single attractor.                                                                               |
| **PAC (Phase-Amplitude Coupling)**  | Cross-frequency coupling in which the **phase of a low-frequency oscillation** (theta/alpha) modulates the **amplitude of a high-frequency oscillation** (gamma); proposed as the cortical implementation of **hierarchical precision weighting**. |
| **Precision weighting (Π)**         | In predictive processing: the **inverse variance** of a probability distribution used to modulate confidence in predictions or prediction errors; equivalent to **signal-to-noise gain control**.                                                  |
| **Reservoir computing**             | A computational framework in which a **fixed recurrent network** (the reservoir) maps inputs to a **high-dimensional state space**, with learning confined to a **linear readout layer**; **liquid-state machines** are the biological variant.    |
| **Signed prediction error φ(ε)**    | The **asymmetric nonlinear transformation** of prediction error that preserves **valence polarity** (approach vs. avoidance) — replaces unsigned **                                                                                                |
| **Spectral radius ρ_crit**          | The **largest absolute eigenvalue** of the recurrent weight matrix **W_rec**; governs whether activity **propagates (ρ > 1), decays (ρ < 1), or operates at criticality (ρ ≈ 1)**.                                                                 |
| **θₜ (ignition threshold)**         | The **dynamically regulated threshold** that **precision-weighted prediction error accumulation** must exceed for **global broadcast** to occur; **allosterically modulated** by metabolic state, circadian phase, and interoceptive afference.    |