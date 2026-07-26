# Calibration of a 7-in-1 RS485 Soil Probe

**Dataset:** `Fertilization_Comparison.xlsx` — 18 spiked samples on a single base soil
(lab baseline: pH 7.8, N 0.05 %, P 3.8 mg/kg, K 0.37 meq/100 g).
**Fitted on:** 16 samples (K5 and P1 excluded, see §2).
**Scripts:** `sensor_calibration_analysis.py` (analysis), `soil_probe_calibration.py` (deployment).

---

## 1. The finding that governs everything else

The three "NPK" outputs are not three measurements. Regressing the K channel on
the P channel over all 18 samples gives

$$K_{\text{raw}} = 2.57856\times10^{-3}\,P_{\text{raw}} - 2.06425\times10^{-2}$$

with **R² = 0.999969**, residual σ = 0.00122 raw units, max |residual| = 0.00282.
That is a firmware affine transform, not an independent electrode.

Principal component analysis on the log-transformed channel matrix:

| PC | variance explained |
|----|--------------------|
| 1  | 94.96 % |
| 2  | 5.04 % |
| 3  | 0.01 % |

Variance inflation factors: N = 5, P = 2 737, K = 2 744.

The 3 × 3 sensitivity matrix **K** (rows = sensor channels, columns = applied nutrient,
each entry the OLS slope from the corresponding single-nutrient series):

|              | applied N | applied P | applied K |
|--------------|-----------|-----------|-----------|
| sensor N (%) | 2.0e-05   | 2.0e-05   | 6.53e-04  |
| sensor P     | 0.580     | 0.213     | 1.246     |
| sensor K     | 1.484e-03 | 5.933e-04 | 3.197e-03 |

- det(**K**) = 1.68 × 10⁻⁸
- singular values = [1.391, 2.68 × 10⁻⁴, 4.50 × 10⁻⁵]
- cond(**K**) = 30 913

**Effective rank 1.** Recovering (N, P, K) from this probe is an under-determined
inverse problem with a two-dimensional null space. This is a property of the
hardware, not of the model — no regression, kernel method, or neural network
can invert a rank-deficient forward map.

Supporting evidence from the sensitivity matrix itself: the off-diagonal R²
values equal or exceed the diagonal ones. Applying urea explains the *phosphorus*
channel (R² = 0.95) better than the *nitrogen* channel (R² = 0.78); applying MoP
moves the P channel (R² = 0.93) as strongly as the K channel (R² = 0.93).

**Quantisation.** The N channel LSB is 0.001 % = 10 mg/kg. Across the urea series
(25 → 125 mg/kg applied N) it returns three distinct values: 0.003, 0.003, 0.003,
0.004, 0.005 — a span of two LSB over a five-fold dose range.

---

## 2. Outlier exclusion

| sample | channel | value | reason |
|--------|---------|-------|--------|
| K5 | N | 0.100 % | 14× jump from K4 (0.007 %) caused by adding *potassium*; saturation or fault |
| P1 | pH | 3.4 | physically impossible in a soil with lab pH 7.8; probe-contact fault |

Both excluded from all calibration fits. Rule: studentised residual |t| > 2.0 within
the series, plus physical-plausibility screening.

---

## 3. Forward model (sensor as a function of known loading)

Fitting the P channel against the full DOE design matrix:

$$P_{\text{raw}} = 74.03 + 0.7254\,N + 0.1845\,P + 1.2346\,K$$

(N, P, K in mg/kg added; R² = 0.9706, adjusted R² = 0.9632, RMSE = 14.80 raw counts)

| coefficient | value | 95 % CI | t | p |
|-------------|-------|---------|---|---|
| b₀ | +74.03 | ±12.77 | 12.63 | < 0.001 |
| b_N | +0.7254 | ±0.1772 | 8.92 | < 0.001 |
| b_P | +0.1845 | ±0.2912 | 1.38 | **0.19** |
| b_K | +1.2346 | ±0.1687 | 15.94 | < 0.001 |

**The phosphorus coefficient is not statistically significant.** Within the tested
range (15–75 mg/kg P as TSP) phosphorus is invisible to this probe. That is
mechanistically expected: the probe transduces bulk ionic conductivity, and
H₂PO₄⁻ has low limiting molar conductivity (≈ 36 S cm² mol⁻¹) and adsorbs strongly
onto an alkaline soil matrix.

Sensitivity expressed per mmol of element (slope × molar mass):

| element | counts per mmol/kg | dominant ion | λ° (S cm² mol⁻¹) |
|---------|--------------------|--------------|------------------|
| K | 48.3 | K⁺ + Cl⁻ | 73.5 + 76.3 |
| N | 10.2 | NH₄⁺ after urea hydrolysis | 73.5 (urea itself: 0) |
| P | 5.7 | H₂PO₄⁻ | ≈ 36 |

The ordering matches ionic mobility, confirming the transduction mechanism is
bulk EC rather than nutrient-specific ion-selective sensing.

---

## 4. The deployable calibration

Because the forward map is rank 1, the only invertible quantity is a **composite
ionic loading index (CIL)**, expressed on a K-equivalent basis by normalising the
forward coefficients by b_K:

$$\mathrm{CIL} = 0.5876\,N + 0.1494\,P + 1.0000\,K \quad [\text{mg/kg K-equivalent}]$$

The inverse — **the equation to put in your firmware or Streamlit app**:

$$\boxed{\widehat{\mathrm{CIL}} = 0.78615\,P_{\text{raw}} - 56.404}$$

| metric | value |
|--------|-------|
| R² | 0.9706 |
| RMSE | 10.94 mg/kg K-eq |
| slope SE | 0.03658 |
| intercept SE | 6.104 |
| LOD (3σ) | 32.8 mg/kg K-eq |
| LOQ (10σ) | 109.4 mg/kg K-eq |
| 95 % prediction interval | ± 23.5 mg/kg K-eq |
| validity range | P_raw ∈ [71, 352], RH ∈ [49, 63] %, T ≈ 28 °C |

Do not use the K channel as a second input — substitute the firmware relation from
§1 and use it only as an integrity check (flag if |K_raw − (0.00257856·P_raw − 0.020643)| > 0.00365).

---

## 5. Validation (leave-one-out cross-validation)

| target | RMSE_cv | RMSE (predict the mean) | RPD | NSE | verdict |
|--------|---------|--------------------------|-----|-----|---------|
| Applied N (mg/kg) | 44.79 | 45.93 | 1.06 | 0.05 | marginal — barely beats the mean |
| Applied P (mg/kg) | 43.08 | 28.05 | **0.67** | **−1.36** | **fails — worse than predicting the mean** |
| Applied K (mg/kg) | 27.71 | 48.86 | 1.82 | 0.68 | usable *within this DOE only* (see caveat) |
| **Composite CIL** | **14.25** | **59.65** | **4.32** | **0.94** | **usable** |

RPD = SD / RMSE_cv; RPD ≥ 1.4 is the conventional threshold for screening-grade
soil sensing, RPD < 1.0 means the model loses to a constant.

**Caveat on the K result.** The apparent success of applied-K prediction is an
artefact of the experimental design: K was spiked as KCl, the most conductive
amendment used, so it dominates the EC signal and the model effectively learns
EC → K. In a field soil where N, P, K and background salts vary independently,
that mapping collapses. Do not present it as a validated potassium calibration.

**Ordinal classification.** Binning CIL into tertiles and evaluating on
leave-one-out predictions:

- class edges: 29.4 and 60.0 mg/kg K-equivalent
- accuracy: **93.8 %**
- Cohen's κ: **0.906**

This is the result that survives validation and the one that should feed the
FRG-2018 soil-test class lookup in Stage B.

---

## 6. pH channel

Single-point offset from the baseline soil: pH_cal = pH_raw + 1.70.

This is **not recommended for use**. The scatter across 17 nominally identical
samples is σ = 0.45 pH units (range 6.9–8.6), far exceeding any real variation in
a single homogenised soil, and applying the +1.70 offset pushes several readings
above pH 9.5, which is not physical for this soil. A slope correction cannot be
fitted because only one lab pH value exists.

**Required:** a 2-point (pH 4.01 / 7.00) or 3-point buffer calibration, then
replace the offset with a proper slope-and-intercept pair. The offset is disabled
by default in `soil_probe_calibration.py` (`apply_ph_offset=False`).

---

## 7. Figures

| file | content |
|------|---------|
| `fig1_channel_diagnostics.png` | correlation heatmap, PCA scree, K–P collinearity, K residuals, N quantisation staircase, moisture confound |
| `fig2_sensitivity_matrix.png` | 3 × 3 response grid, diagonal boxed |
| `fig3_composite_calibration.png` | calibration curve with confidence and prediction bands, Bland–Altman, RPD comparison |
| `fig4_pernutrient_vs_ordinal.png` | LOO predicted-vs-reference for N, P, K against the mean baseline; LOO-validated ordinal confusion matrix |
| `fig5_ph_channel.png` | pH scatter and distribution against the lab reference |

---

## 8. Known limitations to state in the thesis

1. **n = 1 per sample.** No replicate readings exist, so instrumental repeatability
   σ was never measured; all uncertainty figures above are regression residuals,
   which conflate sensor noise with spike-recovery error. Take 5–10 repeat
   insertions on three samples before submission.
2. **Reference values are assumed, not measured.** The "Calculated Data" column is
   baseline lab values plus spike arithmetic, i.e. 100 % recovery is assumed.
   Phosphate fixation in an alkaline soil almost certainly makes the true available
   P lower than the nominal value, which would partly explain the null P result.
3. **One soil.** All conclusions are conditional on this soil's texture, organic
   matter and background EC. The calibration is not transferable without
   re-derivation.
4. **No independent EC measurement.** The rank-1 argument is inferred from channel
   collinearity. Measuring EC directly on the same samples would prove it and turn
   a limitation into a characterisation result.
5. **Narrow moisture and temperature bands.** Moisture varied only 49–63 % and
   temperature 27.8–28.8 °C, so the moisture and temperature cross-terms are not
   quantified. EC in soil rises roughly 2 % per K.

---

## 9. What to write in Chapter 4

The defensible claim is:

> The 7-in-1 probe is not a nutrient-selective sensor. Spike-recovery experiments
> across independent N, P and K gradients show a sensitivity matrix of effective
> rank 1 (det = 1.7 × 10⁻⁸, cond = 3.1 × 10⁴), with the potassium output an exact
> affine transform of the phosphorus output (R² = 0.99997). The device is
> therefore characterised here as a single-channel bulk ionic-loading transducer.
> Calibrated on that basis it recovers a composite K-equivalent loading index with
> R²_cv = 0.94 and RPD = 4.3, and assigns a three-level ordinal soil-test class
> with 93.8 % leave-one-out accuracy (κ = 0.91). Per-nutrient continuous
> calibration was attempted and rejected: phosphorus prediction is worse than the
> sample mean (RPD = 0.67, NSE = −1.36) and nitrogen prediction is at the mean
> level (RPD = 1.06).

That is a stronger contribution than a weak per-nutrient calibration would have
been, because it characterises the hardware honestly and still delivers a usable
input to the FRG-2018 recommendation stage.
