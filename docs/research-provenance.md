# Research provenance

Source: *Machine Learning-Based Fertilizer Recommendation System Using Calibrated
Soil NPK Sensor Data for Bangladesh*, Md. Rezwanus Samam Toha and Tasbir Alam,
Department of Mechanical Engineering, RUET, August 2026. Supervisor:
Dr. Md. Rokunuzzaman.

The thesis was supplied as `Thesis_2002132_2002154.docx`. The full document is not
copied into the web server or offered for download.

## Mapping from the thesis to this application

| Application decision | Source |
| --- | --- |
| 333 records, 9 rice combinations, 37 levels | Section 3.6.1 and Table 3.6 |
| Urea, MoP, TSP outputs in g/decimal | Abstract, symbol list, Table 3.6 |
| Crop labels use BRRI; original encoded identifiers retained internally | Section 1.3.3 and supplied model schema |
| Do not claim learned independent nutrient responses | Sections 3.6.2, 4.3.1 and 5.3 |
| Restrict to known crops | Tables 4.3–4.4 and Section 4.3.3 |
| Calibration formulas and valid input ranges | Table 4.2; Equations 4.3–4.5 |
| Weak N, unsupported P, strongest K calibration | Sections 4.2.4 and 5.3 |
| No field/yield/emissions claims | Sections 1.3.3 and 5.3 |

## Calibration

| Channel | Corrected value | Valid sensor input range |
| --- | --- | --- |
| Nitrogen | 0.8255 × sensor N + 0.05026 | 0.001–0.013% |
| Phosphorus | 0.07761 × sensor P + 14.55 | 72–352 mg/kg |
| Potassium | 0.5840 × sensor K + 0.2508 | 0.164–0.885 meq/100 g |

These are the reported rounded coefficients, not new fits. Sixteen retained
observations are stated in the results chapter after excluding P1 and K5.
The phosphorus slope has p = 0.458 and is not quantitatively supported.
No dose or confidence interval is derived from that calibration result.

## Reported validation metrics

The following MAE values come from the Random Forest rows of Tables 4.3 and 4.4.
They were shown on the earlier portfolio landing page; the simplified calculator
keeps them in this documentation:

| Fertilizer | Random 5-fold MAE | Leave-one-crop-out MAE |
| --- | --- | --- |
| Urea | 19.92 | 210.05 |
| MoP | 12.53 | 125.25 |
| TSP | 8.20 | 41.20 |

All units are g/decimal. These are attributed to the thesis, not presented as a
fresh evaluation of the saved artifact. Section 4.3.3's prose calls 26.25 an MAE,
while Table 4.3 labels it RMSE; the interface uses the table's explicit MAE column.
Hold-out figures (17.19, 10.16, 8.14) refer to a different evaluation and are not
mixed into this comparison.

## Supplied artifact versus the written pipeline

Inspection of `fertilizer_rf_model.pkl` confirms a single multi-output
`RandomForestRegressor`, 300 estimators, three targets, eleven features, serialized
with scikit-learn 1.6.1. Feature order is obtained directly from the model.
The scaler names are `n/k` and `p`, with bounds [0, 0] and [0.36, 36].
The notebook target order is `urea`, `MoP`, `tsP`.

The thesis describes three independent regressors and a composed sensor pipeline.
The artifacts supplied here contain neither three separate nutrient regressors
nor a mapping from independent N/K sensor values into the shared source-table
feature. The application therefore preserves the working model contract and
exposes calibration separately. It does not silently average nitrogen and
potassium, overwrite one with the other, or claim that the full sensor-to-dose
chain has been validated.

No retraining or new performance claim is part of this interface update. A future
full pipeline requires a documented independent nutrient schema, a compatible
training set, saved preprocessing, and evaluation on representative inputs.
