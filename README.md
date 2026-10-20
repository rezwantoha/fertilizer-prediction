# Fertilizer calculator / সারের হিসাব

A minimal bilingual calculator for rice fertilizer estimates in Bangladesh,
based on the supplied mechanical engineering thesis. Bangla is the default.
English is available through the language switch without clearing inputs or results.

## Run locally

Python 3.13 is used in the existing environment.

```powershell
# First-time setup only, if .venv does not already exist:
py -3.13 -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000 or http://127.0.0.1:5000/predict for the calculator.
Use `?lang=bn` or `?lang=en` for a specific language. Only the language preference
is stored in a local cookie; soil values and results are not stored.
No Node build or external font service is required. The app binds to localhost
with debug mode disabled.

The old Streamlit implementation is preserved as `streamlit_legacy.py` for reference.
It is not the maintained entry point and retains the old prototype's limitations.
The `.streamlit` configuration applies only to that archived prototype.

## Features

- One page with two steps: rice/land details, followed by soil-guide values.
- Bangla and English labels, crop names, instructions, results, errors, and CSV downloads.
- Accepts both Bangla and English numerals, including decimal values.
- Minimum 18 px on-screen text, larger input labels and result amounts, 48–58 px controls.
- Live prediction using the provided 300-tree Random Forest and fitted MinMaxScaler.
- Nine supported rice varieties with readable labels and original model encodings.
- Area in decimals, acres, or hectares; results in grams or kilograms.
- CSV export, printable results, sample inputs, and explicit result invalidation when inputs change.
- Server-side range, numeric, schema, and supported-crop validation.
- Research details and sensor calibration placed in expandable sections below the calculator.
- Keyboard navigation, labelled form fields, reduced-motion support, and no user-data storage.

## Scientific contract

The thesis confirms output units of **grams per decimal**. One decimal is one
hundredth of an acre (40.468564224 m²). Total mass = rate × area in decimals.

The saved artifact accepts `n/k`, `p`, and nine one-hot crop features, in its
recorded feature order. Its outputs are **Urea, MoP, TSP** in that order. It does
not accept independent N, P, and K channels. Inputs use the source-table scales
0–0.36 and 0–36; values outside those bounds are rejected.

All 333 training rows have P = 100 × N/K. Inputs off this line receive an
exploratory-output notice even when both individual values are within range.
Being on that line does not establish field validity.

Calibration uses thesis Equations 4.3–4.5 and Table 4.2, fitted to 16 observations
from one High Barind Tract soil. The phosphorus equation is shown as a research
calculation and explicitly not a reliable measurement. Corrected N/P/K values are
not automatically sent to the fertilizer model because no validated mapping into
its shared N/K feature is supplied.

See [research provenance](docs/research-provenance.md) for source sections,
reported thesis validation figures, and artifact discrepancies. The interface
does not claim these are newly computed scores for the provided pickle.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests load the real model, check a known prediction, all supported crops,
area conversion, training-range boundaries, invalid payloads, and calibration
equations. They also check Bangla numeric input, translation completeness,
language selection, and language-independent error codes. Keep scikit-learn
at **1.6.1**, matching artifact serialization.
Only load trusted local pickle/joblib artifacts.

## Structure

```text
app.py                    Flask routes and JSON API
prediction.py             Model loading, input validation, calibration equations
templates/                Minimal calculator page
static/                   CSS, JavaScript, bilingual text, and retained artwork
tests/test_app.py         Model and API contract checks
docs/                     Research provenance and artwork details
streamlit_legacy.py        Archived prototype
```

## Hosting

This is a Python server application, not a static-only website. For hosting, use
a Python WSGI service with the repository root as its working directory and
`app:app` as the application entry point. Include both `.pkl` files, `prediction.py`,
`templates`, and `static`. The model is loaded once per worker, so allow memory for
each worker. Use a production WSGI server and the platform's HTTPS proxy; do not
expose Flask's development server as a public deployment.

For example, after installing Waitress in the deployment environment:

```powershell
.\.venv\Scripts\python.exe -m pip install waitress
.\.venv\Scripts\waitress-serve.exe --host=127.0.0.1 --port=5000 app:app
```

The current implementation is local; it has not been publicly deployed.
