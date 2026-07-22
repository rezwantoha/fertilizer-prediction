"""Flask web app for fertilizer recommendations."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fertilizer_rf_model.pkl"
SCALER_PATH = BASE_DIR / "fertilizer_scaler.pkl"
DATA_PATH = BASE_DIR / "df_fertilizers_normalized.csv"

app = Flask(__name__)


def load_artifacts():
    """Load the model, scaler, and crop names once when the server starts."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    cols = pd.read_csv(DATA_PATH, nrows=1).columns
    crops = sorted([col[5:] for col in cols if col.startswith("Crop_")])
    return model, scaler, crops


MODEL, SCALER, CROPS = load_artifacts()


def format_amount(value: float) -> str:
    """Keep recommended amounts readable while retaining one decimal if needed."""
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form = {"crop": "", "nk": "", "p": "", "land_area": "1"}

    if request.method == "POST":
        form = {
            "crop": request.form.get("crop", "").strip(),
            "nk": request.form.get("nk", "").strip(),
            "p": request.form.get("p", "").strip(),
            "land_area": request.form.get("land_area", "1").strip(),
        }

        try:
            if not form["crop"] or form["crop"] not in CROPS:
                raise ValueError("Please choose a crop from the list.")
            if not form["nk"]:
                raise ValueError("Please enter N/K test value.")
            if not form["p"]:
                raise ValueError("Please enter P/S test value.")
            if not form["land_area"]:
                raise ValueError("Please enter land area.")

            nk_value = float(form["nk"])
            p_value = float(form["p"])
            land_area = float(form["land_area"])

            if nk_value < 0 or p_value < 0:
                raise ValueError("Soil-test values cannot be negative.")
            if land_area <= 0:
                raise ValueError("Land area must be greater than 0.")

            # Scale the two continuous fields exactly as in model training.
            scaled_values = SCALER.transform(pd.DataFrame([[nk_value, p_value]], columns=["n/k", "p"]))[0]
            model_input = {"n/k": scaled_values[0], "p": scaled_values[1]}
            model_input.update({f"Crop_{crop}": int(crop == form["crop"]) for crop in CROPS})

            features = pd.DataFrame([model_input])
            if hasattr(MODEL, "feature_names_in_"):
                features = features.reindex(columns=MODEL.feature_names_in_, fill_value=0)

            urea_rate, mop_rate, tsp_rate = MODEL.predict(features)[0]

            urea_total = max(0.0, urea_rate * land_area)
            mop_total = max(0.0, mop_rate * land_area)
            tsp_total = max(0.0, tsp_rate * land_area)

            prediction = {
                "unit_param": "gm/decimal",
                "land_area": format_amount(land_area),
                "fertilizers": {
                    "Urea": {
                        "total": format_amount(urea_total),
                        "rate": format_amount(urea_rate),
                    },
                    "MoP": {
                        "total": format_amount(mop_total),
                        "rate": format_amount(mop_rate),
                    },
                    "TSP": {
                        "total": format_amount(tsp_total),
                        "rate": format_amount(tsp_rate),
                    },
                },
            }
        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Prediction failed")
            error = "The recommendation could not be calculated. Please check the inputs and try again."

    return render_template("index.html", crops=CROPS, form=form, prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)
