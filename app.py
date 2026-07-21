"""Flask web app for fertilizer recommendations."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fertilizer_rf_model.pkl"
SCALER_PATH = BASE_DIR / "fertilizer_scaler.pkl"
DATA_PATH = BASE_DIR / "fertilizers_nonzero.csv"

app = Flask(__name__)


def load_artifacts():
    """Load the model, scaler, and crop names once when the server starts."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    crops = sorted(pd.read_csv(DATA_PATH, usecols=["Crop"])["Crop"].dropna().unique())
    return model, scaler, crops


MODEL, SCALER, CROPS = load_artifacts()


def format_amount(value: float) -> str:
    """Keep recommended amounts readable while retaining one decimal if needed."""
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form = {"crop": "", "nk": "", "p": ""}

    if request.method == "POST":
        form = {
            "crop": request.form.get("crop", "").strip(),
            "nk": request.form.get("nk", "").strip(),
            "p": request.form.get("p", "").strip(),
        }

        try:
            if form["crop"] not in CROPS:
                raise ValueError("Please choose a crop from the list.")

            nk_value = float(form["nk"])
            p_value = float(form["p"])
            if nk_value < 0 or p_value < 0:
                raise ValueError("Soil-test values cannot be negative.")

            # Scale the two continuous fields exactly as in model training.
            scaled_values = SCALER.transform(pd.DataFrame([[nk_value, p_value]], columns=["n/k", "p"]))[0]
            model_input = {"n/k": scaled_values[0], "p": scaled_values[1]}
            model_input.update({f"Crop_{crop}": int(crop == form["crop"]) for crop in CROPS})

            features = pd.DataFrame([model_input])
            if hasattr(MODEL, "feature_names_in_"):
                features = features.reindex(columns=MODEL.feature_names_in_, fill_value=0)

            urea, mop, tsp = MODEL.predict(features)[0]
            prediction = {
                "Urea": format_amount(urea),
                "MoP": format_amount(mop),
                "TSP": format_amount(tsp),
            }
        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Prediction failed")
            error = "The recommendation could not be calculated. Please check the inputs and try again."

    return render_template("index.html", crops=CROPS, form=form, prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)
