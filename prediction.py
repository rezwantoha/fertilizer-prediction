"""Model contract and validation, independent of the web interface.

The supplied RF artifact has two continuous features, not separate N/P/K
features. Calibration therefore remains a separate exploratory operation.
"""
from __future__ import annotations
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

AREA_FACTORS = {"decimal": 1.0, "acre": 100.0, "hectare": 10000 / 40.468564224}
TARGETS = (("urea", "Urea", "Nitrogen fertilizer"),
           ("mop", "MoP", "Muriate of potash"),
           ("tsp", "TSP", "Triple superphosphate"))
CROP_LABELS = {
    "Aus(BRI 27)": "Aus · BRRI 27", "Aus(BRI 42)": "Aus · BRRI 42",
    "B. Aman": "B. Aman", "Boro Rice(BRI 28)": "Boro · BRRI 28",
    "Boro Rice(BRI 29)": "Boro · BRRI 29", "Boro Rice(BRI 36)": "Boro · BRRI 36",
    "T. Aman(BRRI 25)": "T. Aman · BRRI 25", "T. Aman(BRRI 51)": "T. Aman · BRRI 51",
    "T. Aman(Binadhan 9)": "T. Aman · Binadhan 9",
}
CALIBRATION = {
    "n": {"label": "Nitrogen", "unit": "%", "slope": .8255, "intercept": .05026,
          "min": .001, "max": .013, "quality": "Weak calibration",
          "note": "A limited association in the study soil. Not a validated field measurement."},
    "p": {"label": "Phosphorus", "unit": "mg/kg", "slope": .07761, "intercept": 14.55,
          "min": 72., "max": 352., "quality": "Not quantitatively supported",
          "note": "The fitted slope was not significant (p = 0.458). This is an equation output, not a reliable phosphorus measurement."},
    "k": {"label": "Potassium", "unit": "meq/100 g", "slope": .5840, "intercept": .2508,
          "min": .164, "max": .885, "quality": "Strongest calibration in the study",
          "note": "Supported within the study's single-soil experiment; transfer to other soils has not been demonstrated."},
}


class InputError(ValueError):
    def __init__(self, message: str, field: str | None = None, code="invalid_payload", **params):
        super().__init__(message)
        self.field = field
        self.code = code
        self.params = params


def number(payload, key, label, minimum, maximum, *, exclusive_min=False):
    raw = payload.get(key)
    if raw is None or isinstance(raw, (bool, dict, list)):
        raise InputError(f"Enter a numeric value for {label}.", key, "invalid_number")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise InputError(f"Enter a numeric value for {label}.", key, "invalid_number") from None
    if not math.isfinite(value):
        raise InputError(f"{label} must be a finite number.", key, "invalid_number")
    if value < minimum or (exclusive_min and value == minimum) or value > maximum:
        low = f"greater than {minimum:g}" if exclusive_min else f"at least {minimum:g}"
        raise InputError(f"{label} must be {low} and no more than {maximum:g}.", key,
                         "invalid_range", min=minimum, max=maximum)
    return value


class Predictor:
    def __init__(self, directory: Path):
        # Only load trusted repository artifacts, never uploaded pickle files.
        self.model = joblib.load(directory / "fertilizer_rf_model.pkl")
        self.scaler = joblib.load(directory / "fertilizer_scaler.pkl")
        self.features = list(self.model.feature_names_in_)
        if list(self.scaler.feature_names_in_) != ["n/k", "p"]:
            raise RuntimeError("The scaler schema does not match this app.")
        expected = {"n/k", "p"} | {f"Crop_{crop}" for crop in CROP_LABELS}
        if set(self.features) != expected or self.model.n_outputs_ != 3:
            raise RuntimeError("The model schema does not match this app.")
        self.crops = [{"value": c, "label": label} for c, label in CROP_LABELS.items()]
        self.bounds = {
            key: {"min": float(self.scaler.data_min_[i]), "max": float(self.scaler.data_max_[i])}
            for i, key in enumerate(("nk", "p"))
        }
        if not np.allclose(self.scaler.data_min_, [0, 0]) or not np.allclose(self.scaler.data_max_, [.36, 36]):
            raise RuntimeError("The scaler range differs from the documented source table.")

    def predict(self, payload):
        crop = payload.get("crop")
        if not isinstance(crop, str) or crop not in CROP_LABELS:
            raise InputError("Choose one of the nine supported rice varieties.", "crop", "invalid_crop")
        nk = number(payload, "nk", "N/K guide value", **self._limits("nk"))
        p = number(payload, "p", "P guide value", **self._limits("p"))
        unit = payload.get("area_unit", "decimal")
        if not isinstance(unit, str) or unit not in AREA_FACTORS:
            raise InputError("Choose decimal, acre, or hectare for the area.", "area_unit", "invalid_unit")
        area = number(payload, "area", "land area", .000001, 1_000_000)
        decimals = area * AREA_FACTORS[unit]
        scaled = self.scaler.transform(pd.DataFrame([[nk, p]], columns=["n/k", "p"]))[0]
        row = {feature: 0.0 for feature in self.features}
        row.update({"n/k": scaled[0], "p": scaled[1], f"Crop_{crop}": 1.0})
        rates = self.model.predict(pd.DataFrame([row], columns=self.features))[0]
        if not np.all(np.isfinite(rates)) or np.any(rates < 0):
            raise RuntimeError("The model returned invalid fertilizer amounts.")
        on_diagonal = bool(np.isclose(scaled[0], scaled[1], atol=1e-8, rtol=0))
        warnings = []
        if not on_diagonal:
            warnings.append(
                "This input pair is outside the combinations represented by the training data. "
                "The source table only pairs N/K and P values where P = 100 × N/K. "
                "These outputs are exploratory and are not validated for independent nutrient values."
            )
        return {
            "crop": crop, "crop_label": CROP_LABELS[crop],
            "inputs": {"nk": nk, "p": p, "area": area, "area_unit": unit},
            "area_decimals": decimals, "on_training_diagonal": on_diagonal, "warnings": warnings,
            "fertilizers": [
                {"id": id_, "name": name, "description": description,
                 "rate_g_per_decimal": float(rate), "total_g": float(rate * decimals),
                 "total_kg": float(rate * decimals / 1000)}
                for (id_, name, description), rate in zip(TARGETS, rates)
            ],
            "basis": "Saved 300-tree Random Forest; source-table estimates, not field-validated doses.",
        }

    def _limits(self, key):
        return {"minimum": self.bounds[key]["min"], "maximum": self.bounds[key]["max"]}


def calibrate(payload):
    readings = []
    for key, spec in CALIBRATION.items():
        value = number(payload, key, spec["label"] + " sensor reading", spec["min"], spec["max"])
        readings.append({"id": key, **spec, "raw": value, "corrected": spec["slope"] * value + spec["intercept"]})
    return {
        "readings": readings,
        "basis": "Thesis Table 4.2 and Equations 4.3–4.5; 16 retained observations from one High Barind Tract soil.",
        "limitation": (
            "These three corrected values are not passed to the fertilizer model: the saved "
            "artifact has a shared N/K feature, and no validated mapping from separate N/P/K "
            "measurements to that feature is supplied."
        ),
    }
