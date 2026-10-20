"""A bilingual fertilizer calculator grounded in the supplied thesis."""
from __future__ import annotations

import logging
import json
import os
from functools import lru_cache
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest
from prediction import InputError, Predictor, calibrate

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor(BASE_DIR)


@app.get("/")
def index():
    return calculator()


@lru_cache(maxsize=1)
def translations():
    return json.loads((BASE_DIR / "static" / "translations.json").read_text(encoding="utf-8"))


@app.context_processor
def language_context():
    language = request.args.get("lang", request.cookies.get("calculator_language", "bn"))
    if language not in translations():
        language = "bn"
    return {"language": language, "copy": translations()[language], "translations": translations()}


@app.get("/predict")
def calculator():
    try:
        predictor = get_predictor()
        return render_template("predict.html", page="predict", crops=predictor.crops,
                               bounds=predictor.bounds, unavailable=False)
    except Exception:
        app.logger.exception("Could not load prediction artifacts")
        return render_template("predict.html", page="predict", crops=[], bounds={}, unavailable=True), 503


def read_payload():
    if not request.is_json:
        raise InputError("Send a JSON object containing the form values.")
    try:
        payload = request.get_json()
    except BadRequest:
        raise InputError("The submitted data could not be read.") from None
    if not isinstance(payload, dict):
        raise InputError("Send a JSON object containing the form values.")
    return payload


@app.post("/api/predict")
def predict():
    try:
        payload = read_payload()
        return jsonify(get_predictor().predict(payload))
    except InputError as exc:
        return jsonify(error=str(exc), field=exc.field, code=exc.code, params=exc.params), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="The prediction service is unavailable. Please try again later.", code="unavailable"), 503


@app.post("/api/calibrate")
def calibration():
    try:
        return jsonify(calibrate(read_payload()))
    except InputError as exc:
        return jsonify(error=str(exc), field=exc.field, code=exc.code, params=exc.params), 400


@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
        "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
    )
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
