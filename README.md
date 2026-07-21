# Fertilizer Prediction

A Flask web app that predicts Urea, MoP, and TSP amounts using the included
Random Forest model.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

The form accepts the crop, N/K test value, and P/S test value. The app applies
the saved scaler before sending data to the saved model.
