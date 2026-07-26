# Fertilizer Prediction

A Streamlit web app that predicts Urea, MoP, and TSP amounts using the included
Random Forest model. Enter the crop, N/K test value, P/S test value, and land
area, and the app applies the saved scaler before sending data to the model.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. Because `.streamlit/config.toml` binds to
`0.0.0.0`, a phone on the same Wi-Fi can reach it at `http://<your-pc-ip>:8501`
— the app prints the exact address in the "How to access from your Mobile
Phone" section.

## Deploy to Streamlit Community Cloud

Free, and no server config needed:

1. Push this branch to GitHub.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **New app** → repository `rezwantoha/fertilizer-prediction`, branch `main`,
   main file `app.py`.
4. **Deploy**. The first build takes a few minutes while scikit-learn installs.

Every later push to `main` redeploys automatically.

### Notes

- `requirements.txt` pins `scikit-learn==1.6.1` exactly. The `.pkl` artifacts
  were produced by that version and unpickle unreliably under a different one —
  do not loosen this pin without retraining.
- The theme colors live in `.streamlit/config.toml`; the rest of the styling is
  injected from `static/style.css`.
- The free tier sleeps after ~7 days of inactivity and needs a click to wake.
- `templates/index.html` is left over from the earlier Flask version and is not
  used by the Streamlit app.
