"""Streamlit web application for soil-guided fertilizer recommendations.

Deployable locally or on Streamlit Cloud, accessible from mobile phones.
"""

from __future__ import annotations

import socket
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fertilizer_rf_model.pkl"
SCALER_PATH = BASE_DIR / "fertilizer_scaler.pkl"
DATA_PATH = BASE_DIR / "df_fertilizers_normalized.csv"
CSS_PATH = BASE_DIR / "static" / "style.css"


# Configure page settings
st.set_page_config(
    page_title="Fertilizer Predictor",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_ml_artifacts():
    """Load Random Forest model and scaler once and cache in memory."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_crop_list():
    """Load available crops from dataset schema."""
    cols = pd.read_csv(DATA_PATH, nrows=1).columns
    crops = sorted([col[5:] for col in cols if col.startswith("Crop_")])
    return crops


def get_local_ip() -> str:
    """Retrieve local IP address for Wi-Fi mobile access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "YOUR_PC_IP"


def inject_custom_css():
    """Inject custom stylesheet for responsive mobile design."""
    if CSS_PATH.exists():
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def format_amount(value: float) -> str:
    """Format recommendation values cleanly."""
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def main():
    inject_custom_css()

    # Load machine learning model & data
    model, scaler = load_ml_artifacts()
    crops = load_crop_list()
    local_ip = get_local_ip()

    # Header / Hero section
    st.markdown(
        """
        <div class="intro">
            <p class="eyebrow">Soil-guided recommendation</p>
            <h1 style="margin: 0; padding: 0;">Fertilizer Predictor 🌱</h1>
            <p style="margin-top: 8px;">Enter crop type and soil test parameters to compute precise fertilizer application rates.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Input Card
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        with st.form(key="prediction_form"):
            selected_crop = st.selectbox(
                "Select Crop",
                options=crops,
                index=0,
                help="Choose the crop you plan to cultivate.",
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                nk_value = st.number_input(
                    "N/K Test Value",
                    min_value=0.0,
                    value=50.0,
                    step=1.0,
                    help="Soil test value for Nitrogen/Potassium.",
                )
            with col2:
                p_value = st.number_input(
                    "P/S Test Value",
                    min_value=0.0,
                    value=25.0,
                    step=1.0,
                    help="Soil test value for Phosphorus/Sulfur.",
                )
            with col3:
                land_area = st.number_input(
                    "Land Area (decimal)",
                    min_value=0.01,
                    value=1.0,
                    step=0.5,
                    help="Land area measured in decimal units.",
                )

            submit_button = st.form_submit_button(
                label="Calculate Recommendation",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # Process Form Submission
    if submit_button:
        try:
            if nk_value < 0 or p_value < 0:
                st.error("Soil-test values cannot be negative.")
                return
            if land_area <= 0:
                st.error("Land area must be greater than 0.")
                return

            # Scale inputs exactly as trained
            scaled_values = scaler.transform(
                pd.DataFrame([[nk_value, p_value]], columns=["n/k", "p"])
            )[0]

            model_input = {"n/k": scaled_values[0], "p": scaled_values[1]}
            model_input.update({f"Crop_{c}": int(c == selected_crop) for c in crops})

            features = pd.DataFrame([model_input])
            if hasattr(model, "feature_names_in_"):
                features = features.reindex(columns=model.feature_names_in_, fill_value=0)

            # Predict fertilizer rates (gm/decimal)
            urea_rate, mop_rate, tsp_rate = model.predict(features)[0]

            urea_total = max(0.0, urea_rate * land_area)
            mop_total = max(0.0, mop_rate * land_area)
            tsp_total = max(0.0, tsp_rate * land_area)

            # Render Results
            st.markdown(
                f"""
                <section class="results">
                    <p class="eyebrow">Predicted fertilizer requirement ({format_amount(land_area)} decimal area)</p>
                    <div class="result-grid">
                        <article>
                            <h2>Urea</h2>
                            <p>{format_amount(urea_total)} gm</p>
                            <span>Rate: {format_amount(urea_rate)} gm/decimal</span>
                        </article>
                        <article>
                            <h2>MoP</h2>
                            <p>{format_amount(mop_total)} gm</p>
                            <span>Rate: {format_amount(mop_rate)} gm/decimal</span>
                        </article>
                        <article>
                            <h2>TSP</h2>
                            <p>{format_amount(tsp_total)} gm</p>
                            <span>Rate: {format_amount(tsp_rate)} gm/decimal</span>
                        </article>
                    </div>
                </section>
                """,
                unsafe_allow_html=True,
            )

        except Exception as exc:
            st.error(f"The recommendation could not be calculated. Error: {str(exc)}")

    # Mobile Access Instructions Section
    st.markdown("---")
    with st.expander("📱 How to access from your Mobile Phone"):
        st.markdown(
            f"""
            ### 1️⃣ Same Wi-Fi Network Access (Local Mobile Access)
            Make sure your PC and mobile phone are connected to the **same Wi-Fi network**.
            Open the browser on your phone and type:
            
            ```text
            http://{local_ip}:8501
            ```

            ---

            ### 2️⃣ Streamlit Community Cloud (Internet Access Anywhere)
            To access this app from your phone anywhere over mobile data/internet:
            1. Push this project repository to **GitHub**.
            2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
            3. Click **"New App"**, select repository `fertilizer-prediction`, branch `main`, and main file `app.py`.
            4. Click **"Deploy"**! You will get a custom URL to open on any mobile phone.
            """
        )


if __name__ == "__main__":
    main()
