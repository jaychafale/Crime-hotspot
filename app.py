import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------
# LOAD MODEL + METADATA
# -------------------------
MODEL_PATH = Path("models/best_model.joblib")
META_PATH  = Path("models/feature_meta.json")

model = joblib.load(MODEL_PATH)

with open(META_PATH, "r") as f:
    meta = json.load(f)

target_col = meta["target"]
numeric_features = meta["numeric_features"]

# -------------------------
# USER-FRIENDLY LABELS
# -------------------------
feature_label_map = {
    "population": "Population",
    "racepctblack": "% Black Population",
    "medIncome": "Median Income",
    "racePctHisp": "% Hispanic Population",
    "racePctWhite": "% White Population",
    "PctPopUnderPov": "% Under Poverty Line",
}

# Only ask for these 6 features
ASK_FEATURES = list(feature_label_map.keys())

# Allowed defaults (optional)
default_values = {
    "population": 50000,
    "racepctblack": 10,
    "medIncome": 35000,
    "racePctHisp": 15,
    "racePctWhite": 60,
    "PctPopUnderPov": 12,
}

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Crime Prediction", layout="centered")

st.title("🔮 Crime Rate Prediction (Violent Crimes per Capita)")
st.write("Enter community details below to estimate violent crime rate.")

# -------------------------
# INPUT FIELDS
# -------------------------
user_inputs = {}

st.header("Community Inputs")

for feat in ASK_FEATURES:
    label = feature_label_map[feat]
    default_val = default_values.get(feat, 0)

    # numeric input
    val = st.number_input(
        label,
        min_value=0.0,
        max_value=1e12,
        value=float(default_val),
        step=1.0
    )

    user_inputs[feat] = val

# -------------------------
# CREATE INPUT DF
# -------------------------
# Fill the required columns of the model input
input_row = {col: 0.0 for col in numeric_features}

# Override the selected 6
for k, v in user_inputs.items():
    input_row[k] = v

X_input = pd.DataFrame([input_row])

# -------------------------
# PREDICT BUTTON
# -------------------------
st.markdown("---")

if st.button("Predict Crime Level"):
    try:
        pred = model.predict(X_input)[0]

        st.success(f"✅ **Predicted Violent Crime Rate:** {pred:.4f}")
        st.info("This value represents violent crimes per capita (scaled).")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

st.markdown("---")
st.caption("Model trained on UCI Communities & Crime dataset.")
