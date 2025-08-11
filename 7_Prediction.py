import streamlit as st
import pandas as pd
import numpy as np

st.title("📈 Make Predictions")

# === Checks ===
required_keys = ["model", "scaler", "selected_features", "target_variable"]
if any(key not in st.session_state for key in required_keys):
    st.error("🚫 Missing model or preprocessing steps. Train a model first.")
    st.stop()

model = st.session_state["model"]
scaler = st.session_state["scaler"]
selected_features = list(st.session_state["selected_features"])
target_variable = st.session_state["target_variable"]
label_encoders = st.session_state.get("label_encoders", {})
encoding_strategy = st.session_state.get("encoding_strategy", None)

# Canonical feature order from training
feature_order = (
    list(getattr(scaler, "feature_names_in_", []))
    or selected_features
)
if set(feature_order) != set(selected_features):
    st.info("Using scaler's feature order to ensure consistency.")
    feature_order = [f for f in feature_order if f in selected_features]

st.caption(f"Expected features ({len(feature_order)}): {', '.join(feature_order)}")

# === Helpers ===
def ensure_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to match training order, add missing with 0."""
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 0
    return df[feature_order]

def encode_label_value(col: str, val):
    """Encode a single categorical value."""
    le = label_encoders.get(col)
    if le is None:
        return val
    classes = set(le.classes_.tolist())
    v = str(val)
    if v not in classes:
        if "Unknown" in classes:
            v = "Unknown"
        else:
            raise ValueError(f"Value '{val}' not seen for '{col}', and no 'Unknown' class was fitted.")
    return le.transform([v])[0]

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all values are numeric."""
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.isnull().any().any():
        bad = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Non-numeric or missing values found after encoding: {bad}")
    return df

# === Single Input Form ===
st.subheader("📝 Enter Feature Values")

inputs = {}
for f in feature_order:
    if f in label_encoders and encoding_strategy == "Label Encoding":
        classes = label_encoders[f].classes_.tolist()
        val = st.selectbox(f"{f} (categorical)", classes, key=f"inp_{f}")
        inputs[f] = encode_label_value(f, val)
    else:
        inputs[f] = st.number_input(f"{f} (numeric)", value=0.0, key=f"inp_{f}")

# === Predict Button ===
if st.button("🚀 Predict now", use_container_width=True):
    with st.spinner("Crunching the numbers..."):
        try:
            input_df = pd.DataFrame([inputs])
            input_df = ensure_feature_order(input_df)
            input_df = coerce_numeric(input_df)
            input_scaled = scaler.transform(input_df)
            pred = float(model.predict(input_scaled)[0])
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    st.balloons()
    # Fancy result card
    st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
    padding: 22px 24px; border-radius: 18px; color: white;
    box-shadow: 0 10px 28px rgba(0,0,0,0.18); margin-top: 8px;">
  <div style="font-size: 18px; opacity: 0.9;">🎯 Predicted {target_variable}</div>
  <div style="font-size: 42px; font-weight: 800; margin-top: 6px;">
    {pred:,.2f}
  </div>
  <div style="font-size: 15px; margin-top: 8px; opacity: 0.95;">
    Looks like you just went viral — your forecast is pulling serious weight. 📈🔥
  </div>
</div>
    """, unsafe_allow_html=True)
