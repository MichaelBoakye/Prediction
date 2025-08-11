import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.inspection import permutation_importance

st.title("🧐 Model Interpretation")

# ---------- Prerequisites ----------
needed = ["model", "X_scaled", "encoded_data", "selected_features", "scaler", "target_variable"]
if any(k not in st.session_state for k in needed):
    st.error("Required objects not found. Train a model and run evaluation first.")
    st.stop()

model = st.session_state["model"]
X_scaled: pd.DataFrame = st.session_state["X_scaled"]
df_encoded: pd.DataFrame = st.session_state["encoded_data"]
selected_features: list = st.session_state["selected_features"]
scaler = st.session_state["scaler"]
target = st.session_state["target_variable"]

# Canonical order used to fit the scaler/model
feature_order = list(getattr(scaler, "feature_names_in_", [])) or selected_features
# keep only the features we actually used
feature_order = [f for f in feature_order if f in selected_features]
X_used = X_scaled[feature_order].copy()

# Helper: get original (unscaled) feature matrix aligned to selected_features
X_encoded_full = df_encoded[selected_features].copy()

# ---------- 1) Coefficients: directional impact (Lasso/linear only) ----------
st.header("📉 Directional Impact (Model Coefficients)")
if hasattr(model, "coef_"):
    coef_df = pd.DataFrame(
        {"Feature": feature_order, "Coefficient": model.coef_}
    )
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs", ascending=False)

    top_k = st.slider("Show top N features by |coefficient|", 5, min(20, len(coef_df)), 10)
    top_coef = coef_df.head(top_k).copy()

    fig_coef = px.bar(
        top_coef.sort_values("Coefficient"),
        x="Coefficient", y="Feature", orientation="h",
        color="Coefficient", color_continuous_scale="RdBu",
        template="plotly_white", title="Top directional effects"
    )
    fig_coef.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("""
**How to read this:**  
Positive coefficients push the predicted loan amount up as the feature increases.  
Negative coefficients pull it down. Bars are ranked by absolute size so you can see the heaviest hitters first.
    """)
else:
    st.info("This model does not expose linear coefficients. Skip to permutation importance for a model-agnostic view.")

# ---------- 2) Permutation Importance: what the model actually uses ----------
st.header("🧪 Permutation Importance (Model-agnostic)")

with st.spinner("Computing permutation importance…"):
    # Small sample for speed if needed
    if len(X_used) > 10000:
        sample_idx = np.random.RandomState(42).choice(len(X_used), size=10000, replace=False)
        X_pi = X_used.iloc[sample_idx]
        y_pi = df_encoded[target].iloc[sample_idx]
    else:
        X_pi = X_used
        y_pi = df_encoded[target]

    # Score by default is model.score (R² for regressors). We’ll report mean decrease.
    pi = permutation_importance(model, X_pi, y_pi, n_repeats=10, random_state=42, n_jobs=-1)
    pi_df = pd.DataFrame({
        "Feature": X_pi.columns,
        "MeanDecreaseScore": pi.importances_mean,
        "Std": pi.importances_std
    }).sort_values("MeanDecreaseScore", ascending=False)

k_pi = st.slider("Show top N by permutation importance", 5, min(20, len(pi_df)), 10, key="pi_k")
top_pi = pi_df.head(k_pi)

fig_pi = px.bar(
    top_pi.sort_values("MeanDecreaseScore"),
    x="MeanDecreaseScore", y="Feature", orientation="h",
    error_x="Std",
    template="plotly_white", title="Top features by permutation importance"
)
fig_pi.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig_pi, use_container_width=True)

st.markdown("""
**Why this matters:**  
Permutation importance shuffles one feature at a time and checks how much the model’s score drops.  
Bigger drops mean the model relied on that feature more. This is a good sanity check that often differs from raw coefficients.
""")

# ---------- 3) Partial Dependence: average effect curves ----------
st.header("📈 Partial Dependence (Average effect)")

# choose top 3 numeric features from permutation importance
numeric_mask = (X_encoded_full.dtypes.apply(lambda t: np.issubdtype(t, np.number)))
numeric_feats = [f for f in top_pi["Feature"].tolist() if f in X_encoded_full.columns and numeric_mask[f]]

if not numeric_feats:
    st.info("No numeric features available for partial dependence.")
else:
    pdp_feats = numeric_feats[:3]
    st.caption(f"Showing partial dependence for: {', '.join(pdp_feats)}")

    def pdp_curve(feature: str, points: int = 25):
        # grid from 5th to 95th percentile in original units
        series = X_encoded_full[feature].dropna()
        if series.nunique() < 2:
            return None
        q = np.linspace(0.05, 0.95, points)
        grid = np.quantile(series, q)
        # baseline row = medians of original encoded features
        base_row = X_encoded_full.median(numeric_only=True)
        curves = []
        for v in grid:
            row = base_row.copy()
            row[feature] = v
            row_df = pd.DataFrame([row])[feature_order]
            # scale then predict
            row_scaled = scaler.transform(row_df)
            y_hat = float(model.predict(row_scaled)[0])
            curves.append((v, y_hat))
        return pd.DataFrame(curves, columns=[feature, "Predicted"])

    cols = st.columns(len(pdp_feats))
    for ax, ftr in zip(cols, pdp_feats):
        df_pdp = pdp_curve(ftr)
        if df_pdp is None:
            ax.info(f"Not enough variation in {ftr} to compute curve.")
            continue
        fig_pdp = px.line(
            df_pdp, x=ftr, y="Predicted",
            title=f"Partial Dependence: {ftr}",
            template="plotly_white"
        )
        fig_pdp.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        ax.plotly_chart(fig_pdp, use_container_width=True)

    st.markdown("""
**Reading the curves:**  
Each chart shows the model’s average prediction as we vary one feature, holding others near typical values.  
Flat lines mean little effect. Slopes or bends suggest stronger influence or non-linear behavior.
""")

# ---------- 4) What-if simulator ----------
st.header("🧮 What-if Simulator")

st.caption("Tweak the top features to see how the prediction moves. Sliders use the 5th–95th percentile ranges.")

# build slider ranges from original encoded data
def slider_range(s: pd.Series):
    s = s.dropna()
    low, high = np.quantile(s, [0.05, 0.95]) if len(s) else (0.0, 1.0)
    if low == high:
        high = low + 1.0
    step = (high - low) / 100 if high > low else 1.0
    return float(low), float(high), float(step)

sim_feats = pdp_feats if pdp_feats else feature_order[:3]
defaults = X_encoded_full[sim_feats].median(numeric_only=True).to_dict()

sim_values = {}
for f in sim_feats:
    lo, hi, step = slider_range(X_encoded_full[f]) if f in X_encoded_full.columns else (0.0, 1.0, 0.01)
    sim_values[f] = st.slider(f"{f}", min_value=lo, max_value=hi, value=float(defaults.get(f, lo)), step=step)

if st.button("🚀 Simulate prediction"):
    with st.spinner("Calculating…"):
        base = X_encoded_full.median(numeric_only=True)
        for f, v in sim_values.items():
            base[f] = v
        row = pd.DataFrame([base])[feature_order]
        row_scaled = scaler.transform(row)
        y_hat = float(model.predict(row_scaled)[0])

    st.balloons()
    st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #6366F1 0%, #06B6D4 100%);
    padding: 20px 22px; border-radius: 16px; color: white;
    box-shadow: 0 10px 28px rgba(0,0,0,0.18); margin-top: 8px;">
  <div style="font-size: 16px; opacity: 0.9;">Simulated prediction for your settings</div>
  <div style="font-size: 38px; font-weight: 800; margin-top: 6px;">
    {y_hat:,.2f}
  </div>
  <div style="font-size: 14px; margin-top: 8px; opacity: 0.95;">
    Nudge the sliders to see how the forecast shifts. Helpful for explaining decisions to non-technical audiences.
  </div>
</div>
    """, unsafe_allow_html=True)

# ---------- 5) Takeaways ----------
st.header("📝 Quick Takeaways")
st.markdown("""
- **Direction vs dependence:** Coefficients show the signed effect for linear models.  
  Permutation importance shows how much the model relied on each feature in practice.
- **Non-linearities:** Partial dependence can reveal curved or threshold effects that coefficients may miss.
- **Actionable levers:** Use the what-if simulator to explain how changes in key drivers might influence the predicted loan amount.
""")
