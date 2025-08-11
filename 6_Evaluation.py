import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
import plotly.express as px

st.title("📉 Model Evaluation - K-Fold Cross-Validation")

# --- Check prerequisites ---
required_keys = ["model", "X_scaled", "encoded_data", "target_variable", "selected_features"]
if any(key not in st.session_state for key in required_keys):
    st.error("🚫 Required data missing. Please complete model training first.")
    st.stop()

base_model = st.session_state["model"]          # keep the trained model intact
X = st.session_state["X_scaled"]
df_encoded = st.session_state["encoded_data"]
target = st.session_state["target_variable"]
features = st.session_state["selected_features"]

# --- Safety checks ---
if target not in df_encoded.columns:
    st.error(f"🚫 Target variable `{target}` not found in encoded data.")
    st.stop()

y = df_encoded[target]

if X.isnull().any().any() or pd.isnull(y).any():
    st.error("🚫 NaNs detected in features or target. Please fix missing values before evaluation.")
    st.stop()

# --- K-Fold Configuration ---
st.subheader("🔁 K-Fold Configuration")
k = st.slider("Number of Folds (K):", min_value=3, max_value=10, value=5)
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# --- Cross-validated predictions (overall) ---
predicted = cross_val_predict(clone(base_model), X, y, cv=kf)
actual = y.reset_index(drop=True)

# --- Fold-by-Fold Metrics (without mutating base_model) ---
mae_list, mse_list, rmse_list, r2_list = [], [], [], []
for train_idx, test_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

    m = clone(base_model)
    m.fit(X_train, y_train)
    y_val_pred = m.predict(X_val)

    mae_list.append(mean_absolute_error(y_val, y_val_pred))
    mse = mean_squared_error(y_val, y_val_pred)
    mse_list.append(mse)
    rmse_list.append(np.sqrt(mse))
    r2_list.append(r2_score(y_val, y_val_pred))

# --- Display Average Metrics ---
st.subheader("📊 Average Evaluation Metrics (Cross-Validation)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{np.mean(mae_list):,.2f}")
col2.metric("MSE", f"{np.mean(mse_list):,.2f}")
col3.metric("RMSE", f"{np.mean(rmse_list):,.2f}")
col4.metric("R²", f"{np.mean(r2_list):.3f}")

# --- Fold-by-Fold Plot ---
metrics_df = pd.DataFrame({
    "Fold": list(range(1, k + 1)),
    "MAE": mae_list,
    "MSE": mse_list,
    "RMSE": rmse_list,
    "R2": r2_list
})
fig = px.line(
    metrics_df, x="Fold", y=["MAE", "MSE", "RMSE", "R2"],
    title="📈 Metrics Across Folds", markers=True,
    template="plotly_white"
)
fig.update_layout(legend_title_text="Metric")
st.plotly_chart(fig, use_container_width=True)

# --- Actual vs Predicted ---
st.subheader("🎯 Actual vs Predicted")
results_df = pd.DataFrame({"Actual": actual, "Predicted": predicted})
try:
    fig1 = px.scatter(
        results_df, x="Actual", y="Predicted", trendline="ols",
        title="Actual vs Predicted", template="plotly_white"
    )
except Exception:
    fig1 = px.scatter(
        results_df, x="Actual", y="Predicted",
        title="Actual vs Predicted", template="plotly_white"
    )
    st.info("Trendline disabled (statsmodels not installed).")
fig1.update_traces(marker=dict(size=6, opacity=0.6))
st.plotly_chart(fig1, use_container_width=True)

# --- Residuals vs Prediction ---
st.subheader("🔍 Residual Analysis")
results_df["Residual"] = results_df["Actual"] - results_df["Predicted"]
try:
    fig2 = px.scatter(
        results_df, x="Predicted", y="Residual", trendline="ols",
        title="Residuals vs Predicted", template="plotly_white"
    )
except Exception:
    fig2 = px.scatter(
        results_df, x="Predicted", y="Residual",
        title="Residuals vs Predicted", template="plotly_white"
    )
    st.info("Trendline disabled (statsmodels not installed).")
fig2.add_hline(y=0, line_dash="dot")
st.plotly_chart(fig2, use_container_width=True)

# --- Residual Histogram ---
fig3 = px.histogram(
    results_df, x="Residual", nbins=30,
    title="Distribution of Residuals", template="plotly_white"
)
fig3.add_vline(x=0, line_dash="dot")
st.plotly_chart(fig3, use_container_width=True)
