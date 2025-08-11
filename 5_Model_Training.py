import streamlit as st
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import plotly.express as px

st.title("🤖 Model Training - Lasso Regression")

# ===== Session checks =====
required = ["X_scaled", "selected_features", "encoded_data", "scaler"]
if any(k not in st.session_state for k in required):
    st.error("🚫 Required data missing. Please complete preprocessing and feature selection first.")
    st.stop()

X_scaled_df: pd.DataFrame = st.session_state["X_scaled"]
selected_features = st.session_state["selected_features"]
df_encoded: pd.DataFrame = st.session_state["encoded_data"]

# Ensure columns and lengths align
if list(X_scaled_df.columns) != list(selected_features):
    st.info("Selected feature list didn’t match X_scaled columns. Using X_scaled columns.")
    selected_features = list(X_scaled_df.columns)

if len(X_scaled_df) != len(df_encoded):
    st.error("X_scaled and encoded_data have different row counts. Re-run preprocessing/selection.")
    st.stop()

# ===== Target selection =====
st.subheader("🎯 Select Target Variable")
numeric_cols = df_encoded.select_dtypes(include=["number"]).columns.tolist()
if not numeric_cols:
    st.warning("⚠ No numeric columns found in encoded data.")
    st.stop()

default_idx = numeric_cols.index("loan_amount") if "loan_amount" in numeric_cols else 0
target = st.selectbox("Target Variable:", numeric_cols, index=default_idx)
y = df_encoded[target]

# Final NaN guard before split
if X_scaled_df.isnull().any().any() or y.isnull().any():
    st.error("NaNs detected in features or target. Please fix missing values before training.")
    st.stop()

# ===== Train/Test split =====
st.subheader("🔀 Train-Test Split")
test_size = st.slider("Test Size (%)", 10, 50, 20, step=5)
random_state = st.number_input("Random State:", 0, value=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=test_size / 100, random_state=int(random_state)
)

# ===== Train Lasso =====
st.subheader("🧪 Train Lasso Model")
alpha = st.slider("Alpha (Regularization Strength):", 0.001, 1.0, 0.1, 0.01)
# random_state only used if selection='random'; we keep default cyclic
model = Lasso(alpha=float(alpha))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ===== Metrics =====
st.subheader("📊 Model Performance")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

c1, c2, c3 = st.columns(3)
c1.metric("MSE", f"{mse:,.2f}")
c2.metric("RMSE", f"{rmse:,.2f}")
c3.metric("R²", f"{r2:.4f}")

# ===== Coefficients (Feature importance) =====
st.subheader("📉 Feature Importance (Lasso Coefficients)")
coef_df = pd.DataFrame({"Feature": selected_features, "Coefficient": model.coef_})
coef_df["|Coefficient|"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values("|Coefficient|", ascending=False)

fig_coef = px.bar(
    coef_df,
    x="Coefficient",
    y="Feature",
    orientation="h",
    title="Lasso Regression Coefficients",
    color="Coefficient",
    color_continuous_scale="RdBu",
    template="plotly_white"
)
fig_coef.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig_coef, use_container_width=True)
st.dataframe(coef_df.drop(columns="|Coefficient|"), use_container_width=True)

# ===== Actual vs Predicted =====
st.subheader("📌 Actual vs Predicted (Scatter Plot)")
scatter_df = pd.DataFrame({"Actual": y_test.reset_index(drop=True), "Predicted": y_pred})
try:
    fig_scatter = px.scatter(
        scatter_df,
        x="Actual",
        y="Predicted",
        trendline="ols",
        title="Actual vs Predicted Loan Amounts",
        labels={"Actual": "Actual Value", "Predicted": "Predicted Value"},
        template="plotly_white"
    )
except Exception:
    fig_scatter = px.scatter(
        scatter_df,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted Loan Amounts",
        labels={"Actual": "Actual Value", "Predicted": "Predicted Value"},
        template="plotly_white"
    )
    st.info("Trendline disabled (statsmodels not installed).")
fig_scatter.update_traces(marker=dict(size=6, opacity=0.6))
fig_scatter.update_layout(height=500, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig_scatter, use_container_width=True)

# ===== Save model & metadata =====
st.session_state["model"] = model
st.session_state["target_variable"] = target
st.success("✅ Model training complete and saved.")
