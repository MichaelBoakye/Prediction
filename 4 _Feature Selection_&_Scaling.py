import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px

st.title("🔨 Feature Selection & Scaling")

# --------- Guards: data availability ---------
if "X_encoded" not in st.session_state or "y" not in st.session_state:
    st.error("🚫 Encoded data not found. Please complete preprocessing first.")
    st.stop()

# Use numeric-only copy for selectors and scaling
X_full = st.session_state["X_encoded"]
y = st.session_state["y"]

# Filter to numeric columns only (RF, RFE, KBest, scalers all expect numeric)
X_num = X_full.select_dtypes(include=["number"]).copy()

if X_num.empty:
    st.error("No numeric features available after encoding. Check preprocessing.")
    st.stop()

# Drop columns that are entirely NaN (just in case)
all_nan_cols = X_num.columns[X_num.isnull().all()].tolist()
if all_nan_cols:
    st.warning(f"Dropping all-NaN columns: {all_nan_cols}")
    X_num = X_num.drop(columns=all_nan_cols)

# Final NaN check
if X_num.isnull().any().any():
    st.error("There are still missing values in numeric features. Please resolve them in preprocessing.")
    st.stop()

# Let user know if we dropped any non-numeric columns
non_numeric = set(X_full.columns) - set(X_num.columns)
if non_numeric:
    st.info(f"Using {X_num.shape[1]} numeric features. Ignored non-numeric columns: {len(non_numeric)}")

# --------- Feature Selection ---------
st.header("🎯 Feature Selection")

method = st.selectbox(
    "Select Feature Selection Method",
    ["SelectKBest", "RFE (LinearRegression)", "RandomForest Importance"]
)

max_k = min(20, X_num.shape[1])
k = st.slider("Select number of top features", min_value=1, max_value=max_k, value=min(10, max_k))

if method == "SelectKBest":
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_num, y)
    scores = selector.scores_
    support = selector.get_support()
    selected_features = X_num.columns[support]
    feature_scores_df = (
        pd.DataFrame({"Feature": X_num.columns, "Score": scores})
        .sort_values(by="Score", ascending=False)
        .head(k)
    )

elif method == "RFE (LinearRegression)":
    model = LinearRegression()
    selector = RFE(model, n_features_to_select=k)
    selector.fit(X_num, y)
    support = selector.get_support()
    ranking = selector.ranking_
    selected_features = X_num.columns[support]
    feature_scores_df = (
        pd.DataFrame({"Feature": X_num.columns, "Ranking": ranking})
        .sort_values(by="Ranking")
        .rename(columns={"Ranking": "Score"})
        .head(k)
    )

elif method == "RandomForest Importance":
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(X_num, y)
    importances = model.feature_importances_
    feature_scores_df = (
        pd.DataFrame({"Feature": X_num.columns, "Score": importances})
        .sort_values(by="Score", ascending=False)
        .head(k)
    )
    selected_features = feature_scores_df["Feature"].tolist()

# --------- Feature scores bar chart ---------
fig = px.bar(
    feature_scores_df,
    x="Score",
    y="Feature",
    orientation="h",
    title=f"Top {k} Features ({method})",
    template="plotly_white"
)
fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)
st.dataframe(feature_scores_df, use_container_width=True)

# --------- Correlation Heatmap (Top features + target) ---------
st.subheader("🔗 Correlation Heatmap")
top_features = feature_scores_df["Feature"].tolist()
heat_df = pd.concat([X_num[top_features], y.rename(getattr(y, "name", "target"))], axis=1)

corr_matrix = heat_df.corr(numeric_only=True)

try:
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap (Top Features + Target)",
        template="plotly_white",
        width=900,
        height=700
    )
except TypeError:
    # for older plotly without text_auto
    fig_heatmap = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap (Top Features + Target)",
        template="plotly_white",
        width=900,
        height=700
    )

st.plotly_chart(fig_heatmap, use_container_width=True)

# --------- Scaling ---------
st.header("📏 Scaling Features")

scaling_option = st.radio(
    "Choose a scaling method:",
    ["Standardization", "Normalization"],
    horizontal=True
)

scaler = StandardScaler() if scaling_option == "Standardization" else MinMaxScaler()

X_selected_df = X_num[selected_features].copy()
X_scaled = scaler.fit_transform(X_selected_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features, index=X_selected_df.index)

st.success("✅ Features selected and scaled successfully.")

# --------- Save to Session State ---------
st.session_state["X_scaled"] = X_scaled_df
st.session_state["selected_features"] = selected_features if isinstance(selected_features, list) else list(selected_features)
st.session_state["scaler"] = scaler

st.subheader("🔍 Preview of Scaled Features")
st.dataframe(X_scaled_df.head(), use_container_width=True)
