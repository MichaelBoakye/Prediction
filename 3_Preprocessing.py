import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.title("🛠️ Data Preprocessing (Regression: loan_amount)")

# ========= Load dataset from session =========
if "df_active" in st.session_state:
    df_active = st.session_state["df_active"]
elif "df_default" in st.session_state:
    df_active = st.session_state["df_default"]
else:
    st.error("🚫 No dataset found. Please return to the homepage and load a dataset.")
    st.stop()

df_processed = df_active.copy()
st.success("✅ Dataset loaded.")

# ========= Target and features =========
target_col = "loan_amount"  # Regression target
if target_col not in df_processed.columns:
    st.error(f"🚫 Target column `{target_col}` not found.")
    st.stop()

y = df_processed[target_col]
X = df_processed.drop(columns=[target_col])

# ========= Missing values handling =========
st.header("📉 Handling Missing Data")

missing = X.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    st.success("✅ No missing values found.")
else:
    st.warning(f"⚠️ {len(missing)} columns with missing values found.")

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values,
        "Missing (%)": (missing.values / len(X)) * 100,
        "Data Type": [X[col].dtype for col in missing.index]
    }).sort_values("Missing (%)", ascending=False)
    st.dataframe(missing_df, use_container_width=True)

    # Build type lists from X and intersect with columns that are actually missing
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    missing_cols = set(missing.index.tolist())

    cat_cols = [c for c in cat_cols if c in missing_cols]
    num_cols = [c for c in num_cols if c in missing_cols]

    st.subheader("🔢 Strategy for Numeric Columns")
    num_strategy = st.radio(
        "Choose numeric imputation strategy",
        ["Mean", "Median", "Mode", "Custom"],
        horizontal=True
    )

    st.subheader("🔠 Strategy for Categorical Columns")
    cat_strategy = st.radio(
        "Choose categorical imputation strategy",
        ["Mode", "Custom"],
        horizontal=True
    )

    # Collect custom values
    custom_values = {}

    if num_strategy == "Custom" and num_cols:
        st.markdown("Enter custom numeric fills (leave blank to skip a column):")
        for col in num_cols:
            val = st.text_input(f"Custom fill for `{col}`:", key=f"num_custom_{col}")
            if val != "":
                try:
                    custom_values[col] = float(val)
                except ValueError:
                    st.warning(f"Invalid numeric input for `{col}`. Skipped.")

    if cat_strategy == "Custom" and cat_cols:
        st.markdown("Enter custom categorical fills (leave blank to skip a column):")
        for col in cat_cols:
            val = st.text_input(f"Custom fill for `{col}`:", key=f"cat_custom_{col}")
            if val != "":
                custom_values[col] = val

    # Safe mode helper
    def safe_mode(series: pd.Series):
        m = series.mode(dropna=True)
        return m.iloc[0] if not m.empty else np.nan

    # Apply numeric imputation
    for col in num_cols:
        if col in custom_values:
            X[col] = X[col].fillna(custom_values[col])
        elif num_strategy == "Mean":
            X[col] = X[col].fillna(X[col].mean())
        elif num_strategy == "Median":
            X[col] = X[col].fillna(X[col].median())
        elif num_strategy == "Mode":
            X[col] = X[col].fillna(safe_mode(X[col]))

    # Apply categorical imputation
    for col in cat_cols:
        if col in custom_values:
            X[col] = X[col].fillna(custom_values[col])
        elif cat_strategy == "Mode":
            X[col] = X[col].fillna(safe_mode(X[col]))

    # Verify post-imputation
    updated_missing = X.isnull().sum()
    remaining = updated_missing[updated_missing > 0]
    if not remaining.empty:
        st.warning("⚠️ Some missing values remain:")
        st.dataframe(remaining)
    else:
        st.info("📜 No missing values remaining.")

# ========= Encoding =========
st.header("🌡️ Encoding Categorical Variables")

cat_cols_all = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
X_encoded = X.copy()

if not cat_cols_all:
    encoding_strategy = None
    st.success("✅ No categorical columns found.")
else:
    encoding_strategy = st.radio(
        "Choose encoding strategy",
        ["Label Encoding", "One-Hot Encoding"],
        horizontal=True
    )

    # Replace NaN with a clear marker before encoding
    for col in cat_cols_all:
        X_encoded[col] = X_encoded[col].astype("object").where(~X_encoded[col].isna(), "Unknown")

    if encoding_strategy == "Label Encoding":
        label_encoders = {}
        for col in cat_cols_all:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            label_encoders[col] = le
        st.session_state["label_encoders"] = label_encoders
        st.success("✅ Label Encoding applied.")

    elif encoding_strategy == "One-Hot Encoding":
        X_encoded = pd.get_dummies(X_encoded, columns=cat_cols_all, drop_first=False)
        st.success("✅ One-Hot Encoding applied.")

# ========= Save for next pages =========
st.session_state["X_encoded"] = X_encoded
st.session_state["y"] = y
st.session_state["encoding_strategy"] = encoding_strategy
st.session_state["categorical_columns"] = cat_cols_all
st.session_state["encoded_data"] = X_encoded.join(y)

st.success("✅ Preprocessing complete. You can proceed to feature selection, scaling, and model training.")

# ========= Preview =========
st.subheader("🔍 Preview of Preprocessed Data")
st.dataframe(X_encoded.head(), use_container_width=True)

# Optional quick summary
st.markdown(f"""
**Summary:**  
- Target: **{target_col}** (regression)  
- Rows: **{len(X_encoded):,}**  
- Features after encoding: **{X_encoded.shape[1]:,}**  
- Categorical encoding: **{encoding_strategy if encoding_strategy else "None"}**
""")
