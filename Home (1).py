import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import zipfile
import json
import base64
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.base import clone

# Set page config
st.set_page_config(page_title="Loan Default Prediction", layout="wide", page_icon="💸")

# Function to encode image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpg;base64,{encoded}"

# ========== Page 1: Home ==========
def home_page():
    # Inject the image as a full-width banner
    try:
        image_path = os.path.join(os.path.dirname(__file__), "loan.jpg")
        image_base64 = get_base64_image(image_path)
        st.markdown(
            f"""
            <div style='width:100%; text-align:center;'>
                <img src="{image_base64}" style="width:100%; max-height:250px; object-fit:cover;" alt="Loan Header Image">
            </div>
            """,
            unsafe_allow_html=True
        )
    except:
        pass

    # Introduction
    st.markdown("""
    ### 📊 Predict Loan Defaults Based on Financial and Demographic Data

    Banks generate major revenue from lending, but it often comes with risk—borrowers may default on their loans. To address this, banks are turning to Machine Learning to improve credit risk assessments.

    They've collected historical data on past borrowers and now want you to build a robust ML model to predict whether new applicants are likely to default.

    The dataset includes multiple deterministic factors like borrower income, gender, and loan purpose. 
    """)

    st.markdown("""
    Use the sidebar to explore:
    - 🔍 Data Exploration  
    - 📊 Visualization
    - 🛠️ Preprocessing  
    - 🔨 Feature Selection & Scaling
    - 🤖 Model training  
    - 📉 Evaluation  
    - 🧮 Interactive predictions
    """)

    # Data loading section
    st.subheader("📂 Load Your Data")

    # Option 1: Upload your own data
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Upload your own dataset in CSV or Excel format"
    )

    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:  # Excel file
                df = pd.read_excel(uploaded_file)

            # Store in session state
            st.session_state["df_default"] = df
            st.session_state["df_active"] = df.copy()
            st.success("✅ Your dataset was loaded successfully!")

            # Show preview
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

    # Option 2: Use sample data
    st.markdown("---")
    st.subheader("Or use sample data")

    if st.button("Load Sample Dataset"):
        @st.cache_data(show_spinner=False)
        def load_sample_dataset():
            try:
                # Create a simple synthetic dataset as fallback
                np.random.seed(42)
                n_samples = 1000

                df = pd.DataFrame({
                    'loan_amount': np.random.normal(100000, 30000, n_samples).astype(int),
                    'property_value': np.random.normal(150000, 50000, n_samples).astype(int),
                    'income': np.random.normal(75000, 20000, n_samples).astype(int),
                    'credit_score': np.random.normal(700, 50, n_samples).astype(int),
                    'loan_term': np.random.choice([15, 30], n_samples),
                    'employment_years': np.random.randint(0, 20, n_samples),
                    'dti_ratio': np.random.uniform(10, 50, n_samples).round(1),
                    'gender': np.random.choice(['Male', 'Female', 'Joint'], n_samples),
                    'default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
                })

                # Ensure no negative values for amounts
                df['loan_amount'] = df['loan_amount'].abs()
                df['property_value'] = df['property_value'].abs()
                df['income'] = df['income'].abs()
                df['credit_score'] = df['credit_score'].clip(300, 850)

                return df

            except Exception as e:
                st.error(f"Failed to create sample data: {e}")
                return None

        with st.spinner("Creating sample data..."):
            sample_df = load_sample_dataset()
            if sample_df is not None:
                st.session_state["df_default"] = sample_df
                st.session_state["df_active"] = sample_df.copy()
                st.success("✅ Sample dataset created successfully.")
                st.dataframe(sample_df.head())

                st.warning("Note: This is synthetic data. For better results, upload your own dataset.")
            else:
                st.error("Failed to create sample data. Please try uploading your own file.")


# ========== Page 2: Data Exploration ==========
def data_exploration():
    st.title("🔍 Data Overview Dashboard")

    # Load from Session State
    if "df_default" in st.session_state:
        df_active = st.session_state["df_default"]
        st.success("✅ Using Kaggle default dataset.")
    else:
        st.warning("⚠️ Refresh homepage.")
        st.stop()

    # Stop early if dataset is empty
    if df_active.empty:
        st.warning("Dataset is empty. Please load data on the homepage.")
        st.stop()

    # Save Active Copy
    st.session_state["df_active"] = df_active

    # Sidebar Control
    option = st.sidebar.radio(
        "Choose view",
        options=[
            "Show Variables Overview",
            "Show Descriptive Summary",
            "Show Missing Values"
        ],
        index=0
    )

    # Variable Overview
    if option == "Show Variables Overview":
        with st.expander("📌 Data Variables", expanded=True):
            categorical = df_active.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
            continuous = df_active.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # Safe padding even if both lists are empty
            max_len = max(len(categorical), len(continuous), 1)
            categorical += [''] * (max_len - len(categorical))
            continuous += [''] * (max_len - len(continuous))

            grouped_df = pd.DataFrame({
                'Categorical Variables': [c.replace('_', ' ').title() if c else '' for c in categorical],
                'Continuous Variables': [c.replace('_', ' ').title() if c else '' for c in continuous]
            })

            st.dataframe(grouped_df, use_container_width=True)

        # Summary note for Variables Overview
        n_cat = len([c for c in df_active.select_dtypes(include=['object', 'bool', 'category']).columns])
        n_con = len([c for c in df_active.select_dtypes(include=['int64', 'float64']).columns])
        st.markdown(f"""
    **🔎 Summary Note:**  
    This dataset includes **{n_cat} categorical** fields (e.g., borrower traits and loan attributes) and **{n_con} continuous** fields (e.g., amounts, rates, income, and credit scores).  
    Categorical fields help the model learn patterns across borrower or loan types, while continuous fields capture the numeric drivers of credit risk. Together they give a rounded view for analysis and prediction.
    """)

    # Descriptive Summary
    elif option == "Show Descriptive Summary":
        st.markdown("**Continuous Variables**")
        con_summary = df_active.describe().T.applymap(
            lambda x: f"{x:,.2f}" if pd.notnull(x) and isinstance(x, (int, float)) else ""
        )
        st.dataframe(con_summary, use_container_width=True)

        st.markdown("""
    **📊 Summary Note:**  
    The table shows averages, spread, and range for numeric fields. Watch for skewed values and outliers that can pull the mean up or down.  
    This view guides steps like scaling, outlier handling, or log transforms before model training.
    """)

        st.markdown("**Categorical Variables**")
        cat_summary = df_active.describe(include=['object', 'category', 'bool']).T
        st.dataframe(cat_summary, use_container_width=True)

        st.markdown("""
    **🧩 Categorical Note:**  
    The table highlights how many unique categories exist and the most frequent label.  
    Imbalanced categories and rare labels may need regrouping (e.g., an **Other** bucket) and careful encoding.
    """)

    # Missing Values
    elif option == "Show Missing Values":
        st.markdown("📌 Missing Values")
        cat_col = df_active.select_dtypes(include=['object', 'bool', 'category'])
        con_col = df_active.select_dtypes(include=['int64', 'float64'])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Categorical Variables")
            cat_missing = (
                cat_col.isnull()
                       .sum()
                       .reset_index()
                       .rename(columns={'index': 'Variable', 0: 'Missing Count'})
                       .sort_values('Missing Count', ascending=False)
            )
            st.dataframe(cat_missing, use_container_width=True)

        with col2:
            st.subheader("Continuous Variables")
            con_missing = (
                con_col.isnull()
                       .sum()
                       .reset_index()
                       .rename(columns={'index': 'Variable', 0: 'Missing Count'})
                       .sort_values('Missing Count', ascending=False)
            )
            st.dataframe(con_missing, use_container_width=True)

        # Summary note for Missing Values
        total_missing = int(df_active.isnull().sum().sum())
        cols_with_na = int((df_active.isnull().sum() > 0).sum())
        st.markdown(f"""
    **⚠️ Summary Note:**  
    There are **{total_missing:,}** missing values across **{cols_with_na}** columns.  
    Handle gaps in key predictors like income or credit score with suitable imputation.  
    Consistent treatment of missing data improves model stability and accuracy.
    """)

# ========== Page 3: Visualization ==========
def visualization():
    # Helpers
    def get_df_from_session():
        """Load dataframe from session; stop if missing."""
        if "df_active" in st.session_state:
            return st.session_state["df_active"]
        if "df_default" in st.session_state:
            return st.session_state["df_default"]
        st.error("🚫 No dataset found. Please return to the homepage and load data.")
        st.stop()

    def find_col(df, target):
        """Case/underscore-insensitive match to a column name."""
        t = target.lower().replace(" ", "_")
        for c in df.columns:
            if c.lower().replace(" ", "_") == t:
                return c
        return None

    # Page
    st.title("📈 Visual Data Explorer")

    df_active = get_df_from_session()
    st.success("✅ Dataset loaded from session.")

    # Sidebar
    chart_option = st.sidebar.radio(
        "Choose a chart to display:",
        [
            "Loan Amount Distribution",
            "Loan Amount by Property Value",
            "Gender Distribution",
            "Credit Score vs Loan Amount",
            "Correlation Matrix",
            "All Box Plots (Numeric)"
        ]
    )

    # 1) Histogram
    if chart_option == "Loan Amount Distribution":
        col_loan = find_col(df_active, "loan_amount")
        if col_loan:
            st.subheader("🔹 Loan Amount Distribution")
            fig = px.histogram(df_active, x=col_loan, nbins=50, template="plotly_white")
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
    **Insight:**  
    The distribution is right-skewed. Most loans sit between 100k and 500k, while a small number exceed 1M.  
    Those extreme values are outliers and may need capping or a log transform before modelling.
            """)
        else:
            st.warning("`loan_amount` column not found.")

    # 2) Box by Property Value
    elif chart_option == "Loan Amount by Property Value":
        col_prop = find_col(df_active, "property_value")
        col_loan = find_col(df_active, "loan_amount")
        if col_prop and col_loan:
            st.subheader("🔹 Loan Amount by Property Value")
            fig = px.box(df_active, x=col_prop, y=col_loan, template="plotly_white")
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
    **Insight:**  
    Loan size tends to rise with property value. Most loans cluster around low to mid property values, with a few high-value outliers.  
    Consider scaling and robust models so these extremes do not dominate the fit.
            """)
        else:
            st.warning("`property_value` and/or `loan_amount` column not found.")

    # 3) Gender Bar
    elif chart_option == "Gender Distribution":
        col_gender = find_col(df_active, "gender")
        if col_gender:
            st.subheader("🔹 Gender Distribution")
            gender_counts = df_active[col_gender].value_counts(dropna=False).reset_index()
            gender_counts.columns = ["Gender", "Count"]
            fig = px.bar(gender_counts, x="Gender", y="Count", color="Gender", template="plotly_white")
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
    **Insight:**  
    Male applicants are the largest group, followed by joint applications.  
    Female applicants are the smallest, and some entries have missing or "Not Available" gender.  
    Imbalance can affect fairness and should be handled during preprocessing.
            """)
        else:
            st.warning("`gender` column not found.")

    # 4) Scatter (clean style)
    elif chart_option == "Credit Score vs Loan Amount":
        col_cs = find_col(df_active, "credit_score")
        col_loan = find_col(df_active, "loan_amount")
        if col_cs and col_loan:
            st.subheader("🔹 Credit Score vs Loan Amount")

            df_scatter = df_active[[col_cs, col_loan]].dropna()
            # Downsample if very large for speed
            if len(df_scatter) > 100_000:
                df_scatter = df_scatter.sample(100_000, random_state=42)

            try:
                fig = px.scatter(
                    df_scatter,
                    x=col_cs,
                    y=col_loan,
                    render_mode="webgl",   # faster
                    opacity=0.6,
                    trendline="ols",
                    trendline_color_override="red",
                    template="plotly_white",
                    title="Credit Score vs Loan Amount"
                )
            except Exception:
                fig = px.scatter(
                    df_scatter,
                    x=col_cs,
                    y=col_loan,
                    render_mode="webgl",
                    opacity=0.6,
                    template="plotly_white",
                    title="Credit Score vs Loan Amount"
                )
                st.info("Trendline disabled (statsmodels not installed).")

            fig.update_traces(marker={"size": 5, "line": {"width": 0}})
            fig.update_layout(height=520, hovermode="closest", margin=dict(l=10, r=10, t=60, b=10))
            fig.update_xaxes(title="Credit Score", showgrid=True, zeroline=False)
            fig.update_yaxes(title="Loan Amount", showgrid=True, zeroline=False)

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
    **Detailed Insight:**  
    Credit score alone does not explain loan size well. Big loans appear at both low and high scores.  
    What to note:
    1. Loan sizes cluster below 500k across the score range, likely due to product caps or policy rules.  
    2. Any trend is weak. Interactions with income, property value, LTV and product type will likely matter more.  
    3. A few points are unusually large given their score. Check for data quality issues or special loan products.  
    Next step: test interactions or non-linear terms and compare performance to a baseline linear fit.
            """)
        else:
            st.warning("`credit_score` and/or `loan_amount` column not found.")

    # 5) Correlation Heatmap
    elif chart_option == "Correlation Matrix":
        st.subheader("🔹 Correlation Matrix (Continuous Variables)")
        con_vars = df_active.select_dtypes(include=["float64", "int64"]).copy()
        # remove obvious IDs/Years (case-insensitive)
        con_vars = con_vars[[c for c in con_vars.columns if not c.lower().startswith(("id", "year"))]]

        if not con_vars.empty:
            corr = con_vars.corr(numeric_only=True)
            try:
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                                     color_continuous_scale="RdBu_r", template="plotly_white")
            except TypeError:
                fig_corr = px.imshow(corr, aspect="auto", color_continuous_scale="RdBu_r",
                                     template="plotly_white")
            fig_corr.update_layout(height=800, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_corr, use_container_width=True)

            # Top 5 absolute correlations (excluding self-pairs)
            mask = ~np.eye(len(corr), dtype=bool)
            top_corrs = (
                corr.where(mask)
                    .abs()
                    .unstack()
                    .dropna()
                    .sort_values(ascending=False)
                    .head(5)
            )
            bullets = "\n".join([f"- **{a} ↔ {b}**: {v:.2f}" for (a, b), v in top_corrs.items()])
            st.markdown(f"""
    **Insight:**  
    Strong correlations can indicate redundancy or multicollinearity. Consider feature selection if needed.  
    Top relationships in this data:  
    {bullets}
            """)
        else:
            st.warning("No numeric columns available after removing ID/Year-like fields.")

    # 6) All Box Plots (Numeric)
    elif chart_option == "All Box Plots (Numeric)":
        st.subheader("🔹 Box Plots for All Numeric Features")

        con_vars = df_active.select_dtypes(include=["float64", "int64"]).copy()
        con_vars = con_vars[[c for c in con_vars.columns if not c.lower().startswith(("id", "year"))]]

        if con_vars.empty:
            st.warning("No eligible numeric columns available for box plots.")
        else:
            # Downsample rows for speed if huge
            df_box = con_vars
            if len(df_box) > 200_000:
                df_box = df_box.sample(200_000, random_state=42)

            long_df = df_box.melt(var_name="Feature", value_name="Value").dropna()

            # Order by variance (most variable first)
            order = (
                long_df.groupby("Feature")["Value"]
                       .var()
                       .sort_values(ascending=False)
                       .index
                       .tolist()
            )

            fig = px.box(
                long_df,
                x="Feature",
                y="Value",
                points="outliers",  # set to False to render even faster
                category_orders={"Feature": order},
                template="plotly_white",
                title="Box Plots of Numeric Features"
            )
            fig.update_layout(
                height=620,
                margin=dict(l=10, r=10, t=60, b=10),
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
    **Detailed Insight:**  
    This view shows distribution, spread, and outliers for every numeric feature in one place.  
    What to look for:  
    1. **Scale gaps**: very large scales (loan amount, property value) vs small scales (rates, ratios). Plan to scale before modelling.  
    2. **Outliers**: many dots beyond the whiskers can skew training. Consider winsorising or robust scalers.  
    3. **Skewed features**: long whiskers or off-centre boxes often benefit from log or Box-Cox transforms.  
    4. **High-variance features**: those at the left tend to dominate distance-based models unless you standardise.  
    Action: decide which features to transform, cap, or standardise based on their box shapes and your model choice.
            """)

# ========== Page 4: Preprocessing ==========
def preprocessing():
    st.title("🛠️ Data Preprocessing (Regression: loan_amount)")

    # Load dataset from session
    if "df_active" in st.session_state:
        df_active = st.session_state["df_active"]
    elif "df_default" in st.session_state:
        df_active = st.session_state["df_default"]
    else:
        st.error("🚫 No dataset found. Please return to the homepage and load a dataset.")
        st.stop()

    df_processed = df_active.copy()
    st.success("✅ Dataset loaded.")

    # Target and features
    target_col = "loan_amount"  # Regression target
    if target_col not in df_processed.columns:
        st.error(f"🚫 Target column `{target_col}` not found.")
        st.stop()

    y = df_processed[target_col]
    X = df_processed.drop(columns=[target_col])

    # Missing values handling
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

    # Encoding
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

    # Save for next pages
    st.session_state["X_encoded"] = X_encoded
    st.session_state["y"] = y
    st.session_state["encoding_strategy"] = encoding_strategy
    st.session_state["categorical_columns"] = cat_cols_all
    st.session_state["encoded_data"] = X_encoded.join(y)

    st.success("✅ Preprocessing complete. You can proceed to feature selection, scaling, and model training.")

    # Preview
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

# ========== Page 5: Feature Selection & Scaling ==========
def feature_selection_scaling():
    st.title("🔨 Feature Selection & Scaling")

    # Guards: data availability
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

    # Feature Selection
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

    # Feature scores bar chart
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

    # Correlation Heatmap (Top features + target)
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

    # Scaling
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

    # Save to Session State
    st.session_state["X_scaled"] = X_scaled_df
    st.session_state["selected_features"] = selected_features if isinstance(selected_features, list) else list(selected_features)
    st.session_state["scaler"] = scaler

    st.subheader("🔍 Preview of Scaled Features")
    st.dataframe(X_scaled_df.head(), use_container_width=True)

# ========== Page 6: Model Training ==========
def model_training():
    st.title("🤖 Model Training - Lasso Regression")

    # Session checks
    required = ["X_scaled", "selected_features", "encoded_data", "scaler"]
    if any(k not in st.session_state for k in required):
        st.error("🚫 Required data missing. Please complete preprocessing and feature selection first.")
        st.stop()

    X_scaled_df: pd.DataFrame = st.session_state["X_scaled"]
    selected_features = st.session_state["selected_features"]
    df_encoded: pd.DataFrame = st.session_state["encoded_data"]

    # Ensure columns and lengths align
    if list(X_scaled_df.columns) != list(selected_features):
        st.info("Selected feature list didn't match X_scaled columns. Using X_scaled columns.")
        selected_features = list(X_scaled_df.columns)

    if len(X_scaled_df) != len(df_encoded):
        st.error("X_scaled and encoded_data have different row counts. Re-run preprocessing/selection.")
        st.stop()

    # Target selection
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

    # Train/Test split
    st.subheader("🔀 Train-Test Split")
    test_size = st.slider("Test Size (%)", 10, 50, 20, step=5)
    random_state = st.number_input("Random State:", 0, value=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=test_size / 100, random_state=int(random_state)
    )

    # Train Lasso
    st.subheader("🧪 Train Lasso Model")
    alpha = st.slider("Alpha (Regularization Strength):", 0.001, 1.0, 0.1, 0.01)
    # random_state only used if selection='random'; we keep default cyclic
    model = Lasso(alpha=float(alpha))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    st.subheader("📊 Model Performance")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("MSE", f"{mse:,.2f}")
    c2.metric("RMSE", f"{rmse:,.2f}")
    c3.metric("R²", f"{r2:.4f}")

    # Coefficients (Feature importance)
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

    # Actual vs Predicted
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

    # Save model & metadata
    st.session_state["model"] = model
    st.session_state["target_variable"] = target
    st.success("✅ Model training complete and saved.")

# ========== Page 7: Evaluation ==========
def evaluation():
    st.title("📉 Model Evaluation - K-Fold Cross-Validation")

    # Check prerequisites
    required_keys = ["model", "X_scaled", "encoded_data", "target_variable", "selected_features"]
    if any(key not in st.session_state for key in required_keys):
        st.error("🚫 Required data missing. Please complete model training first.")
        st.stop()

    base_model = st.session_state["model"]          # keep the trained model intact
    X = st.session_state["X_scaled"]
    df_encoded = st.session_state["encoded_data"]
    target = st.session_state["target_variable"]
    features = st.session_state["selected_features"]

    # Safety checks
    if target not in df_encoded.columns:
        st.error(f"🚫 Target variable `{target}` not found in encoded data.")
        st.stop()

    y = df_encoded[target]

    if X.isnull().any().any() or pd.isnull(y).any():
        st.error("🚫 NaNs detected in features or target. Please fix missing values before evaluation.")
        st.stop()

    # K-Fold Configuration
    st.subheader("🔁 K-Fold Configuration")
    k = st.slider("Number of Folds (K):", min_value=3, max_value=10, value=5)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Cross-validated predictions (overall)
    predicted = cross_val_predict(clone(base_model), X, y, cv=kf)
    actual = y.reset_index(drop=True)

    # Fold-by-Fold Metrics (without mutating base_model)
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

    # Display Average Metrics
    st.subheader("📊 Average Evaluation Metrics (Cross-Validation)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{np.mean(mae_list):,.2f}")
    col2.metric("MSE", f"{np.mean(mse_list):,.2f}")
    col3.metric("RMSE", f"{np.mean(rmse_list):,.2f}")
    col4.metric("R²", f"{np.mean(r2_list):.3f}")

    # Fold-by-Fold Plot
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

    # Actual vs Predicted
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

    # Residuals vs Prediction
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

    # Residual Histogram
    fig3 = px.histogram(
        results_df, x="Residual", nbins=30,
        title="Distribution of Residuals", template="plotly_white"
    )
    fig3.add_vline(x=0, line_dash="dot")
    st.plotly_chart(fig3, use_container_width=True)

# ========== Page 8: Prediction ==========
def prediction():
    st.title("📈 Make Predictions")

    # Checks
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

    # Helpers
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

    # Single Input Form
    st.subheader("📝 Enter Feature Values")

    inputs = {}
    for f in feature_order:
        if f in label_encoders and encoding_strategy == "Label Encoding":
            classes = label_encoders[f].classes_.tolist()
            val = st.selectbox(f"{f} (categorical)", classes, key=f"inp_{f}")
            inputs[f] = encode_label_value(f, val)
        else:
            inputs[f] = st.number_input(f"{f} (numeric)", value=0.0, key=f"inp_{f}")

    # Predict Button
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

# ========== Page 9: Model Interpretation ==========
def model_interpretation():
    st.title("🧐 Model Interpretation")

    # Prerequisites
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

    # 1) Coefficients: directional impact (Lasso/linear only)
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

    # 2) Permutation Importance: what the model actually uses
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

        # Score by default is model.score (R² for regressors). We'll report mean decrease.
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
    Permutation importance shuffles one feature at a time and checks how much the model's score drops.  
    Bigger drops mean the model relied on that feature more. This is a good sanity check that often differs from raw coefficients.
    """)

    # 3) Partial Dependence: average effect curves
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
    Each chart shows the model's average prediction as we vary one feature, holding others near typical values.  
    Flat lines mean little effect. Slopes or bends suggest stronger influence or non-linear behavior.
    """)

    # 4) What-if simulator
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

    # 5) Takeaways
    st.header("📝 Quick Takeaways")
    st.markdown("""
    - **Direction vs dependence:** Coefficients show the signed effect for linear models.  
      Permutation importance shows how much the model relied on each feature in practice.
    - **Non-linearities:** Partial dependence can reveal curved or threshold effects that coefficients may miss.
    - **Actionable levers:** Use the what-if simulator to explain how changes in key drivers might influence the predicted loan amount.
    """)

# ========== Main App ==========
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠 Home",
            "🔍 Data Exploration",
            "📊 Visualization",
            "🛠️ Preprocessing",
            "🔨 Feature Selection & Scaling",
            "🤖 Model Training",
            "📉 Evaluation",
            "📈 Prediction",
            "🧐 Model Interpretation"
        ]
    )

    if page == "🏠 Home":
        home_page()
    elif page == "🔍 Data Exploration":
        data_exploration()
    elif page == "📊 Visualization":
        visualization()
    elif page == "🛠️ Preprocessing":
        preprocessing()
    elif page == "🔨 Feature Selection & Scaling":
        feature_selection_scaling()
    elif page == "🤖 Model Training":
        model_training()
    elif page == "📉 Evaluation":
        evaluation()
    elif page == "📈 Prediction":
        prediction()
    elif page == "🧐 Model Interpretation":
        model_interpretation()

if __name__ == "__main__":
    main()
