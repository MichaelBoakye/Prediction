import streamlit as st
import pandas as pd
import plotly.express as px

# ========== Page Title ==========
st.title("🔍 Data Overview Dashboard")

# ========== Load from Session State ==========
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

# ========== Save Active Copy ==========
st.session_state["df_active"] = df_active

# ========== Sidebar Control ==========
with st.sidebar:
    st.markdown("Select a section to display below:")
    option = st.radio(
        "Choose view",
        options=[
            "Show Variables Overview",
            "Show Descriptive Summary",
            "Show Missing Values"
        ],
        index=0
    )

# ========== Variable Overview ==========
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

    # ---- Summary note for Variables Overview ----
    n_cat = len([c for c in df_active.select_dtypes(include=['object', 'bool', 'category']).columns])
    n_con = len([c for c in df_active.select_dtypes(include=['int64', 'float64']).columns])
    st.markdown(f"""
**🔎 Summary Note:**  
This dataset includes **{n_cat} categorical** fields (e.g., borrower traits and loan attributes) and **{n_con} continuous** fields (e.g., amounts, rates, income, and credit scores).  
Categorical fields help the model learn patterns across borrower or loan types, while continuous fields capture the numeric drivers of credit risk. Together they give a rounded view for analysis and prediction.
""")

# ========== Descriptive Summary ==========
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

# ========== Missing Values ==========
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

    # ---- Summary note for Missing Values ----
    total_missing = int(df_active.isnull().sum().sum())
    cols_with_na = int((df_active.isnull().sum() > 0).sum())
    st.markdown(f"""
**⚠️ Summary Note:**  
There are **{total_missing:,}** missing values across **{cols_with_na}** columns.  
Handle gaps in key predictors like income or credit score with suitable imputation.  
Consistent treatment of missing data improves model stability and accuracy.
""")
