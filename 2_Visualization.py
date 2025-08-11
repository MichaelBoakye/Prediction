import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ----------------- Helpers -----------------
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

# ----------------- Page -----------------
st.title("📈 Visual Data Explorer")

df_active = get_df_from_session()
st.success("✅ Dataset loaded from session.")

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("📊 Select Visualization")
    chart_option = st.radio(
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

# ----------------- 1) Histogram -----------------
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

# ----------------- 2) Box by Property Value -----------------
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

# ----------------- 3) Gender Bar -----------------
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
Female applicants are the smallest, and some entries have missing or “Not Available” gender.  
Imbalance can affect fairness and should be handled during preprocessing.
        """)
    else:
        st.warning("`gender` column not found.")

# ----------------- 4) Scatter (clean style) -----------------
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

# ----------------- 5) Correlation Heatmap -----------------
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

# ----------------- 6) All Box Plots (Numeric) -----------------
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
