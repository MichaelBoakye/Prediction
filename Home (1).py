import streamlit as st
import os
import zipfile
import pandas as pd
import subprocess
import json
import base64

        # ========== STYLING & HEADER ==========

import streamlit as st
import base64
import os

# Set page config
st.set_page_config(page_title="Loan Default Prediction", layout="wide", page_icon="💸")

# Function to encode image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpg;base64,{encoded}"

# Ensure the image is in the same folder as this script
image_path = os.path.join(os.path.dirname(__file__), "loan.jpg")  # or "loan.jpg"

# Get the base64-encoded image
image_base64 = get_base64_image(image_path)

# Inject the image as a full-width banner
st.markdown(
    f"""
    <div style='width:100%; text-align:center;'>
        <img src="{image_base64}" style="width:100%; max-height:250px; object-fit:cover;" alt="Loan Header Image">
    </div>
    """,
    unsafe_allow_html=True
)



# ========== INTRODUCTION ==========
st.markdown("""
### 📊 Predict Loan Defaults Based on Financial and Demographic Data

Banks generate major revenue from lending, but it often comes with risk—borrowers may default on their loans. To address this, banks are turning to Machine Learning to improve credit risk assessments.

They’ve collected historical data on past borrowers and now want you to build a robust ML model to predict whether new applicants are likely to default.

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

# ========== LOAD KAGGLE DEFAULT DATA ==========

@st.cache_data(show_spinner=False)
def load_default_dataset():
    try:
        with open(".kaggle/kaggle.json") as f:
            creds = json.load(f)
            os.environ["KAGGLE_USERNAME"] = creds["username"]
            os.environ["KAGGLE_KEY"] = creds["key"]

        dataset = "yasserh/loan-default-dataset"
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset], check=True)

        zip_file = "loan-default-dataset.zip"
        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall("loan_dataset")

        csv_path = os.path.join("loan_dataset", "loan_default.csv")
        df = pd.read_csv(csv_path)
        return df

    except Exception as e:
        st.error(f"❌ Failed to load Kaggle dataset: {e}")
        return None

df_default = load_default_dataset()

if df_default is not None:
    st.session_state["df_default"] = df_default
    st.success("✅ Default Kaggle dataset loaded successfully.")
    st.dataframe(df_default.head())
