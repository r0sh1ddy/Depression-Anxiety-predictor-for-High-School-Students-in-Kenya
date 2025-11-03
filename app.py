# ==============================================
# Mental Health Screening App - Kenya (Final)
# ==============================================
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import base64

# ==============================================
# PAGE CONFIG & BACKGROUND IMAGE
# ==============================================
st.set_page_config(
    page_title="Mental Health Screening Tool - Kenya",
    page_icon="🧠",
    layout="wide",
)

# ---- Set global background ----
def add_bg_from_local(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 👉 Change this path to your background image
add_bg_from_local("background.jpg")

# ==============================================
# APP TITLE
# ==============================================
st.title("🇰🇪 AI-Powered Mental Health Screening Tool")
st.markdown("#### For High School Students in Kenya")
st.write("---")

# ==============================================
# LOAD TRAINED MODELS
# ==============================================
model_files = {
    "Logistic": "Logistic_model.pkl",
    "RandomForest": "RandomForest_model.pkl",
    "XGBoost": "XGBoost_model.pkl",
    "LightGBM": "LightGBM_model.pkl"
}

pipelines = {}
for name, file in model_files.items():
    try:
        pipelines[name] = joblib.load(file)
    except Exception as e:
        st.warning(f"⚠️ Could not load {file}: {e}")

# ==============================================
# USER INPUT SECTION
# ==============================================
st.sidebar.header("🧍‍♀️ Student Information")

boarding_day = st.sidebar.selectbox("🏫 School Type", ["Boarding", "Day", "Day & Boarding"])
school_type = st.sidebar.selectbox("🏫 School Category", ["County", "Extracounty", "Subcounty"])
school_demo = st.sidebar.selectbox("👩‍🏫 School Demographics", ["Boys", "Girls", "Mixed"])
school_county = st.sidebar.selectbox("📍 County", ["Kiambu", "Machakos", "Makueni", "Nairobi"])
age = st.sidebar.slider("🎂 Age", 12, 25, 16)
gender = st.sidebar.selectbox("⚧ Gender", ["Male", "Female"])
form = st.sidebar.selectbox("📘 Form", [1, 2, 3, 4])
religion = st.sidebar.selectbox("🙏 Religion", ["Christian", "Muslim", "Other"])
parents_home = st.sidebar.selectbox("🏠 Parents Living Situation", ["Both parents", "One parent", "None"])
parents_dead = st.sidebar.number_input("☠️ Deceased Parents", min_value=0, max_value=4, value=0)
fathers_edu = st.sidebar.selectbox("👨 Father's Education", ["None", "Primary", "Secondary", "Tertiary", "University"])
mothers_edu = st.sidebar.selectbox("👩 Mother's Education", ["None", "Primary", "Secondary", "Tertiary", "University"])
co_curr = st.sidebar.selectbox("🎭 Co-Curricular Activities", ["Yes", "No"])
sports = st.sidebar.selectbox("⚽ Sports Participation", ["Yes", "No"])
acad_ability = st.sidebar.slider("📈 Academic Self-Rating", 1, 5, 3)

# ==============================================
# ENCODING INPUTS
# ==============================================
encoded_data = {
    "Boarding_day_encoded": {"Boarding": 1, "Day": 2, "Day & Boarding": 3}[boarding_day],
    "School_type_encoded": {"County": 1, "Extracounty": 2, "Subcounty": 3}[school_type],
    "School_Demographics_encoded": {"Boys": 1, "Girls": 2, "Mixed": 3}[school_demo],
    "School_County_encoded": {"Kiambu": 1, "Machakos": 2, "Makueni": 3, "Nairobi": 4}[school_county],
    "Age": age,
    "Gender": 1 if gender == "Male" else 2,
    "Form": form,
    "Religion": 1 if religion == "Christian" else 2 if religion == "Muslim" else 3,
    "Parents_Home": {"None": 0, "One parent": 1, "Both parents": 2}[parents_home],
    "Parents_Dead": parents_dead,
    "Fathers_Education": {"None": 1, "Primary": 2, "Secondary": 3, "Tertiary": 4, "University": 5}[fathers_edu],
    "Mothers_Education": {"None": 1, "Primary": 2, "Secondary": 3, "Tertiary": 4, "University": 5}[mothers_edu],
    "Co_Curricular": 1 if co_curr == "Yes" else 0,
    "Sports": 1 if sports == "Yes" else 0,
    "Percieved_Academic_Abilities": acad_ability
}

user_df = pd.DataFrame([encoded_data])

# ==============================================
# PREDICT SECTION
# ==============================================
st.markdown("---")
st.header("🔮 Screening Results")

try:
    best_dep_model = "LightGBM"
    best_anx_model = "LightGBM"

    dep_model = pipelines.get(best_dep_model)
    anx_model = pipelines.get(best_anx_model)

    phq_total = int(dep_model.predict(user_df)[0])
    gad_total = int(anx_model.predict(user_df)[0])
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# ==============================================
# SEVERITY FUNCTION
# ==============================================
def get_severity_info(score, target="depression"):
    if target == "depression":
        if score < 5: return {"level": "Minimal", "color": "#4caf50"}
        elif score < 10: return {"level": "Mild", "color": "#ffc107"}
        elif score < 15: return {"level": "Moderate", "color": "#ff9800"}
        elif score < 20: return {"level": "Moderately Severe", "color": "#f57c00"}
        else: return {"level": "Severe", "color": "#f44336"}
    else:
        if score < 5: return {"level": "Minimal", "color": "#4caf50"}
        elif score < 10: return {"level": "Mild", "color": "#ffc107"}
        elif score < 15: return {"level": "Moderate", "color": "#ff9800"}
        else: return {"level": "Severe", "color": "#f44336"}

dep_info = get_severity_info(phq_total, "depression")
anx_info = get_severity_info(gad_total, "anxiety")

# ==============================================
# DISPLAY RESULTS WITH IMAGE CARDS
# ==============================================
dep_card = f"""
<div style="
    background-image: url('https://cdn.pixabay.com/photo/2020/04/09/09/36/mental-health-5022268_1280.jpg');
    background-size: cover;
    background-position: center;
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
">
    <h3 style="text-align:center;">Depression Assessment</h3>
    <h2 style="text-align:center; color:{dep_info['color']};">
        {dep_info['level']} (PHQ-8 Score: {phq_total})
    </h2>
</div>
"""

anx_card = f"""
<div style="
    background-image: url('https://cdn.pixabay.com/photo/2020/05/21/11/13/anxiety-5191260_1280.jpg');
    background-size: cover;
    background-position: center;
    border-radius: 20px;
    padding: 25px;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
">
    <h3 style="text-align:center;">Anxiety Assessment</h3>
    <h2 style="text-align:center; color:{anx_info['color']};">
        {anx_info['level']} (GAD-7 Score: {gad_total})
    </h2>
</div>
"""

col1, col2 = st.columns(2)
with col1: st.markdown(dep_card, unsafe_allow_html=True)
with col2: st.markdown(anx_card, unsafe_allow_html=True)

# ==============================================
# SHAP EXPLANATIONS
# ==============================================
st.markdown("---")
st.markdown("### Understanding Your Results")
st.markdown("*These charts show which factors had the most influence on your assessment:*")

tab1, tab2 = st.tabs(["Depression Factors", "Anxiety Factors"])

# --- Depression ---
with tab1:
    try:
        sel_pipe = pipelines[best_dep_model]
        pre = sel_pipe.named_steps['preprocessor']
        clf = sel_pipe.named_steps['clf']

        X_trans = pre.transform(user_df)
        if hasattr(X_trans, "toarray"): X_trans = X_trans.toarray()

        try:
            feature_names = pre.get_feature_names_out()
        except:
            feature_names = [f"Feature_{i}" for i in range(X_trans.shape[1])]

        base_model = clf.estimators_[0] if hasattr(clf, "estimators_") else clf
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_values = np.abs(shap_values).mean(axis=0)

        mean_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_shap)[-10:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        top_vals = mean_shap[top_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_features, top_vals, color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features))))
        ax.set_title("Top 10 Factors Influencing Depression", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not generate explanation: {e}")

# --- Anxiety ---
with tab2:
    try:
        sel_pipe = pipelines[best_anx_model]
        pre = sel_pipe.named_steps['preprocessor']
        clf = sel_pipe.named_steps['clf']

        X_trans = pre.transform(user_df)
        if hasattr(X_trans, "toarray"): X_trans = X_trans.toarray()

        try:
            feature_names = pre.get_feature_names_out()
        except:
            feature_names = [f"Feature_{i}" for i in range(X_trans.shape[1])]

        base_model = clf.estimators_[1] if hasattr(clf, "estimators_") else clf
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_values = np.abs(shap_values).mean(axis=0)

        mean_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_shap)[-10:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        top_vals = mean_shap[top_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_features, top_vals, color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features))))
        ax.set_title("Top 10 Factors Influencing Anxiety", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not generate explanation: {e}")

st.info("**How to read the charts:** Longer bars = stronger influence on your screening result.")

# ==============================================
# FOOTER
# ==============================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed for Kenyan High School Mental Health Screening • © 2025</p>",
    unsafe_allow_html=True
)
