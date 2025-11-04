import streamlit as st
import pandas as pd
import pickle, os, numpy as np
import shap, matplotlib.pyplot as plt
import base64

# ----------------------------------------------------------------------
#  PATHS & LOAD MODELS
# ----------------------------------------------------------------------
BASE = os.path.dirname(__file__)
PIPELINE_FILE = os.path.join(BASE, "trained_pipelines.pkl")
METRICS_FILE = os.path.join(BASE, "model_metrics.pkl")

@st.cache_resource
def load_models():
    pipelines = {}
    model_metrics = {}
    if os.path.exists(PIPELINE_FILE):
        with open(PIPELINE_FILE, "rb") as f:
            pipelines = pickle.load(f)
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "rb") as f:
            model_metrics = pickle.load(f)
    return pipelines, model_metrics

pipelines, model_metrics = load_models()

# ----------------------------------------------------------------------
#  BACKGROUND IMAGE
# ----------------------------------------------------------------------
def set_background():
    bg_path = os.path.join(BASE, "images", "background.jpg")
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
            .stApp {{
                background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.7)),
                            url("data:image/jpg;base64,{b64}") no-repeat center center fixed;
                background-size: cover;
            }}
            .score-card {{background: rgba(255,255,255,0.94); backdrop-filter: blur(12px); border-radius: 15px; padding: 1.8rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}}
            .stButton>button {{background:#1f77b4; color:white; font-weight:bold; border-radius:10px; padding:0.75rem;}}
            .stButton>button:hover {{background:#155a8a; transform:scale(1.02);}}
            .stError, .stWarning {{border-radius:10px;}}
            .progress-text {{font-size:0.9rem; color:#666; margin-top:0.5rem;}}
        </style>
        """, unsafe_allow_html=True)

set_background()

# ----------------------------------------------------------------------
#  PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Mental Health Screening", page_icon="magnifying-glass")

# ----------------------------------------------------------------------
#  SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")

    selection_mode = st.radio("Model Mode", ["Auto", "Manual"], help="Auto = best recall")
    if selection_mode == "Manual" and pipelines:
        manual_dep_model = st.selectbox("Depression", list(pipelines.keys()))
        manual_anx_model = st.selectbox("Anxiety", list(pipelines.keys()))
    else:
        manual_dep_model = manual_anx_model = None

    st.markdown("---")
    if model_metrics:
        st.markdown("**Best Models**")
        best_dep = max(model_metrics.items(), key=lambda x: x[1].get('test_recall_per_target',{}).get('Is_Depressed',0))
        best_anx = max(model_metrics.items(), key=lambda x: x[1].get('test_recall_per_target',{}).get('Has_anxiety',0))
        for (name, m), label in [(best_dep, "Dep"), (best_anx, "Anx")]:
            r = m['test_recall_per_target'].get('Is_Depressed' if label=="Dep" else 'Has_anxiety', 0)
            st.markdown(f"**{label}:** `{name}` — Recall: `{r:.1%}`")

    st.markdown("---")
    st.caption("Screening Tool • No Diagnosis")

# ----------------------------------------------------------------------
#  FORM
# ----------------------------------------------------------------------
st.markdown("## Mental Health Screening")

# Anonymous ID
st.markdown("### Optional: Anonymous ID")
screening_id = st.text_input("Student ID (e.g. S12345)", placeholder="Leave blank if preferred", help="For follow-up only. No names.")

tab1, tab2, tab3 = st.tabs(["Info", "PHQ-8", "GAD-7"])

# 27 RAW COLUMNS
RAW_COLUMNS = [
    'Age', 'Gender', 'Form', 'Religion', 'Parents_Home', 'Parents_Dead',
    'Fathers_Education', 'Mothers_Education', 'Co_Curricular', 'Sports',
    'Percieved_Academic_Abilities',
    'Boarding_day', 'School_type', 'School_Demographics', 'School_County',
    'PHQ_1', 'PHQ_2', 'PHQ_3', 'PHQ_4', 'PHQ_5', 'PHQ_6', 'PHQ_7', 'PHQ_8',
    'GAD_1', 'GAD_2', 'GAD_3', 'GAD_4', 'GAD_5', 'GAD_6', 'GAD_7'
]

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        boarding_day = st.selectbox("School Type", ["Boarding", "Day"])
        school_type = st.selectbox("Gender", ["Boys", "Girls", "Mixed"])
        school_demo = st.selectbox("Location", ["Urban", "Rural", "Semi-urban"])
        school_county = st.selectbox("County", ["Nairobi","Kiambu","Kisumu","Mombasa","Nakuru","Makueni","Other"])
        age = st.slider("Age", 12, 25, 16)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        form = st.selectbox("Form", [1,2,3,4])
        religion = st.selectbox("Religion", ["Christian", "Muslim", "Other"])
        parents_home = st.selectbox("Parents at Home", ["Both", "One", "None"])
        parents_dead = st.number_input("Deceased Parents", 0, 4, 0)
        fathers_edu = st.selectbox("Father Edu", ["None","Primary","Secondary","Tertiary","University"])
        mothers_edu = st.selectbox("Mother Edu", ["None","Primary","Secondary","Tertiary","University"])
    c3, c4, c5 = st.columns(3)
    with c3: co_curr = st.selectbox("Co-curr", ["Yes", "No"])
    with c4: sports = st.selectbox("Sports", ["Yes", "No"])
    with c5: acad_ability = st.slider("Academic", 1, 5, 3)

with tab2:
    phq = {}
    for i in range(1, 9):
        q = st.select_slider(f"Q{i}", options=[0,1,2,3],
                             format_func=lambda x: ["Not at all", "Several days", "More than half", "Nearly every day"][x])
        phq[f'PHQ_{i}'] = q
    phq_total = sum(phq.values())
    st.markdown(f"**PHQ-8 Score: {phq_total}/24**")

with tab3:
    gad = {}
    for i in range(1, 8):
        q = st.select_slider(f"Q{i}", options=[0,1,2,3],
                             format_func=lambda x: ["Not at all", "Several days", "More than half", "Nearly every day"][x])
        gad[f'GAD_{i}'] = q
    gad_total = sum(gad.values())
    st.markdown(f"**GAD-7 Score: {gad_total}/21**")

# Progress Bar
progress = 0
if 'age' in locals(): progress += 0.33
if 'phq_total' in locals(): progress += 0.33
if 'gad_total' in locals(): progress += 0.34
st.progress(progress)
st.markdown(f"<div class='progress-text'>{int(progress*100)}% Complete</div>", unsafe_allow_html=True)

# Submit
st.markdown("---")
submitted = st.button("Run Screening", use_container_width=True)

# ----------------------------------------------------------------------
#  SHAP PLOT (Demographics in ORANGE)
# ----------------------------------------------------------------------
def generate_shap_plot(pipe, user_df, target_idx, title):
    try:
        X = pipe.named_steps['preprocessor'].transform(user_df)
        if hasattr(X, "toarray"): X = X.toarray()
        if X.shape[1] != 27:
            st.error(f"Expected 27 features, got {X.shape[1]}")
            return False

        feat = pipe.named_steps['preprocessor'].get_feature_names_out()
        base = pipe.named_steps['clf'].estimators_[target_idx] if hasattr(pipe.named_steps['clf'], 'estimators_') else pipe.named_steps['clf']

        expl = shap.TreeExplainer(base) if "Tree" in type(base).__name__ else shap.LinearExplainer(base, X)
        sv = expl.shap_values(X)
        if isinstance(sv, list): sv = sv[1]
        if sv.ndim > 1: sv = sv.flatten()

        mean_abs = np.abs(sv).mean(axis=0)
        top_i = np.argsort(mean_abs)[-10:][::-1]
        top_f = [feat[i] for i in top_i]
        top_v = mean_abs[top_i]

        # Color: orange = demographics, blue = PHQ/GAD
        colors = ['orange' if 'cat__' in f or 'num__' in f else '#1f77b4' for f in top_f]

        fig, ax = plt.subplots(figsize=(10,6))
        bars = ax.barh(range(len(top_f)), top_v, color=colors)
        ax.set_yticks(range(len(top_f)))
        ax.set_yticklabels(top_f, fontsize=9)
        ax.set_xlabel("Impact on Prediction")
        ax.set_title(title, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption("**Orange** = Demographics | **Blue** = PHQ/GAD")
        return True
    except Exception as e:
        st.error(f"SHAP Error: {e}")
        return False

# ----------------------------------------------------------------------
#  PROCESS SUBMISSION
# ----------------------------------------------------------------------
if submitted:
    with st.spinner("Analyzing..."):
        edu_map = {"None":0, "Primary":1, "Secondary":2, "Tertiary":3, "University":4}
        input_data = {
            "Age": int(age), "Gender": 1 if gender == "Male" else 2, "Form": int(form),
            "Religion": 1 if religion == "Christian" else 2 if religion == "Muslim" else 3,
            "Parents_Home": {"None":0, "One parent":1, "Both parents":2}.get(parents_home, 0),
            "Parents_Dead": int(parents_dead),
            "Fathers_Education": edu_map.get(fathers_edu, 0),
            "Mothers_Education": edu_map.get(mothers_edu, 0),
            "Co_Curricular": 1 if co_curr == "Yes" else 0,
            "Sports": 1 if sports == "Yes" else 0,
            "Percieved_Academic_Abilities": int(acad_ability),
            "Boarding_day": boarding_day, "School_type": school_type,
            "School_Demographics": school_demo, "School_County": school_county
        }
        input_data.update(phq)
        input_data.update(gad)
        user_df = pd.DataFrame([input_data])[RAW_COLUMNS]

        preds = {}
        for name, pipe in pipelines.items():
            try:
                p = pipe.predict(user_df)[0]
                preds[name] = {'dep': int(p[0]), 'anx': int(p[1]) if len(p)>1 else None}
            except:
                pass

        if selection_mode == "Manual":
            best_dep_model = manual_dep_model
            best_anx_model = manual_anx_model
        else:
            best_dep_model = max(model_metrics.items(), key=lambda x: x[1].get('test_recall_per_target',{}).get('Is_Depressed',0))[0]
            best_anx_model = max(model_metrics.items(), key=lambda x: x[1].get('test_recall_per_target',{}).get('Has_anxiety',0))[0]

        dep_pred = preds.get(best_dep_model, {}).get('dep', 'N/A')
        anx_pred = preds.get(best_anx_model, {}).get('anx', 'N/A')

        dep_cat = ["Minimal","Mild","Moderate","Moderately Severe","Severe"][min(phq_total//5, 4)]
        anx_cat = ["Minimal","Mild","Moderate","Severe"][min(gad_total//5, 3)]

        st.success("Screening Complete")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="score-card">
                <h3>PHQ-8</h3>
                <div style="font-size:3rem;font-weight:bold">{phq_total}<span style="font-size:1.5rem">/24</span></div>
                <div style="font-size:1.2rem">{dep_cat}</div>
                <div><strong>Model:</strong> {best_dep_model} → <code>{dep_pred}</code></div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="score-card">
                <h3>GAD-7</h3>
                <div style="font-size:3rem;font-weight:bold">{gad_total}<span style="font-size:1.5rem">/21</span></div>
                <div style="font-size:1.2rem">{anx_cat}</div>
                <div><strong>Model:</strong> {best_anx_model} → <code>{anx_pred}</code></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Model Explanations")
        t1, t2 = st.tabs(["Depression", "Anxiety"])
        with t1:
            if pipelines.get(best_dep_model):
                generate_shap_plot(pipelines[best_dep_model], user_df, 0, "Depression Risk Factors")
        with t2:
            if pipelines.get(best_anx_model):
                generate_shap_plot(pipelines[best_anx_model], user_df, 1 if best_dep_model == best_anx_model else 0, "Anxiety Risk Factors")

        # Crisis Alert
        if phq_total >= 15 or gad_total >= 15:
            st.markdown("---")
            st.error("### High Score Detected")
            st.markdown("""
            **Immediate support is recommended.**
            - **Kenya Red Cross:** 1199  
            - **Befrienders Kenya:** +254 722 178 177  
            - **School Counselor**
            """)

        # Download
        st.markdown("---")
        report = f"""
SCREENING RESULTS
ID: {screening_id or 'Not provided'}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

PHQ-8 Score: {phq_total}/24 → {dep_cat}
GAD-7 Score: {gad_total}/21 → {anx_cat}

Model (Depression): {best_dep_model or 'N/A'} → Prediction: {dep_pred}
Model (Anxiety): {best_anx_model or 'N/A'} → Prediction: {anx_pred}

This is a screening tool only. Not a diagnosis.
        """
        st.download_button("Download Results", report,
                           file_name=f"screening_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                           mime="text/plain")

        # Disclaimer
        st.markdown("---")
        st.warning("""
        **SCREENING ONLY**  
        This tool uses PHQ-8 and GAD-7 for screening.  
        Results are **not a diagnosis**.  
        High scores indicate need for professional evaluation.
        """)

        # Retake
        if st.button("Retake Screening"):
            st.experimental_rerun()