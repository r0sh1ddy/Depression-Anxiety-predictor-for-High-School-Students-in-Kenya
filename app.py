import streamlit as st
import pandas as pd
import pickle, os, numpy as np
import shap, matplotlib.pyplot as plt
import plotly.graph_objects as go

# ----------------------------------------------------------------------
#  Page config & CSS
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AdolecentMind",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="brain"
)

st.markdown("""
<style>
    .main-header {font-size:2.5rem;font-weight:bold;color:#1f77b4;text-align:center;margin-bottom:1rem;}
    .sub-header {font-size:1.2rem;color:#555;text-align:center;margin-bottom:2rem;}
    .score-card {padding:2rem;border-radius:15px;text-align:center;margin:1rem 0;
                 box-shadow:0 4px 6px rgba(0,0,0,0.1);}
    .score-number {font-size:4rem;font-weight:bold;margin:0.5rem 0;}
    .score-label {font-size:1.2rem;font-weight:600;}
    .best-model-card {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                      padding:1rem;border-radius:10px;color:white;margin:0.5rem 0;}
    .stButton>button {width:100%;background-color:#1f77b4;color:white;
                      font-size:1.2rem;padding:0.75rem;border-radius:10px;
                      border:none;font-weight:bold;}
    .stButton>button:hover {background-color:#155a8a;transform:scale(1.02);}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
#  Header
# ----------------------------------------------------------------------
BASE = os.path.dirname(__file__)
LOGO_PATH = os.path.join(BASE, "images", "logo.png")

if os.path.exists(LOGO_PATH):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(LOGO_PATH, use_container_width=True)
else:
    st.markdown('<div class="main-header">Mental Health Screening Tool</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-header">Depression & Anxiety Screening for Kenyan High School Students</div>',
            unsafe_allow_html=True)

# ----------------------------------------------------------------------
#  Load models
# ----------------------------------------------------------------------
PIPELINE_FILE = os.path.join(BASE, "trained_pipelines.pkl")
METRICS_FILE  = os.path.join(BASE, "model_metrics.pkl")

pipelines = {}
model_metrics = {}

if os.path.exists(PIPELINE_FILE):
    with open(PIPELINE_FILE, "rb") as f:
        pipelines = pickle.load(f)

if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "rb") as f:
        model_metrics = pickle.load(f)

# ----------------------------------------------------------------------
#  Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("Settings")
    selection_mode = st.radio(
        "Model Selection:",
        ["Auto-Select Best", "Manual Selection", "View All Models"]
    )

    if selection_mode == "Manual Selection":
        st.markdown("### Choose Models")
        manual_dep_model = st.selectbox(
            "Depression Model:",
            list(pipelines.keys()) if pipelines else ["No models available"]
        )
        manual_anx_model = st.selectbox(
            "Anxiety Model:",
            list(pipelines.keys()) if pipelines else ["No models available"]
        )

    st.markdown("---")

    if model_metrics:
        st.markdown("### Best Models by Recall")
        best_dep_recall = -1
        best_dep_model_info = None
        for name, m in model_metrics.items():
            r = m.get('test_recall_per_target', {}).get('Is_Depressed', 0)
            if r > best_dep_recall:
                best_dep_recall = r
                best_dep_model_info = (name, m)

        best_anx_recall = -1
        best_anx_model_info = None
        for name, m in model_metrics.items():
            r = m.get('test_recall_per_target', {}).get('Has_anxiety', 0)
            if r > best_anx_recall:
                best_anx_recall = r
                best_anx_model_info = (name, m)

        if best_dep_model_info:
            n, m = best_dep_model_info
            st.markdown(f"#### Depression")
            st.markdown(f"""
            <div class="best-model-card">
                <div style="font-size:0.9rem;margin-bottom:0.5rem;">Selected Model</div>
                <div style="font-size:1.2rem;font-weight:bold;margin-bottom:0.5rem;">{n}</div>
                <div style="display:flex;justify-content:space-around;">
                    <div><div style="font-size:0.8rem;opacity:0.9;">Recall</div>
                         <div style="font-size:1.5rem;font-weight:bold;">{m['test_recall_per_target']['Is_Depressed']:.1%}</div></div>
                    <div><div style="font-size:0.8rem;opacity:0.9;">Accuracy</div>
                         <div style="font-size:1.5rem;font-weight:bold;">{m['test_accuracy_per_target']['Is_Depressed']:.1%}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if best_anx_model_info:
            n, m = best_anx_model_info
            st.markdown(f"#### Anxiety")
            st.markdown(f"""
            <div class="best-model-card">
                <div style="font-size:0.9rem;margin-bottom:0.5rem;">Selected Model</div>
                <div style="font-size:1.2rem;font-weight:bold;margin-bottom:0.5rem;">{n}</div>
                <div style="display:flex;justify-content:space-around;">
                    <div><div style="font-size:0.8rem;opacity:0.9;">Recall</div>
                         <div style="font-size:1.5rem;font-weight:bold;">{m['test_recall_per_target']['Has_anxiety']:.1%}</div></div>
                    <div><div style="font-size:0.8rem;opacity:0.9;">Accuracy</div>
                         <div style="font-size:1.5rem;font-weight:bold;">{m['test_accuracy_per_target']['Has_anxiety']:.1%}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("All Models Comparison", expanded=False):
        if model_metrics:
            view_metric = st.selectbox("Metric:", ["Recall", "Accuracy"], key="metric")
            key = 'test_recall_per_target' if view_metric == "Recall" else 'test_accuracy_per_target'
            dep, anx, names = [], [], []
            for n, m in model_metrics.items():
                if key in m:
                    names.append(n)
                    dep.append(m[key].get('Is_Depressed', 0) * 100)
                    anx.append(m[key].get('Has_anxiety', 0) * 100)
            if names:
                fig = go.Figure(data=[
                    go.Bar(name='Depression', x=names, y=dep, marker_color='#1f77b4'),
                    go.Bar(name='Anxiety', x=names, y=anx, marker_color='#ff7f0e')
                ])
                fig.update_layout(barmode='group', height=300, title=f"{view_metric} (%)",
                                  yaxis_title=f"{view_metric} (%)", margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.info("**Recall** measures ability to identify students who may need support.")

# ----------------------------------------------------------------------
#  Form
# ----------------------------------------------------------------------
st.markdown("## Complete the Screening")

tab1, tab2, tab3 = st.tabs(["Demographics", "PHQ-8", "GAD-7"])

with tab1:
    st.markdown("### School & Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        boarding_day = st.selectbox("School Demo", ["Boarding", "Day"])
        school_type = st.selectbox("School Gender", ["Boys", "Girls", "Mixed"])
        school_demo = st.selectbox("School Type", ['Subcounty', 'Extracounty', 'County'])
        school_county = st.selectbox("County", ["Nairobi","Kiambu","Makueni","Machakos"])
        age = st.slider("Age", 12, 25, 16)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        form = st.selectbox("Form", [1,2,3,4])
        religion = st.selectbox("Religion", ["Christian", "Muslim", "Other"])
        parents_home = st.selectbox("Parents at Home", ["Both parents", "One parent", "None"])
        parents_dead = st.number_input("Deceased Parents", 0, 4, 0)
        fathers_edu = st.selectbox("Father's Education", ["None","Primary","Secondary","Tertiary","University"])
        mothers_edu = st.selectbox("Mother's Education", ["None","Primary","Secondary","Tertiary","University"])
    col3, col4, col5 = st.columns(3)
    with col3: co_curr = st.selectbox("Co-curricular", ["Yes", "No"])
    with col4: sports = st.selectbox("Sports", ["Yes", "No"])
    with col5: acad_ability = st.slider("Academic Self-Rating", 1, 5, 3)

with tab2:
    st.markdown("### PHQ-8 (Past 2 Weeks)")
    phq_qs = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself — or that you are a failure",
        "Trouble concentrating on things",
        "Moving or speaking slowly, or being fidgety"
    ]
    likert = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
    phq = {}
    for i, q in enumerate(phq_qs, 1):
        c1, c2 = st.columns([3,1])
        with c1: st.markdown(f"**{i}.** {q}")
        with c2:
            phq[f'PHQ_{i}'] = st.select_slider(f"p{i}", options=[0,1,2,3],
                                               format_func=lambda x: likert[x],
                                               label_visibility="collapsed")
        st.markdown("---")
    phq_total = sum(phq.values())
    st.markdown(f"### PHQ-8 Score: **{phq_total}** / 24")

with tab3:
    st.markdown("### GAD-7 (Past 2 Weeks)")
    gad_qs = [
        "Feeling nervous, anxious, or on edge",
        "Not being able to stop or control worrying",
        "Worrying too much about different things",
        "Trouble relaxing",
        "Being so restless that it is hard to sit still",
        "Becoming easily annoyed or irritable",
        "Feeling afraid as if something awful might happen"
    ]
    gad = {}
    for i, q in enumerate(gad_qs, 1):
        c1, c2 = st.columns([3,1])
        with c1: st.markdown(f"**{i}.** {q}")
        with c2:
            gad[f'GAD_{i}'] = st.select_slider(f"g{i}", options=[0,1,2,3],
                                               format_func=lambda x: likert[x],
                                               label_visibility="collapsed")
        st.markdown("---")
    gad_total = sum(gad.values())
    st.markdown(f"### GAD-7 Score: **{gad_total}** / 21")

# ----------------------------------------------------------------------
#  Submit
# ----------------------------------------------------------------------
st.markdown("---")
_, c, _ = st.columns([1,2,1])
with c:
    submitted = st.button("Run Screening", use_container_width=True)

# ----------------------------------------------------------------------
#  SHAP Function
# ----------------------------------------------------------------------
def generate_shap_plot(pipe, user_df, target_idx, title):
    try:
        pre = pipe.named_steps['preprocessor']
        clf = pipe.named_steps['clf']
        X = pre.transform(user_df)
        if hasattr(X, "toarray"): X = X.toarray()
        try:
            feat = pre.get_feature_names_out()
        except:
            feat = [f"f{i}" for i in range(X.shape[1])]

        base = clf.estimators_[target_idx] if hasattr(clf, "estimators_") else clf

        if "Logistic" in type(base).__name__:
            expl = shap.LinearExplainer(base, X)
            sv = expl.shap_values(X)
        else:
            expl = shap.TreeExplainer(base)
            sv = expl.shap_values(X)
            if isinstance(sv, list) and len(sv) == 2:
                sv = sv[1]

        if len(sv.shape) > 1 and sv.shape[0] == 1:
            sv = sv.flatten()
        mean_abs = np.abs(sv).mean(axis=0) if len(sv.shape) > 1 else np.abs(sv)
        top_i = np.argsort(mean_abs)[-10:][::-1]
        top_f = [feat[i] for i in top_i]
        top_v = mean_abs[top_i]

        fig, ax = plt.subplots(figsize=(10,6))
        bars = ax.barh(range(len(top_f)), top_v, color=plt.cm.RdYlGn_r(np.linspace(0.2,0.8,10)))
        ax.set_yticks(range(len(top_f)))
        ax.set_yticklabels(top_f, fontsize=10)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(title, fontweight="bold")
        ax.invert_yaxis()
        for b, v in zip(bars, top_v):
            ax.text(b.get_width()+0.001, b.get_y()+b.get_height()/2, f"{v:.3f}",
                    va="center", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Longer bars = greater influence on model output.")
        return True
    except Exception as e:
        st.warning(f"SHAP failed: {e}")
        return False

# ----------------------------------------------------------------------
#  Process Submission
# ----------------------------------------------------------------------
if submitted:
    with st.spinner("Processing..."):
        # --- Clean input ---
        edu_map = {"None":0, "Primary":1, "Secondary":2, "Tertiary":3, "University":4}
        input_data = {
            "Boarding_day": boarding_day,
            "School_type": school_type,
            "School_Demographics": school_demo,
            "School_County": school_county,
            "Age": int(age),
            "Gender": 1 if gender == "Male" else 2,
            "Form": int(form),
            "Religion": 1 if religion == "Christian" else 2 if religion == "Muslim" else 3,
            "Parents_Home": {"None":0, "One parent":1, "Both parents":2}.get(parents_home, 0),
            "Parents_Dead": int(parents_dead),
            "Fathers_Education": edu_map.get(fathers_edu, 0),
            "Mothers_Education": edu_map.get(mothers_edu, 0),
            "Co_Curricular": 1 if co_curr == "Yes" else 0,
            "Sports": 1 if sports == "Yes" else 0,
            "Percieved_Academic_Abilities": int(acad_ability)
        }
        input_data.update(phq)
        input_data.update(gad)
        user_df = pd.DataFrame([input_data])

        # --- Predictions ---
        all_preds = {}
        for name, pipe in pipelines.items():
            try:
                p = pipe.predict(user_df)[0]
                all_preds[name] = {'dep': p[0], 'anx': p[1] if len(p) > 1 else None}
            except:
                all_preds[name] = {'dep': None, 'anx': None}

        # --- Select models ---
        if selection_mode == "Manual Selection":
            best_dep_model = manual_dep_model
            best_anx_model = manual_anx_model
        else:
            best_dep_model = max(model_metrics.items(),
                                 key=lambda x: x[1].get('test_recall_per_target',{}).get('Is_Depressed',0))[0] \
                             if model_metrics else None
            best_anx_model = max(model_metrics.items(),
                                 key=lambda x: x[1].get('test_recall_per_target',{}).get('Has_anxiety',0))[0] \
                             if model_metrics else None

        dep_pred = all_preds.get(best_dep_model, {}).get('dep', 'N/A')
        anx_pred = all_preds.get(best_anx_model, {}).get('anx', 'N/A')

        dep_rec = model_metrics.get(best_dep_model, {}).get('test_recall_per_target', {}).get('Is_Depressed', 0)
        dep_acc = model_metrics.get(best_dep_model, {}).get('test_accuracy_per_target', {}).get('Is_Depressed', 0)
        anx_rec = model_metrics.get(best_anx_model, {}).get('test_recall_per_target', {}).get('Has_anxiety', 0)
        anx_acc = model_metrics.get(best_anx_model, {}).get('test_accuracy_per_target', {}).get('Has_anxiety', 0)

        # --- Severity categories (standardized) ---
        dep_cat = "Minimal" if phq_total < 5 else "Mild" if phq_total < 10 else "Moderate" if phq_total < 15 else "Moderately Severe" if phq_total < 20 else "Severe"
        anx_cat = "Minimal" if gad_total < 5 else "Mild" if gad_total < 10 else "Moderate" if gad_total < 15 else "Severe"

        # --- Results --
        st.markdown("---")
        st.markdown("## Screening Results")

        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Depression Model:** {best_dep_model or 'N/A'}")
            if best_dep_model:
                st.caption(f"Recall: {dep_rec:.1%} | Accuracy: {dep_acc:.1%}")
        with c2:
            st.info(f"**Anxiety Model:** {best_anx_model or 'N/A'}")
            if best_anx_model:
                st.caption(f"Recall: {anx_rec:.1%} | Accuracy: {anx_acc:.1%}")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="score-card" style="border-left:5px solid #1f77b4">
                <h3 style="margin:0;color:#1f77b4">PHQ-8</h3>
                <div class="score-number">{phq_total}<span style="font-size:2rem;color:#666">/24</span></div>
                <div class="score-label">{dep_cat}</div>
            </div>
            """, unsafe_allow_html=True)
            if best_dep_model:
                st.markdown(f"**Model Prediction:** `{dep_pred}`")

        with col2:
            st.markdown(f"""
            <div class="score-card" style="border-left:5px solid #ff7f0e">
                <h3 style="margin:0;color:#ff7f0e">GAD-7</h3>
                <div class="score-number">{gad_total}<span style="font-size:2rem;color:#666">/21</span></div>
                <div class="score-label">{anx_cat}</div>
            </div>
            """, unsafe_allow_html=True)
            if best_anx_model:
                st.markdown(f"**Model Prediction:** `{anx_pred}`")

        # --- SHAP ---
        st.markdown("---")
        st.markdown("### Model Explanations")
        tab_d, tab_a = st.tabs(["Depression", "Anxiety"])
        with tab_d:
            if best_dep_model and pipelines.get(best_dep_model):
                generate_shap_plot(pipelines[best_dep_model], user_df,
                                   target_idx=0, title="Depression Risk Factors")
            else:
                st.info("No model available.")
        with tab_a:
            if best_anx_model and pipelines.get(best_anx_model):
                generate_shap_plot(pipelines[best_anx_model], user_df,
                                   target_idx=0 if best_dep_model != best_anx_model else 1,
                                   title="Anxiety Risk Factors")
            else:
                st.info("No model available.")

        # --- Crisis Alert ---
        if phq_total >= 15 or gad_total >= 15:
            st.markdown("---")
            st.error("### High Score Detected")
            st.markdown("""
            **Immediate support is recommended.**
            - **Kenya Red Cross:** 1199  
            - **Befrienders Kenya:** +254 722 178 177  
            - **School Counselor**
            """)

        # --- Download ---
        st.markdown("---")
        report = f"""
SCREENING RESULTS
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

        # --- Disclaimer ---
        st.markdown("---")
        st.warning("""
        **SCREENING ONLY**  
        This tool uses PHQ-8 and GAD-7 for screening.  
        Results are **not a diagnosis**.  
        High scores indicate need for professional evaluation.
        """)