import streamlit as st
import pandas as pd
import pickle, os, numpy as np
import shap, matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import recall_score, accuracy_score
from PIL import Image
import base64

# ----------------------------------------------------------------------
#  Page config & Session State
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AdolecentMind",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="brain"
)

# Session state init
for key in ['submitted', 'background_style', 'custom_bg_color', 'bg_image', 
            'dep_card_color', 'anx_card_color']:
    if key not in st.session_state:
        st.session_state[key] = {
            'background_style': 'default',
            'custom_bg_color': '#f0f2f6',
            'bg_image': None,
            'dep_card_color': '#e3f2fd',
            'anx_card_color': '#fff3e0'
        }[key]

# ----------------------------------------------------------------------
#  Background CSS (Dynamic)
# ----------------------------------------------------------------------
def get_background_css():
    if st.session_state.background_style == "custom_color":
        return f"""
        <style>
            .stApp {{ background: {st.session_state.custom_bg_color}; background-attachment: fixed; }}
        </style>
        """
    elif st.session_state.background_style == "gradient_blue":
        return """
        <style>
            .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); background-attachment: fixed; }
        </style>
        """
    elif st.session_state.background_style == "gradient_green":
        return """
        <style>
            .stApp { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); background-attachment: fixed; }
        </style>
        """
    elif st.session_state.background_style == "image" and st.session_state.bg_image:
        return f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{st.session_state.bg_image}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
        </style>
        """
    else:
        return "<style>.stApp { background: #f0f2f6; }</style>"

st.markdown(get_background_css(), unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size:2.5rem;font-weight:bold;color:#1f77b4;text-align:center;margin-bottom:1rem;}
    .sub-header {font-size:1.2rem;color:#555;text-align:center;margin-bottom:2rem;}
    .score-card {padding:2rem;border-radius:15px;text-align:center;margin:1rem 0;
                 box-shadow:0 4px 6px rgba(0,0,0,0.1);transition:0.3s;}
    .score-number {font-size:4rem;font-weight:bold;margin:0.5rem 0;}
    .score-label {font-size:1.2rem;font-weight:600;}
    .best-model-card {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                      padding:1rem;border-radius:10px;color:white;margin:0.5rem 0;}
    .stButton>button {width:100%;background-color:#1f77b4;color:white;
                      font-size:1.2rem;padding:0.75rem;border-radius:10px;
                      border:none;font-weight:bold;}
    .stButton>button:hover {background-color:#155a8a;transform:scale(1.02);}
    .restart-button>button {background-color:#e74c3c !important;}
    .restart-button>button:hover {background-color:#c0392b !important;}
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
    st.markdown('<div class="main-header">AdolecentMind</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-header">Depression & Anxiety Screening for Kenyan High School Students</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
#  Load Models
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
#  Sidebar - Background + Card Styling
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("Appearance")

    # Background
    st.markdown("### Background Style")
    bg_choice = st.selectbox(
        "Choose Background",
        ["Default", "Custom Color", "Gradient Blue", "Gradient Green", "Upload Image"],
        index=["default", "custom_color", "gradient_blue", "gradient_green", "image"].index(st.session_state.background_style)
    )
    if bg_choice == "Custom Color":
        color = st.color_picker("Pick Color", st.session_state.custom_bg_color)
        st.session_state.custom_bg_color = color
    elif bg_choice == "Upload Image":
        uploaded = st.file_uploader("Upload Background (JPG/PNG)", type=["png", "jpg", "jpeg"])
        if uploaded:
            img = Image.open(uploaded)
            import io
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            st.session_state.bg_image = base64.b64encode(buffered.getvalue()).decode()
            st.success("Background image applied!")
    st.session_state.background_style = {
        "Default": "default", "Custom Color": "custom_color",
        "Gradient Blue": "gradient_blue", "Gradient Green": "gradient_green",
        "Upload Image": "image"
    }[bg_choice]

    # Card Colors
    st.markdown("### Card Backgrounds")
    dep_col = st.color_picker("Depression Card", st.session_state.dep_card_color, key="dep_color")
    anx_col = st.color_picker("Anxiety Card", st.session_state.anx_card_color, key="anx_color")
    st.session_state.dep_card_color = dep_col
    st.session_state.anx_card_color = anx_col

    st.markdown("---")

    # Model Selection
    selection_mode = st.radio("Model Selection:", ["Auto-Select Best", "Manual Selection", "View All Models"])
    if selection_mode == "Manual Selection":
        st.markdown("### Choose Models")
        manual_dep_model = st.selectbox("Depression Model:", list(pipelines.keys()) if pipelines else ["No models"])
        manual_anx_model = st.selectbox("Anxiety Model:", list(pipelines.keys()) if pipelines else ["No models"])
        st.session_state.manual_dep_model = manual_dep_model
        st.session_state.manual_anx_model = manual_anx_model

    # Best Models
    if model_metrics:
        st.markdown("### Best Models by Recall")
        best_dep = max(model_metrics.items(), key=lambda x: x[1].get('test_recall_per_target',{}).get('Is_Depressed',0))[0] if model_metrics else None
        best_anx = max(model_metrics.items(), key=lambda x: x[1].get('test_recall_per_target',{}).get('Has_anxiety',0))[0] if model_metrics else None

        if best_dep:
            m = model_metrics[best_dep]
            st.markdown(f"#### Depression: **{best_dep}**")
            st.markdown(f"Recall: **{m['test_recall_per_target']['Is_Depressed']:.1%}** | Accuracy: **{m['test_accuracy_per_target']['Is_Depressed']:.1%}**")
        if best_anx:
            m = model_metrics[best_anx]
            st.markdown(f"#### Anxiety: **{best_anx}**")
            st.markdown(f"Recall: **{m['test_recall_per_target']['Has_anxiety']:.1%}** | Accuracy: **{m['test_accuracy_per_target']['Has_anxiety']:.1%}**")

    st.markdown("---")
    st.info("**Recall** = ability to detect students needing help.")

# ----------------------------------------------------------------------
#  Restart Button
# ----------------------------------------------------------------------
if st.session_state.submitted:
    _, c, _ = st.columns([1,1,1])
    with c:
        if st.button("Restart Screening", key="restart"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# ----------------------------------------------------------------------
#  Form
# ----------------------------------------------------------------------
if not st.session_state.submitted:
    st.markdown("## Complete the Screening")

    tab1, tab2, tab3 = st.tabs(["Demographics", "PHQ-8", "GAD-7"])

    with tab1:
        st.markdown("### School & Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            boarding_day = st.selectbox("School Demo", ["Boarding", "Day"], key="boarding_day")
            school_type = st.selectbox("School Gender", ["Boys", "Girls", "Mixed"], key="school_type")
            school_demo = st.selectbox("School Type", ['Subcounty', 'Extracounty', 'County'], key="school_demo")
            school_county = st.selectbox("County", ["Nairobi","Kiambu","Makueni","Machakos"], key="school_county")
            age = st.slider("Age", 12, 25, 16, key="age")
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        with col2:
            form = st.selectbox("Form", [1,2,3,4], key="form")
            religion = st.selectbox("Religion", ["Christian", "Muslim", "Other"], key="religion")
            parents_home = st.selectbox("Parents at Home", ["Both parents", "One parent", "None"], key="parents_home")
            parents_dead = st.number_input("Deceased Parents", 0, 4, 0, key="parents_dead")
            fathers_edu = st.selectbox("Father's Education", ["None","Primary","Secondary","Tertiary","University"], key="fathers_edu")
            mothers_edu = st.selectbox("Mother's Education", ["None","Primary","Secondary","Tertiary","University"], key="mothers_edu")
        col3, col4, col5 = st.columns(3)
        with col3: co_curr = st.selectbox("Co-curricular", ["Yes", "No"], key="co_curr")
        with col4: sports = st.selectbox("Sports", ["Yes", "No"], key="sports")
        with col5: acad_ability = st.slider("Academic Self-Rating", 1, 5, 3, key="acad_ability")

    with tab2:
        st.markdown("### PHQ-8 Depression Assessment")
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
                phq[f'PHQ_{i}'] = st.select_slider(f"p{i}", options=[0,1,2,3], format_func=lambda x: likert[x], label_visibility="collapsed", key=f"phq_{i}")
            st.markdown("---")
        phq_total = sum(phq.values())
        st.markdown(f"### PHQ-8 Score: **{phq_total}** / 24")

    with tab3:
        st.markdown("### GAD-7 Anxiety Assessment")
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
                gad[f'GAD_{i}'] = st.select_slider(f"g{i}", options=[0,1,2,3], format_func=lambda x: likert[x], label_visibility="collapsed", key=f"gad_{i}")
            st.markdown("---")
        gad_total = sum(gad.values())
        st.markdown(f"### GAD-7 Score: **{gad_total}** / 21")

    # Submit
    st.markdown("---")
    _, c, _ = st.columns([1,2,1])
    with c:
        if st.button("Run Screening", use_container_width=True):
            st.session_state.submitted = True
            st.session_state.phq_total = phq_total
            st.session_state.gad_total = gad_total
            edu_map = {"None":0, "Primary":1, "Secondary":2, "Tertiary":3, "University":4}
            st.session_state.input_data = {
                "Boarding_day": boarding_day, "School_type": school_type, "School_Demographics": school_demo,
                "School_County": school_county, "Age": int(age), "Gender": 1 if gender == "Male" else 2,
                "Form": int(form), "Religion": 1 if religion == "Christian" else 2 if religion == "Muslim" else 3,
                "Parents_Home": {"None":0, "One parent":1, "Both parents":2}.get(parents_home, 0),
                "Parents_Dead": int(parents_dead),
                "Fathers_Education": edu_map.get(fathers_edu, 0),
                "Mothers_Education": edu_map.get(mothers_edu, 0),
                "Co_Curricular": 1 if co_curr == "Yes" else 0,
                "Sports": 1 if sports == "Yes" else 0,
                "Percieved_Academic_Abilities": int(acad_ability)
            }
            st.session_state.input_data.update(phq)
            st.session_state.input_data.update(gad)
            st.rerun()

# ----------------------------------------------------------------------
#  Results
# ----------------------------------------------------------------------
else:
    with st.spinner("Analyzing..."):
        user_df = pd.DataFrame([st.session_state.input_data])
        all_preds = {}
        all_probas = {}

        for name, pipe in pipelines.items():
            try:
                pred = pipe.predict(user_df)[0]
                try:
                    proba = pipe.predict_proba(user_df)[0]
                    dep_prob = proba[0][1] if len(proba) > 0 and len(proba[0]) > 1 else None
                    anx_prob = proba[1][1] if len(proba) > 1 and len(proba[1]) > 1 else None
                except:
                    dep_prob = anx_prob = None
                all_preds[name] = {'dep': pred[0], 'anx': pred[1] if len(pred) > 1 else None}
                all_probas[name] = {'dep': dep_prob, 'anx': anx_prob}
            except Exception as e:
                all_preds[name] = {'dep': None, 'anx': None}
                all_probas[name] = {'dep': None, 'anx': None}

        # Select model
        if selection_mode == "Manual Selection":
            best_dep_model = st.session_state.get("manual_dep_model")
            best_anx_model = st.session_state.get("manual_anx_model")
        else:
            best_dep_model = max(model_metrics.items(), key=lambda x: x[1].get('test_recall_per_target',{}).get('Is_Depressed',0))[0] if model_metrics else None
            best_anx_model = max(model_metrics.items(), key=lambda x: x[1].get('test_recall_per_target',{}).get('Has_anxiety',0))[0] if model_metrics else None

        dep_pred = all_preds.get(best_dep_model, {}).get('dep')
        anx_pred = all_preds.get(best_anx_model, {}).get('anx')
        dep_proba = all_probas.get(best_dep_model, {}).get('dep')
        anx_proba = all_probas.get(best_anx_model, {}).get('anx')

        # Severity
        phq_total = st.session_state.phq_total
        gad_total = st.session_state.gad_total
        dep_cat = ["Minimal","Mild","Moderate","Moderately Severe","Severe"][min(phq_total//5, 4)]
        anx_cat = ["Minimal","Mild","Moderate","Severe"][min(gad_total//5, 3)]

        # Live metrics
        threshold = 0.5
        y_true_dep = [1 if phq_total >= 10 else 0]
        y_true_anx = [1 if gad_total >= 10 else 0]
        y_pred_dep = [1 if (dep_proba is not None and dep_proba >= threshold) else 0]
        y_pred_anx = [1 if (anx_proba is not None and anx_proba >= threshold) else 0]

        live_recall_dep = recall_score(y_true_dep, y_pred_dep, zero_division=0)
        live_acc_dep = accuracy_score(y_true_dep, y_pred_dep)
        live_recall_anx = recall_score(y_true_anx, y_pred_anx, zero_division=0)
        live_acc_anx = accuracy_score(y_true_anx, y_pred_anx)

    st.markdown("---")
    st.markdown("## Screening Results")

    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**Depression Model:** {best_dep_model or 'N/A'}")
        st.caption(f"Live Recall: **{live_recall_dep:.1%}** | Accuracy: **{live_acc_dep:.1%}**")
    with c2:
        st.info(f"**Anxiety Model:** {best_anx_model or 'N/A'}")
        st.caption(f"Live Recall: **{live_recall_anx:.1%}** | Accuracy: **{live_acc_anx:.1%}**")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="score-card" style="background:{st.session_state.dep_card_color};border-left:5px solid #1f77b4">
            <h3 style="margin:0;color:#1f77b4">PHQ-8 Depression Assessment</h3>
            <div class="score-number">{phq_total}<span style="font-size:2rem;color:#666">/24</span></div>
            <div class="score-label">{dep_cat}</div>
        </div>
        """, unsafe_allow_html=True)
        txt = f"**Prediction:** `{dep_pred}`"
        if dep_proba is not None:
            txt += f" | **Confidence: {dep_proba:.1%}**"
        st.markdown(txt)

    with col2:
        st.markdown(f"""
        <div class="score-card" style="background:{st.session_state.anx_card_color};border-left:5px solid #ff7f0e">
            <h3 style="margin:0;color:#ff7f0e">GAD-7 Anxiety Assessment</h3>
            <div class="score-number">{gad_total}<span style="font-size:2rem;color:#666">/21</span></div>
            <div class="score-label">{anx_cat}</div>
        </div>
        """, unsafe_allow_html=True)
        txt = f"**Prediction:** `{anx_pred}`"
        if anx_proba is not None:
            txt += f" | **Confidence: {anx_proba:.1%}**"
        st.markdown(txt)

    # SHAP
    st.markdown("---")
    st.markdown("### Risk Factor Insights")
    tab_d, tab_a = st.tabs(["Depression", "Anxiety"])

    def generate_shap_plot(pipe, df, idx, title):
        try:
            pre = pipe.named_steps['preprocessor']
            clf = pipe.named_steps['clf']
            X = pre.transform(df)
            if hasattr(X, "toarray"): X = X.toarray()
            feat = pre.get_feature_names_out() if hasattr(pre, 'get_feature_names_out') else [f"f{i}" for i in range(X.shape[1])]
            base = clf.estimators_[idx] if hasattr(clf, "estimators_") else clf

            if "Logistic" in type(base).__name__:
                expl = shap.LinearExplainer(base, X)
                sv = expl.shap_values(X)
            else:
                expl = shap.TreeExplainer(base)
                sv = expl.shap_values(X)
                sv = sv[1] if isinstance(sv, list) and len(sv) > 1 else sv

            if sv.ndim > 1: sv = sv.flatten()
            mean_abs = np.abs(sv).mean(axis=0) if sv.ndim > 1 else np.abs(sv)
            top_i = np.argsort(mean_abs)[-10:][::-1]
            top_i = [i for i in top_i if i < len(feat)]
            top_f = [feat[i] for i in top_i]
            top_v = mean_abs[top_i]

            fig, ax = plt.subplots(figsize=(10,6))
            ax.barh(range(len(top_f)), top_v, color=plt.cm.RdYlGn_r(np.linspace(0.2,0.8,len(top_f))))
            ax.set_yticks(range(len(top_f)))
            ax.set_yticklabels(top_f, fontsize=10)
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title(title)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Longer bars = stronger influence")
        except Exception as e:
            st.warning(f"SHAP failed: {str(e)[:100]}")

    with tab_d:
        if best_dep_model and pipelines.get(best_dep_model):
            generate_shap_plot(pipelines[best_dep_model], user_df, 0, "Top Depression Risk Factors")
        else: st.info("No model")

    with tab_a:
        if best_anx_model and pipelines.get(best_anx_model):
            idx = 0 if best_dep_model != best_anx_model else 1
            generate_shap_plot(pipelines[best_anx_model], user_df, idx, "Top Anxiety Risk Factors")
        else: st.info("No model")

    # Crisis
    if phq_total >= 15 or gad_total >= 15:
        st.error("### High Risk Detected — Seek Help Now")
        st.markdown("- **Kenya Red Cross:** 1199  \n- **Befrienders Kenya:** +254 722 178 177  \n- **School Counselor**")

    # Report
    st.markdown("---")
    dep_conf = f" (Conf: {dep_proba:.1%})" if dep_proba else ""
    anx_conf = f" (Conf: {anx_proba:.1%})" if anx_proba else ""
    report = f"""
SCREENING RESULTS
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

PHQ-8: {phq_total}/24 → {dep_cat}
GAD-7: {gad_total}/21 → {anx_cat}

Depression Model: {best_dep_model} → {dep_pred}{dep_conf}
Anxiety Model: {best_anx_model} → {anx_pred}{anx_conf}

Live Performance:
  Depression → Recall: {live_recall_dep:.1%}, Accuracy: {live_acc_dep:.1%}
  Anxiety → Recall: {live_recall_anx:.1%}, Accuracy: {live_acc_anx:.1%}

SCREENING ONLY — NOT A DIAGNOSIS
    """
    st.download_button("Download Report", report, f"screening_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt", "text/plain")

    st.warning("**SCREENING TOOL ONLY** — Consult a professional for diagnosis.")