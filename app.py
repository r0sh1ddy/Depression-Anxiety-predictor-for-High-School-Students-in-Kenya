# app.py (updated)
import streamlit as st
import pandas as pd
import pickle, os, numpy as np
import shap, matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import recall_score, accuracy_score
import base64

# ----------------------------------------------------------------------
#  Page config & Session State Initialization
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AdolescentMind",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Initialize session state
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'background_style' not in st.session_state:
    st.session_state.background_style = "default"
if 'custom_bg_color' not in st.session_state:
    st.session_state.custom_bg_color = "#f0f2f6"
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'uploaded_bg_path' not in st.session_state:
    st.session_state.uploaded_bg_path = None

# ----------------------------------------------------------------------
#  Dynamic Background CSS (supports uploaded image)
# ----------------------------------------------------------------------
def get_background_css():
    style = st.session_state.background_style
    if style == "custom_color":
        return f"""
        <style>
            .stApp {{ 
                background: {st.session_state.custom_bg_color}; 
                background-attachment: fixed;
            }}
            .score-card {{ background: rgba(255,255,255,0.9); }}
        </style>
        """
    elif style == "gradient_blue":
        return """
        <style>
            .stApp { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                background-attachment: fixed;
            }
            .score-card { background: rgba(255,255,255,0.95); }
        </style>
        """
    elif style == "gradient_green":
        return """
        <style>
            .stApp { 
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                background-attachment: fixed;
            }
            .score-card { background: rgba(255,255,255,0.95); }
        </style>
        """
    elif style == "uploaded_image" and st.session_state.uploaded_bg_path:
        try:
            with open(st.session_state.uploaded_bg_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode()
            ext = os.path.splitext(st.session_state.uploaded_bg_path)[1].lower().replace('.', '')
            mime = f"image/{'jpeg' if ext in ['jpg','jpeg'] else ext}"
            return f"""
            <style>
                .stApp {{
                    background-image: url("data:{mime};base64,{b64}");
                    background-size: cover;
                    background-position: center;
                    background-attachment: fixed;
                }}
                .score-card {{ background: rgba(255,255,255,0.9); }}
            </style>
            """
        except Exception as e:
            # fallback to default if reading fails
            return """
            <style>
                .stApp { background: #f0f2f6; }
            </style>
            """
    else:
        return """
        <style>
            .stApp { background: #f0f2f6; }
        </style>
        """

st.markdown(get_background_css(), unsafe_allow_html=True)

st.markdown("""
<style>
    .main-header {font-size:2.2rem;font-weight:bold;color:#1f77b4;text-align:center;margin-bottom:1rem;}
    .sub-header {font-size:1.05rem;color:#555;text-align:center;margin-bottom:1.2rem;}
    .score-card {padding:1.4rem;border-radius:12px;text-align:center;margin:0.6rem 0;
                 box-shadow:0 4px 6px rgba(0,0,0,0.08);background:rgba(255,255,255,0.9);}
    .score-number {font-size:3.2rem;font-weight:bold;margin:0.4rem 0;}
    .score-label {font-size:1rem;font-weight:600;}
    .best-model-card {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                      padding:0.8rem;border-radius:8px;color:white;margin:0.5rem 0;}
    .stButton>button {width:100%;background-color:#1f77b4;color:white;
                      font-size:1.02rem;padding:0.6rem;border-radius:8px;
                      border:none;font-weight:bold;}
    .stButton>button:hover {background-color:#155a8a;transform:scale(1.01);}
    .restart-button>button {background-color:#e74c3c !important;}
    .restart-button>button:hover {background-color:#c0392b !important;}
    /* Tiny styling for top alert close button */
    .alert-close {background:transparent;border:none;color:white;font-weight:bold;margin-left:16px;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Header / Logo
# ----------------------------------------------------------------------
BASE = os.path.dirname(__file__)
LOGO_PATH = os.path.join(BASE, "images", "logo.png")

if os.path.exists(LOGO_PATH):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(LOGO_PATH, use_container_width=True)
else:
    st.markdown('<div class="main-header">AdolescentMind</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-header">Depression & Anxiety Screening for Kenyan High School Students</div>',
            unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Load models (unchanged)
# ----------------------------------------------------------------------
PIPELINE_FILE = os.path.join(BASE, "trained_pipelines.pkl")
METRICS_FILE  = os.path.join(BASE, "model_metrics.pkl")

pipelines = {}
model_metrics = {}
y_true_dep = None
y_true_anx = None

if os.path.exists(PIPELINE_FILE):
    with open(PIPELINE_FILE, "rb") as f:
        pipelines = pickle.load(f)

if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "rb") as f:
        model_metrics = pickle.load(f)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("Settings")

    # Background Selector + Upload
    st.markdown("### Background Style")
    bg_options = ["Default", "Custom Color", "Gradient Blue", "Gradient Green", "Uploaded Image"]
    # Determine index safely
    mapping_to_state = {
        "Default": "default",
        "Custom Color": "custom_color",
        "Gradient Blue": "gradient_blue",
        "Gradient Green": "gradient_green",
        "Uploaded Image": "uploaded_image"
    }
    reverse_map = {v: k for k, v in mapping_to_state.items()}
    try:
        default_index = bg_options.index(reverse_map.get(st.session_state.background_style, "Default"))
    except Exception:
        default_index = 0

    bg_choice = st.selectbox(
        "Choose Background",
        bg_options,
        index=default_index
    )
    # If user chooses custom color
    if bg_choice == "Custom Color":
        color = st.color_picker("Pick Color", st.session_state.custom_bg_color)
        st.session_state.custom_bg_color = color

    # Upload button (next to background)
    st.markdown("### Upload Background Image (optional)")
    uploaded_bg = st.file_uploader("Upload image (png/jpg)", type=["png", "jpg", "jpeg"], key="bg_uploader")
    if uploaded_bg is not None:
        # save to images folder
        images_dir = os.path.join(BASE, "images")
        os.makedirs(images_dir, exist_ok=True)
        save_path = os.path.join(images_dir, f"uploaded_bg_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.{uploaded_bg.name.split('.')[-1]}")
        try:
            with open(save_path, "wb") as out:
                out.write(uploaded_bg.getbuffer())
            st.session_state.uploaded_bg_path = save_path
            st.success("Background image uploaded. Select 'Uploaded Image' in Background Style to apply it.")
            st.image(save_path, caption="Uploaded preview", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to save uploaded image: {e}")

    st.session_state.background_style = mapping_to_state.get(bg_choice, "default")

    # Model Selection
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
        # persist choices
        st.session_state.manual_dep_model = manual_dep_model
        st.session_state.manual_anx_model = manual_anx_model

    st.markdown("---")

    # Best models display (unchanged labels)
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
                <div style="font-size:0.85rem;margin-bottom:0.4rem;">Selected Model</div>
                <div style="font-size:1.05rem;font-weight:bold;margin-bottom:0.4rem;">{n}</div>
                <div style="display:flex;justify-content:space-around;">
                    <div><div style="font-size:0.75rem;opacity:0.9;">Recall</div>
                         <div style="font-size:1.1rem;font-weight:bold;">{m['test_recall_per_target']['Is_Depressed']:.1%}</div></div>
                    <div><div style="font-size:0.75rem;opacity:0.9;">Accuracy</div>
                         <div style="font-size:1.1rem;font-weight:bold;">{m['test_accuracy_per_target']['Is_Depressed']:.1%}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if best_anx_model_info:
            n, m = best_anx_model_info
            st.markdown(f"#### Anxiety")
            st.markdown(f"""
            <div class="best-model-card">
                <div style="font-size:0.85rem;margin-bottom:0.4rem;">Selected Model</div>
                <div style="font-size:1.05rem;font-weight:bold;margin-bottom:0.4rem;">{n}</div>
                <div style="display:flex;justify-content:space-around;">
                    <div><div style="font-size:0.75rem;opacity:0.9;">Recall</div>
                         <div style="font-size:1.1rem;font-weight:bold;">{m['test_recall_per_target']['Has_anxiety']:.1%}</div></div>
                    <div><div style="font-size:0.75rem;opacity:0.9;">Accuracy</div>
                         <div style="font-size:1.1rem;font-weight:bold;">{m['test_accuracy_per_target']['Has_anxiety']:.1%}</div></div>
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
                    go.Bar(name='Depression', x=names, y=dep),
                    go.Bar(name='Anxiety', x=names, y=anx)
                ])
                fig.update_layout(barmode='group', height=300, title=f"{view_metric} (%)",
                                  yaxis_title=f"{view_metric} (%)", margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.info("**Recall** measures ability to identify students who may need support.")

# ----------------------------------------------------------------------
# Restart Button
# ----------------------------------------------------------------------
if st.session_state.submitted:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Restart Screening", key="restart", help="Clear form and start over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.background_style = "default"
            st.session_state.custom_bg_color = "#f0f2f6"
            st.rerun()

# ----------------------------------------------------------------------
# Form (Only if not submitted or after restart)
# ----------------------------------------------------------------------
if not st.session_state.submitted:
    st.markdown("## Complete the Screening")

    tab1, tab2, tab3 = st.tabs(["Demographics", "Depression", "Anxiety"])

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
        st.markdown("### Depression Assessment")
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
                                                   label_visibility="collapsed", key=f"phq_{i}")
            st.markdown("---")
        phq_total = sum(phq.values())
        st.markdown(f"### Depression (PHQ-8) Score: **{phq_total}** / 24")

    with tab3:
        st.markdown("### Anxiety Assessment")
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
                                                   label_visibility="collapsed", key=f"gad_{i}")
            st.markdown("---")
        gad_total = sum(gad.values())
        st.markdown(f"### Anxiety (GAD-7) Score: **{gad_total}** / 21")

    # Submit Button
    st.markdown("---")
    _, c, _ = st.columns([1,2,1])
    with c:
        submitted = st.button("Run Screening", use_container_width=True, key="submit_btn")

    if submitted:
        st.session_state.submitted = True
        st.session_state.phq_total = phq_total
        st.session_state.gad_total = gad_total
        st.session_state.input_data = {
            "Boarding_day": boarding_day, "School_type": school_type, "School_Demographics": school_demo,
            "School_County": school_county, "Age": int(age), "Gender": 1 if gender == "Male" else 2,
            "Form": int(form), "Religion": 1 if religion == "Christian" else 2 if religion == "Muslim" else 3,
            "Parents_Home": {"None":0, "One parent":1, "Both parents":2}.get(parents_home, 0),
            "Parents_Dead": int(parents_dead),
            "Fathers_Education": {"None":0, "Primary":1, "Secondary":2, "Tertiary":3, "University":4}.get(fathers_edu, 0),
            "Mothers_Education": {"None":0, "Primary":1, "Secondary":2, "Tertiary":3, "University":4}.get(mothers_edu, 0),
            "Co_Curricular": 1 if co_curr == "Yes" else 0, "Sports": 1 if sports == "Yes" else 0,
            "Percieved_Academic_Abilities": int(acad_ability)
        }
        st.session_state.input_data.update(phq)
        st.session_state.input_data.update(gad)
        st.rerun()

## ----------------------------------------------------------------------
# Results Section (Only after submission)
# ----------------------------------------------------------------------
else:
    with st.spinner("Running models..."):
        user_df = pd.DataFrame([st.session_state.input_data])
        all_preds = {}
        all_probas = {}

        for name, pipe in pipelines.items():
            try:
                pred = pipe.predict(user_df)[0]
                # Safely get probabilities
                try:
                    proba = pipe.predict_proba(user_df)[0]
                    dep_prob = proba[0][1] if len(proba) > 0 and len(proba[0]) > 1 else None
                    anx_prob = proba[1][1] if len(proba) > 1 and len(proba[1]) > 1 else None
                except:
                    dep_prob = None
                    anx_prob = None

                all_preds[name] = {'dep': pred[0], 'anx': pred[1] if len(pred) > 1 else None}
                all_probas[name] = {'dep': dep_prob, 'anx': anx_prob}
            except Exception as e:
                st.warning(f"Model {name} failed: {e}")
                all_preds[name] = {'dep': None, 'anx': None}
                all_probas[name] = {'dep': None, 'anx': None}

        # Select best models
        if selection_mode == "Manual Selection":
            best_dep_model = st.session_state.get("manual_dep_model", list(pipelines.keys())[0] if pipelines else None)
            best_anx_model = st.session_state.get("manual_anx_model", list(pipelines.keys())[0] if pipelines else None)
        else:
            best_dep_model = max(model_metrics.items(),
                                 key=lambda x: x[1].get('test_recall_per_target',{}).get('Is_Depressed',0))[0] \
                             if model_metrics else None
            best_anx_model = max(model_metrics.items(),
                                 key=lambda x: x[1].get('test_recall_per_target',{}).get('Has_anxiety',0))[0] \
                             if model_metrics else None

        dep_pred = all_preds.get(best_dep_model, {}).get('dep')
        anx_pred = all_preds.get(best_anx_model, {}).get('anx')
        dep_proba = all_probas.get(best_dep_model, {}).get('dep')
        anx_proba = all_probas.get(best_anx_model, {}).get('anx')

        # Severity (labels unchanged, just displayed differently)
        phq_total = st.session_state.phq_total
        gad_total = st.session_state.gad_total
        dep_cat = "Minimal" if phq_total < 5 else "Mild" if phq_total < 10 else "Moderate" if phq_total < 15 else "Moderately Severe" if phq_total < 20 else "Severe"
        anx_cat = "Minimal" if gad_total < 5 else "Mild" if gad_total < 10 else "Moderate" if gad_total < 15 else "Severe"

        # Live Metrics (Safe)
        threshold = 0.5
        y_true_dep = [1 if phq_total >= 10 else 0]
        y_true_anx = [1 if gad_total >= 10 else 0]

        dep_pred_live = 1 if (dep_proba is not None and dep_proba >= threshold) else 0
        anx_pred_live = 1 if (anx_proba is not None and anx_proba >= threshold) else 0

        live_recall_dep = recall_score(y_true_dep, [dep_pred_live], zero_division=0)
        live_acc_dep = accuracy_score(y_true_dep, [dep_pred_live])
        live_recall_anx = recall_score(y_true_anx, [anx_pred_live], zero_division=0)
        live_acc_anx = accuracy_score(y_true_anx, [anx_pred_live])

    # If high score(s) detected, show a dismissible banner / popup
    high_alert = (phq_total >= 15) or (gad_total >= 15)
    if high_alert:
        # big dismissible banner (fixed at top)
        st.markdown(f"""
        <div id="high-alert" style="position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:9999;
                        background:#ff4d4f;color:white;padding:14px 18px;border-radius:8px;
                        box-shadow:0 6px 18px rgba(0,0,0,0.15);font-weight:700;max-width:920px;">
            High score detected — Immediate support recommended.
            <button class="alert-close" onclick="document.getElementById('high-alert').style.display='none'">✕</button>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Screening Results")

    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**Depression Model:** {best_dep_model or 'N/A'}")
        st.caption(f"Live Recall: {live_recall_dep:.1%} | Live Accuracy: {live_acc_dep:.1%}")
    with c2:
        st.info(f"**Anxiety Model:** {best_anx_model or 'N/A'}")
        st.caption(f"Live Recall: {live_recall_anx:.1%} | Live Accuracy: {live_acc_anx:.1%}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="score-card" style="border-left:5px solid #1f77b4">
            <h3 style="margin:0;color:#1f77b4">Depression</h3>
            <div class="score-number">{phq_total}<span style="font-size:1.1rem;color:#666">/24</span></div>
            <div class="score-label">{dep_cat}</div>
        </div>
        """, unsafe_allow_html=True)
        pred_text = f"**Model Prediction:** `{dep_pred}`"
        if dep_proba is not None:
            pred_text += f" (Confidence: **{dep_proba:.1%}**)"
        st.markdown(pred_text)

    with col2:
        st.markdown(f"""
        <div class="score-card" style="border-left:5px solid #ff7f0e">
            <h3 style="margin:0;color:#ff7f0e">Anxiety</h3>
            <div class="score-number">{gad_total}<span style="font-size:1.1rem;color:#666">/21</span></div>
            <div class="score-label">{anx_cat}</div>
        </div>
        """, unsafe_allow_html=True)
        pred_text = f"**Model Prediction:** `{anx_pred}`"
        if anx_proba is not None:
            pred_text += f" (Confidence: **{anx_proba:.1%}**)"
        st.markdown(pred_text)

    # SHAP (unchanged)
    st.markdown("---")
    st.markdown("### Model Explanations")
    tab_d, tab_a = st.tabs(["Depression", "Anxiety"])

       # ----------------------------------------------------------------------
    # SHAP Explanations (Simplified for Non-Technical Users)
    # ----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Model Explanations")
    tab_d, tab_a = st.tabs(["Depression", "Anxiety"])

    def generate_shap_plot(pipe, user_df, target_idx, title):
        try:
            pre = pipe.named_steps['preprocessor']
            clf = pipe.named_steps['clf']
            X = pre.transform(user_df)
            if hasattr(X, "toarray"):
                X = X.toarray()

            # Friendly feature names
            feature_names = pre.get_feature_names_out() if hasattr(pre, 'get_feature_names_out') else [
                f"f{i}" for i in range(X.shape[1])
            ]
            friendly_names = {
                "Parents_Dead": "Number of Parents Deceased",
                "Parents_Home": "Parents at Home",
                "Percieved_Academic_Abilities": "Self-Rated Academic Ability",
                "Co_Curricular": "Participates in Co-curricular",
                "Sports": "Participates in Sports",
                "Gender_1": "Male",
                "Gender_2": "Female",
                "Age": "Age",
                "Form": "Form Level",
            }
            feature_names = [friendly_names.get(f, f.replace("_", " ")) for f in feature_names]

            # Choose the right SHAP explainer automatically
            base = clf.estimators_[target_idx] if hasattr(clf, "estimators_") else clf

            # Use shap.Explainer (unified API handles both tree & linear models)
            explainer = shap.Explainer(base, X, feature_names=feature_names)
            shap_values = explainer(X)

            # Single sample → use shap_values.values[0]
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Impact": shap_values.values[0]
            })
            shap_df["Direction"] = shap_df["Impact"].apply(
                lambda x: "Increased Risk" if x > 0 else "Reduced Risk"
            )
            shap_df["AbsImpact"] = shap_df["Impact"].abs()

            # Top 10 features
            top_features = shap_df.sort_values("AbsImpact", ascending=False).head(10)
            top_features = top_features.sort_values("Impact", ascending=True)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = top_features["Impact"].apply(lambda x: "#e74c3c" if x > 0 else "#2ecc71")
            bars = ax.barh(top_features["Feature"], top_features["Impact"], color=colors)

            ax.set_title(f"Key Factors Influencing {title}", fontweight="bold")
            ax.set_xlabel("Influence on Prediction (← reduces | increases →)")
            ax.axvline(0, color="gray", linewidth=1)
            ax.grid(axis="x", linestyle="--", alpha=0.5)

            for bar, val in zip(bars, top_features["Impact"]):
                ax.text(
                    val + (0.02 if val > 0 else -0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}",
                    va="center",
                    ha="left" if val > 0 else "right",
                    color="black",
                    fontsize=9,
                )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption("""
            **How to interpret:**
            - 🔴 Features on the right **increase** the predicted risk.
            - 🟢 Features on the left **reduce** the predicted risk.
            - Bar length = strength of influence on this prediction.
            """)
            return True

        except Exception as e:
            st.warning(f"Unable to generate simplified SHAP: {e}")
            return False


    with tab_d:
        if best_dep_model and pipelines.get(best_dep_model):
            generate_shap_plot(
                pipelines[best_dep_model],
                user_df,
                target_idx=0,
                title="Depression Prediction"
            )
        else:
            st.info("No model available.")

    with tab_a:
        if best_anx_model and pipelines.get(best_anx_model):
            idx = 0 if best_dep_model != best_anx_model else 1
            generate_shap_plot(
                pipelines[best_anx_model],
                user_df,
                target_idx=idx,
                title="Anxiety Prediction"
            )
        else:
            st.info("No model available.")


    # Crisis Alert details (unchanged contact info)
    if phq_total >= 15 or gad_total >= 15:
        st.markdown("---")
        st.error("### High Score Detected")
        st.markdown("""
        **Immediate support is recommended.**
        - **Kenya Red Cross:** 1199  
        - **Befrienders Kenya:** +254 722 178 177  
        - **School Counselor**
        """)

    # Download Report (unchanged but renamed labels)
    st.markdown("---")
    dep_conf = f" (Conf: {dep_proba:.1%})" if dep_proba is not None else ""
    anx_conf = f" (Conf: {anx_proba:.1%})" if anx_proba is not None else ""

    report = f"""
SCREENING RESULTS
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Depression (PHQ-8) Score: {phq_total}/24 → {dep_cat}
Anxiety (GAD-7) Score: {gad_total}/21 → {anx_cat}

Model (Depression): {best_dep_model or 'N/A'} → Prediction: {dep_pred}{dep_conf}
Model (Anxiety): {best_anx_model or 'N/A'} → Prediction: {anx_pred}{anx_conf}

Live Metrics:
  Depression → Recall: {live_recall_dep:.1%}, Accuracy: {live_acc_dep:.1%}
  Anxiety → Recall: {live_recall_anx:.1%}, Accuracy: {live_acc_anx:.1%}

This is a screening tool only. Not a diagnosis.
    """
    st.download_button("Download Results", report,
                       file_name=f"screening_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                       mime="text/plain")

    st.markdown("---")
    st.warning("""
    **SCREENING ONLY**  
    This tool uses PHQ-8 and GAD-7 for screening.  
    Results are **not a diagnosis**.  
    High scores indicate need for professional evaluation.
    """)
