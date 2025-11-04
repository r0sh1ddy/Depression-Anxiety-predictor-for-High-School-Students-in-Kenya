import streamlit as st
import pandas as pd
import pickle, os, numpy as np
import shap, matplotlib.pyplot as plt
import plotly.graph_objects as go

# ----------------------------------------------------------------------
#  Page config & CSS
# ----------------------------------------------------------------------
icon_path = "app_images/80798728-1633-47f7-a720-6d1cb06d3cae.jpg"
logo_path = "app_images/80798728-1633-47f7-a720-6d1cb06d3cae.jpg"

st.set_page_config(
    page_title="AdolescentMind",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=icon_path if os.path.exists(icon_path) else "brain"
)

if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'background_style' not in st.session_state:
    st.session_state.background_style = "default"
if 'custom_bg_color' not in st.session_state:
    st.session_state.custom_bg_color = "#f0f2f6"
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'custom_bg_image' not in st.session_state:
    st.session_state.custom_bg_image = None


# ----------------------------------------------------------------------
# Dynamic Background CSS (supports uploaded image)
# ----------------------------------------------------------------------
def get_background_css():
    if (st.session_state.background_style == "uploaded_image"
            and st.session_state.get("custom_bg_image")):
        return f"""
        <style>
            .stApp {{
                background-image: url("{st.session_state.custom_bg_image}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .score-card,
            .stMarkdown > div, .stSelectbox > div > div,
            .stTextInput > div > div, .stNumberInput > div > div,
            .stSlider > div > div, .stButton > button,
            .stExpander > div > div {{
                background: rgba(255,255,255,0.92) !important;
            }}
        </style>
        """
    elif st.session_state.background_style == "custom_color":
        return f"""
        <style>
            .stApp {{ 
                background: {st.session_state.custom_bg_color}; 
                background-attachment: fixed;
            }}
            .score-card {{ background: rgba(255,255,255,0.9); }}
        </style>
        """
    elif st.session_state.background_style == "gradient_blue":
        return """
        <style>
            .stApp { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                background-attachment: fixed;
            }
            .score-card { background: rgba(255,255,255,0.95); }
        </style>
        """
    elif st.session_state.background_style == "gradient_green":
        return """
        <style>
            .stApp { 
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                background-attachment: fixed;
            }
            .score-card { background: rgba(255,255,255,0.95); }
        </style>
        """
    else:
        return """
        <style>
            .stApp { background: #f0f2f6; }
        </style>
        """


# inject CSS
st.markdown(get_background_css(), unsafe_allow_html=True)

# CSS
st.markdown("""
<style>
    :root {
        --primary: #1f77b4;
        --primary-dark: #155a8a;
        --danger: #e74c3c;
        --danger-dark: #c0392b;
        --success: #27ae60;
        --text: #2c3e50;
        --bg-card: rgba(255, 255, 255, 0.94);
    }

    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: var(--primary);
        text-align: center;
        margin: 1rem 0 0.5rem;
        font-family: 'Segoe UI', sans-serif;
    }
    .sub-header {
        font-size: 1.25rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }

    .score-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        background: var(--bg-card);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,0.3);
        transition: transform 0.2s;
    }
    .score-card:hover {
        transform: translateY(-4px);
    }

    .score-number {
        font-size: 4.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, var(--primary), #4a90e2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .score-label {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text);
    }

    .best-model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin: 0.8rem 0;
        font-size: 0.95rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: var(--primary) !important;
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        padding: 0.8rem 1rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: var(--primary-dark) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
    }

        .severity-indicator {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .severity-minimal { background: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .severity-mild { background: #fff3cd; color: #856404; border: 2px solid #ffeaa7; }
    .severity-moderate { background: #ffe5b4; color: #d35400; border: 2px solid #ffa500; }
    .severity-mod-severe { background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    .severity-severe { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }

    .score-display {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    /* Restart Button */
    .restart-button > button {
        background: var(--danger) !important;
        color: white !important;
    }
    .restart-button > button:hover {
        background: var(--danger-dark) !important;
        transform: translateY(-2px) !important;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .main-header { font-size: 2.2rem; }
        .score-number { font-size: 3.5rem; }
        .stButton > button { font-size: 1.1rem; padding: 0.7rem; }
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Header & Logo
# ----------------------------------------------------------------------
BASE = os.path.dirname(__file__)
LOGO_PATH = os.path.join(BASE, "app_images", "icon.jpg")
FAVICON_PATH = os.path.join(BASE, "app_images", "Adolescence.jpg")

if os.path.exists(FAVICON_PATH):
    st.set_page_config(page_icon=FAVICON_PATH)

# Header
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    if os.path.exists(LOGO_PATH):
        st.image(
            LOGO_PATH,
            width=280,
            use_container_width=False,
            caption=None
        )
    else:
        st.markdown('<div class="main-header">AdolescentMind</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="sub-header">Depression & Anxiety Screening for Kenyan High School Students</div>',
    unsafe_allow_html=True
)

#  Load models
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


#  Sidebar
with st.sidebar:
    st.title("Settings")
    # Background Style
    st.markdown("### Background Style")
    bg_options = ["Default", "Custom Color", "Gradient Blue",
                  "Gradient Green", "Uploaded Image"]

    # Keep current selection on rerun
    current_idx = 0
    if st.session_state.background_style in ["default", "custom_color",
                                             "gradient_blue", "gradient_green",
                                             "uploaded_image"]:
        mapping = {
            "default": "Default",
            "custom_color": "Custom Color",
            "gradient_blue": "Gradient Blue",
            "gradient_green": "Gradient Green",
            "uploaded_image": "Uploaded Image"
        }
        current_idx = bg_options.index(mapping[st.session_state.background_style])

    bg_choice = st.selectbox(
        "Choose Background",
        bg_options,
        index=current_idx,
        key="bg_selector"
    )

    # Map selection to session_state
    style_map = {
        "Default":        "default",
        "Custom Color":   "custom_color",
        "Gradient Blue":  "gradient_blue",
        "Gradient Green": "gradient_green",
        "Uploaded Image": "uploaded_image"
    }
    st.session_state.background_style = style_map[bg_choice]

    # Custom color picker
    if bg_choice == "Custom Color":
        color = st.color_picker("Pick a colour", st.session_state.custom_bg_color)
        st.session_state.custom_bg_color = color

    # Image uploader
    if bg_choice == "Uploaded Image":
        st.markdown("### Upload Your Background")
        uploaded_bg = st.file_uploader(
            "Choose JPG / PNG",
            type=["png", "jpg", "jpeg"],
            help="Max 5 MB – will be resized to cover the screen"
        )
        if uploaded_bg is not None:
            st.image(uploaded_bg, caption="Preview (200 px)", width=200)
            if st.button("Set as Background", key="apply_bg_image"):
                import base64
                img_bytes = uploaded_bg.read()
                b64 = base64.b64encode(img_bytes).decode()
                data_url = f"data:image/png;base64,{b64}"
                st.session_state.custom_bg_image = data_url
                st.session_state.background_style = "uploaded_image"
                st.success("Background applied!")
                st.rerun()

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

#  Form
if not st.session_state.submitted:
    st.markdown("## Complete the Screening")

    tab1, tab2, tab3 = st.tabs(["Demographics", "PHQ-8", "GAD-7"])

    with tab1:
        st.markdown("### School & Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            boarding_day = st.selectbox("School Type", ["Boarding", "Day"], key="boarding_day")
            school_type = st.selectbox("School Gender", ["Boys", "Girls", "Mixed"], key="school_type")
            school_demo = st.selectbox("School Level", ['Subcounty', 'Extracounty', 'County'], key="school_demo")
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
        st.markdown("**What is PHQ-8?** The Patient Health Questionnaire-8 (PHQ-8) is a simple, validated tool to screen for depression symptoms over the past 2 weeks...")
        st.markdown('<div class="assessment-info">Remember: Your answers are private and used only for this screening.</div>', unsafe_allow_html=True)
        
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
            with c1: 
                st.markdown(f"**{i}.** {q}")
            with c2:
                phq[f'PHQ_{i}'] = st.select_slider(
                    f"p{i}", 
                    options=[0,1,2,3], 
                    format_func=lambda x: likert[x], 
                    label_visibility="collapsed", 
                    key=f"phq_{i}"
                )
            st.markdown("---")
        
        # Real-time calculation and display
        phq_total = sum(phq.values())
        
        # Determine severity category
        if phq_total < 5:
            phq_severity = "Minimal"
            severity_class = "severity-minimal"
        elif phq_total < 10:
            phq_severity = "Mild"
            severity_class = "severity-mild"
        elif phq_total < 15:
            phq_severity = "Moderate"
            severity_class = "severity-moderate"
        elif phq_total < 20:
            phq_severity = "Moderately Severe"
            severity_class = "severity-mod-severe"
        else:
            phq_severity = "Severe"
            severity_class = "severity-severe"
        
        # Display score and severity
        st.markdown(f"""
        <div class="{severity_class} severity-indicator">
            <div style="font-size: 0.9rem; opacity: 0.8;">PHQ-8 Score</div>
            <div class="score-display">{phq_total} / 24</div>
            <div style="font-size: 1.2rem; margin-top: 0.5rem;">Severity: {phq_severity}</div>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### GAD-7 Anxiety Assessment")
        st.markdown("**What is GAD-7?** The Generalized Anxiety Disorder-7 (GAD-7) is a quick, evidence-based questionnaire to assess anxiety symptoms...")
        st.markdown('<div class="assessment-info">Remember: Your answers are private and used only for this screening.</div>', unsafe_allow_html=True)
        
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
            with c1: 
                st.markdown(f"**{i}.** {q}")
            with c2:
                gad[f'GAD_{i}'] = st.select_slider(
                    f"g{i}", 
                    options=[0,1,2,3], 
                    format_func=lambda x: likert[x], 
                    label_visibility="collapsed", 
                    key=f"gad_{i}"
                )
            st.markdown("---")
        
        # Real-time calculation and display
        gad_total = sum(gad.values())
        
        # Determine severity category
        if gad_total < 5:
            gad_severity = "Minimal"
            severity_class = "severity-minimal"
        elif gad_total < 10:
            gad_severity = "Mild"
            severity_class = "severity-mild"
        elif gad_total < 15:
            gad_severity = "Moderate"
            severity_class = "severity-moderate"
        else:
            gad_severity = "Severe"
            severity_class = "severity-severe"
        
        # Display score and severity
        st.markdown(f"""
        <div class="{severity_class} severity-indicator">
            <div style="font-size: 0.9rem; opacity: 0.8;">GAD-7 Score</div>
            <div class="score-display">{gad_total} / 21</div>
            <div style="font-size: 1.2rem; margin-top: 0.5rem;">Severity: {gad_severity}</div>
        </div>
        """, unsafe_allow_html=True)

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
        if hasattr(X, "toarray"): 
            X = X.toarray()
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
        
    if submitted and (phq_total >= 15 or gad_total >= 15):
        st.error("### ⚠️ URGENT: High Score Detected")
        st.markdown("""
        **[Immediate support is recommended - Click for help resources](https://www.healthyplace.com/other-info/resources/mental-health-hotline-numbers-and-referral-resources)**
        
        **Crisis Support:** Kenya Red Cross: **1199** | Befrienders Kenya: **+254 722 178 177** | Lifeline Kenya: **1195**
        """)
        st.markdown("---")

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
           **[Immediate support is recommended - Click for help resources](https://www.healthyplace.com/other-info/resources/mental-health-hotline-numbers-and-referral-resources)**
        
        **Crisis Support:** Kenya Red Cross: **1199** | Befrienders Kenya: **+254 722 178 177** | Lifeline Kenya: **1195**
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