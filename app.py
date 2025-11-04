import streamlit as st
import pandas as pd
import pickle, os, numpy as np
import shap, matplotlib.pyplot as plt
import plotly.graph_objects as go

TEST_DATA_FILE = os.path.join(BASE, "test_data.csv") 

test_data = None
if os.path.exists(TEST_DATA_FILE):
    try:
        if TEST_DATA_FILE.endswith('.pkl'):
            with open(TEST_DATA_FILE, "rb") as f:
                test_data = pickle.load(f)
        else:  # CSV
            test_data = pd.read_csv(TEST_DATA_FILE)
        st.sidebar.success(f"Test data loaded: {len(test_data)} samples")
    except Exception as e:
        st.sidebar.error(f"Test data load failed: {e}")
else:
    st.sidebar.warning("No test data found - using stored metrics")


# Add this function to calculate real-time metrics
def calculate_live_metrics(model, X_test, y_test, target_idx):
    """
    Calculate real-time recall and accuracy for a specific target
    
    Args: session, 1 for anxiety
    
    Returns:
        dict with recall and accuracy
    """
    from sklearn.metrics import recall_score, accuracy_score
    
    try:
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Handle multi-output vs single output
        if len(y_pred.shape) > 1:
            y_pred_target = y_pred[:, target_idx]
        else:
            y_pred_target = y_pred
            
        # Handle multi-output ground truth
        if len(y_test.shape) > 1:
            y_test_target = y_test[:, target_idx]
        else:
            y_test_target = y_test
        
        # Calculate metrics
        recall = recall_score(y_test_target, y_pred_target, zero_division=0)
        accuracy = accuracy_score(y_test_target, y_pred_target)
        
        return {
            'recall': recall,
            'accuracy': accuracy,
            'samples': len(y_test_target)
        }
    except Exception as e:
        st.warning(f"Metric calculation error: {e}")
        return {'recall': 0, 'accuracy': 0, 'samples': 0}

#  Page config 
icon_path = "app_images/icon.jpg"
logo_path = "app_images/icon.jpg"

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

# Dynamic Background CSS 
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
    with st.spinner("Running live predictions and calculating real-time metrics"):
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

        # --- Prepare test data if available ---
        use_live_metrics = test_data is not None
        
        if use_live_metrics:
            # Separate features and labels from test data
            # Adjust these column names based on your actual test data structure
            target_cols = ['Is_Depressed', 'Has_anxiety']
            feature_cols = [col for col in test_data.columns if col not in target_cols]
            
            X_test = test_data[feature_cols]
            y_test = test_data[target_cols].values  # Convert to numpy array

        # --- LIVE PREDICTIONS from all models ---
        st.markdown("---")
        st.markdown("## Live Model Predictions & Real-time Metrics")
        
        if use_live_metrics:
            st.success(f"Computing live metrics on {len(test_data)} test samples")
        else:
            st.info("ℹUsing pre-computed metrics (no test data available)")
        
        live_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (model_name, pipe) in enumerate(pipelines.items()):
            status_text.text(f"Processing {model_name}...")
            progress_bar.progress((idx + 1) / len(pipelines))
            
            try:
                # Get live prediction for user
                prediction = pipe.predict(user_df)[0]
                
                # Get probability scores if available
                try:
                    proba = pipe.predict_proba(user_df)[0]
                    dep_proba = proba[0][1] if len(proba[0]) > 1 else None
                    anx_proba = proba[1][1] if len(proba) > 1 and len(proba[1]) > 1 else None
                except:
                    dep_proba = None
                    anx_proba = None
                
                # Calculate LIVE metrics if test data available
                if use_live_metrics:
                    dep_metrics = calculate_live_metrics(pipe, X_test, y_test, target_idx=0)
                    anx_metrics = calculate_live_metrics(pipe, X_test, y_test, target_idx=1)
                    
                    dep_recall = dep_metrics['recall']
                    dep_accuracy = dep_metrics['accuracy']
                    anx_recall = anx_metrics['recall']
                    anx_accuracy = anx_metrics['accuracy']
                    metric_source = "LIVE"
                else:
                    # Fallback to stored metrics
                    metrics = model_metrics.get(model_name, {})
                    dep_recall = metrics.get('test_recall_per_target', {}).get('Is_Depressed', 0)
                    dep_accuracy = metrics.get('test_accuracy_per_target', {}).get('Is_Depressed', 0)
                    anx_recall = metrics.get('test_recall_per_target', {}).get('Has_anxiety', 0)
                    anx_accuracy = metrics.get('test_accuracy_per_target', {}).get('Has_anxiety', 0)
                    metric_source = "STORED"
                
                # Store results
                live_results.append({
                    'model_name': model_name,
                    'dep_prediction': int(prediction[0]),
                    'anx_prediction': int(prediction[1]) if len(prediction) > 1 else None,
                    'dep_probability': dep_proba,
                    'anx_probability': anx_proba,
                    'dep_recall': dep_recall,
                    'dep_accuracy': dep_accuracy,
                    'anx_recall': anx_recall,
                    'anx_accuracy': anx_accuracy,
                    'metric_source': metric_source
                })
                
            except Exception as e:
                st.warning(f"Model {model_name} failed: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if not live_results:
            st.error("No models could generate predictions. Please check your model files.")
            st.stop()
        
        # --- SELECT BEST MODELS based on HIGHEST RECALL ---
        best_dep = max(live_results, key=lambda x: x['dep_recall'])
        best_anx = max(live_results, key=lambda x: x['anx_recall'])
        
        st.success(f"Generated predictions from {len(live_results)} model(s) | Metrics: {live_results[0]['metric_source']}")
        
        # --- DISPLAY BEST MODEL SELECTIONS ---
        st.markdown("---")
        st.markdown("## Selected Best Models (Highest Recall)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 12px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin:0; color: white;">🧠 Depression Model</h3>
                    <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 0.6rem; border-radius: 6px; font-size: 0.75rem;">
                        {best_dep['metric_source']}
                    </span>
                </div>
                <hr style="border-color: rgba(255,255,255,0.3); margin: 1rem 0;">
                <div style="font-size: 1rem; font-weight: bold; margin: 0.8rem 0; line-height: 1.4;">
                    {best_dep['model_name']}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.2rem;">
                    <div style="text-align: center; background: rgba(255,255,255,0.15); padding: 0.8rem; border-radius: 8px;">
                        <div style="font-size: 0.8rem; opacity: 0.9; margin-bottom: 0.3rem;">Recall</div>
                        <div style="font-size: 1.9rem; font-weight: bold;">{best_dep['dep_recall']:.1%}</div>
                    </div>
                    <div style="text-align: center; background: rgba(255,255,255,0.15); padding: 0.8rem; border-radius: 8px;">
                        <div style="font-size: 0.8rem; opacity: 0.9; margin-bottom: 0.3rem;">Accuracy</div>
                        <div style="font-size: 1.9rem; font-weight: bold;">{best_dep['dep_accuracy']:.1%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                        padding: 1.5rem; border-radius: 12px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin:0; color: white;">😰 Anxiety Model</h3>
                    <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 0.6rem; border-radius: 6px; font-size: 0.75rem;">
                        {best_anx['metric_source']}
                    </span>
                </div>
                <hr style="border-color: rgba(255,255,255,0.3); margin: 1rem 0;">
                <div style="font-size: 1rem; font-weight: bold; margin: 0.8rem 0; line-height: 1.4;">
                    {best_anx['model_name']}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.2rem;">
                    <div style="text-align: center; background: rgba(255,255,255,0.15); padding: 0.8rem; border-radius: 8px;">
                        <div style="font-size: 0.8rem; opacity: 0.9; margin-bottom: 0.3rem;">Recall</div>
                        <div style="font-size: 1.9rem; font-weight: bold;">{best_anx['anx_recall']:.1%}</div>
                    </div>
                    <div style="text-align: center; background: rgba(255,255,255,0.15); padding: 0.8rem; border-radius: 8px;">
                        <div style="font-size: 0.8rem; opacity: 0.9; margin-bottom: 0.3rem;">Accuracy</div>
                        <div style="font-size: 1.9rem; font-weight: bold;">{best_anx['anx_accuracy']:.1%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # --- LIVE PREDICTION RESULTS ---
        st.markdown("---")
        st.markdown("## Your Screening Results")
        
        # Severity categories
        dep_cat = "Minimal" if phq_total < 5 else "Mild" if phq_total < 10 else "Moderate" if phq_total < 15 else "Moderately Severe" if phq_total < 20 else "Severe"
        anx_cat = "Minimal" if gad_total < 5 else "Mild" if gad_total < 10 else "Moderate" if gad_total < 15 else "Severe"
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Depression Result
            dep_risk = "POSITIVE (At Risk)" if best_dep['dep_prediction'] == 1 else "NEGATIVE (Low Risk)"
            dep_color = "#e74c3c" if best_dep['dep_prediction'] == 1 else "#27ae60"
            dep_icon = "🔴" if best_dep['dep_prediction'] == 1 else "🟢"
            
            st.markdown(f"""
            <div class="score-card" style="border-left:5px solid {dep_color}">
                <h3 style="margin:0;color:#1f77b4">PHQ-8 Depression</h3>
                <div class="score-number">{phq_total}<span style="font-size:2rem;color:#666">/24</span></div>
                <div class="score-label">{dep_cat}</div>
                <hr style="margin: 1rem 0;">
                <div style="font-size: 1.3rem; font-weight: bold; color: {dep_color}; margin: 0.5rem 0;">
                    {dep_icon} {dep_risk}
                </div>
                <div style="font-size: 0.85rem; color: #666; margin-top: 0.8rem; background: #f8f9fa; padding: 0.5rem; border-radius: 6px;">
                    <strong>Model:</strong> {best_dep['model_name'][:35]}{"..." if len(best_dep['model_name']) > 35 else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if best_dep['dep_probability'] is not None:
                st.metric("🎯 Risk Probability", f"{best_dep['dep_probability']:.1%}", 
                         help="Model's confidence in this prediction")
        
        with col2:
            # Anxiety Result
            anx_risk = "POSITIVE (At Risk)" if best_anx['anx_prediction'] == 1 else "NEGATIVE (Low Risk)"
            anx_color = "#e74c3c" if best_anx['anx_prediction'] == 1 else "#27ae60"
            anx_icon = "🔴" if best_anx['anx_prediction'] == 1 else "🟢"
            
            st.markdown(f"""
            <div class="score-card" style="border-left:5px solid {anx_color}">
                <h3 style="margin:0;color:#ff7f0e">GAD-7 Anxiety</h3>
                <div class="score-number">{gad_total}<span style="font-size:2rem;color:#666">/21</span></div>
                <div class="score-label">{anx_cat}</div>
                <hr style="margin: 1rem 0;">
                <div style="font-size: 1.3rem; font-weight: bold; color: {anx_color}; margin: 0.5rem 0;">
                    {anx_icon} {anx_risk}
                </div>
                <div style="font-size: 0.85rem; color: #666; margin-top: 0.8rem; background: #f8f9fa; padding: 0.5rem; border-radius: 6px;">
                    <strong>Model:</strong> {best_anx['model_name'][:35]}{"..." if len(best_anx['model_name']) > 35 else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if best_anx['anx_probability'] is not None:
                st.metric("Risk Probability", f"{best_anx['anx_probability']:.1%}",
                         help="Model's confidence in this prediction")
        
        # --- ALL MODELS COMPARISON ---
        st.markdown("---")
        with st.expander("View All Model Predictions & Live Metrics", expanded=False):
            st.markdown("### Depression Predictions")
            dep_df = pd.DataFrame([{
                'Model': r['model_name'],
                'Prediction': '🔴 Positive' if r['dep_prediction'] == 1 else '🟢 Negative',
                'Probability': f"{r['dep_probability']:.1%}" if r['dep_probability'] else 'N/A',
                'Recall': f"{r['dep_recall']:.1%}",
                'Accuracy': f"{r['dep_accuracy']:.1%}",
                'Source': r['metric_source']
            } for r in live_results])
            st.dataframe(dep_df, use_container_width=True, hide_index=True)
            
            st.markdown("### Anxiety Predictions")
            anx_df = pd.DataFrame([{
                'Model': r['model_name'],
                'Prediction': '🔴 Positive' if r['anx_prediction'] == 1 else '🟢 Negative',
                'Probability': f"{r['anx_probability']:.1%}" if r['anx_probability'] else 'N/A',
                'Recall': f"{r['anx_recall']:.1%}",
                'Accuracy': f"{r['anx_accuracy']:.1%}",
                'Source': r['metric_source']
            } for r in live_results if r['anx_prediction'] is not None])
            st.dataframe(anx_df, use_container_width=True, hide_index=True)
        
        # --- SHAP EXPLANATIONS ---
        st.markdown("---")
        st.markdown("## Model Explanations (SHAP Analysis)")
        st.info("Understanding which factors influenced your risk assessment")
        
        tab_d, tab_a = st.tabs(["Depression Factors", "Anxiety Factors"])
        
        with tab_d:
            if best_dep and pipelines.get(best_dep['model_name']):
                generate_shap_plot(
                    pipelines[best_dep['model_name']], 
                    user_df,
                    target_idx=0, 
                    title=f"Depression Risk Factors - {best_dep['model_name']}"
                )
            else:
                st.info("No model available for SHAP analysis.")
        
        with tab_a:
            if best_anx and pipelines.get(best_anx['model_name']):
                # Determine target index based on whether same model is used
                target_idx = 0 if best_dep['model_name'] != best_anx['model_name'] else 1
                generate_shap_plot(
                    pipelines[best_anx['model_name']], 
                    user_df,
                    target_idx=target_idx, 
                    title=f"Anxiety Risk Factors - {best_anx['model_name']}"
                )
            else:
                st.info("No model available for SHAP analysis.")
        
        # --- CRISIS ALERT ---
        if phq_total >= 15 or gad_total >= 15:
            st.markdown("---")
            st.error("### HIGH SCORE ALERT - IMMEDIATE SUPPORT RECOMMENDED")
            st.markdown("""
            **Your scores indicate significant distress. Please reach out for help immediately.**
            
             **Crisis Support Lines (Kenya):**
            - **Kenya Red Cross:** 1199
            - **Befrienders Kenya:** +254 722 178 177
            - **Lifeline Kenya:** 1195
            
             [Click here for more mental health resources](https://www.healthyplace.com/other-info/resources/mental-health-hotline-numbers-and-referral-resources)
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