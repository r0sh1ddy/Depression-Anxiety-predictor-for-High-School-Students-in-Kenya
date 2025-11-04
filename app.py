import streamlit as st
import pandas as pd
import pickle, os, numpy as np
import shap, matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import recall_score, accuracy_score


def update_live_phq():
    total = sum(st.session_state.get(f"phq_live_{i}", 0) for i in range(1, 9))
    st.session_state.live_phq_total = total


def update_live_gad():
    total = sum(st.session_state.get(f"gad_live_{i}", 0) for i in range(1, 8))
    st.session_state.live_gad_total = total


# ----------------------------------------------------------------------
# Page config & Session State Initialization
# ----------------------------------------------------------------------
icon_path = "app_images/icon.jpg"
logo_path = "app_images/icon.jpg"

st.set_page_config(
    page_title="AdolescentMind",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=icon_path if os.path.exists(icon_path) else "brain"
)

# Initialize session state
if 'live_phq_total' not in st.session_state:
    st.session_state.live_phq_total = 0
if 'live_gad_total' not in st.session_state:
    st.session_state.live_gad_total = 0
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

# ----------------------------------------------------------------------
# Global CSS
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Load Models & Metrics
# ----------------------------------------------------------------------
PIPELINE_FILE = os.path.join(BASE, "trained_pipelines.pkl")
METRICS_FILE  = os.path.join(BASE, "model_metrics.pkl")

pipelines = {}
model_metrics = {}

# Load pipelines
if os.path.exists(PIPELINE_FILE):
    try:
        with open(PIPELINE_FILE, "rb") as f:
            loaded = pickle.load(f)
            if isinstance(loaded, dict):
                pipelines = loaded
                st.success(f"Loaded {len(pipelines)} model(s).")
            else:
                st.error("Pipelines file is corrupted or invalid.")
    except Exception as e:
        st.error(f"Failed to load models: {str(e)[:100]}...")
else:
    st.warning("No trained models found. Place `trained_pipelines.pkl` in project folder.")

# Load metrics
if os.path.exists(METRICS_FILE):
    try:
        with open(METRICS_FILE, "rb") as f:
            loaded = pickle.load(f)
            if isinstance(loaded, dict):
                model_metrics = loaded
                st.caption(f"Model performance metrics loaded.")
            else:
                st.error("Metrics file is corrupted.")
    except Exception as e:
        st.error(f"Failed to load metrics: {str(e)[:100]}...")
else:
    st.info("No model metrics found. Performance charts will be unavailable.")


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("Settings")

    # 1. Model Status
    st.markdown("### Model Status")
    if pipelines:
        st.success(f"**{len(pipelines)} model(s)** loaded")
    else:
        st.warning("No models loaded")
    st.markdown("---")

    # 2. Background Style
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

    # 3. Model Selection Mode
    st.markdown("### Model Selection")
    selection_mode = st.radio(
        "Choose Mode:",
        ["Auto-Select Best", "Manual Selection", "View All Models"],
        horizontal=True,
        key="selection_mode"
    )

    if selection_mode == "Manual Selection" and pipelines:
        st.markdown("#### Select Models")
        default_dep = st.session_state.get("manual_dep_model", list(pipelines.keys())[0])
        default_anx = st.session_state.get("manual_anx_model", list(pipelines.keys())[0])

        manual_dep_model = st.selectbox(
            "Depression Model:",
            options=list(pipelines.keys()),
            index=list(pipelines.keys()).index(default_dep),
            key="manual_dep"
        )
        manual_anx_model = st.selectbox(
            "Anxiety Model:",
            options=list(pipelines.keys()),
            index=list(pipelines.keys()).index(default_anx),
            key="manual_anx"
        )
        st.session_state.manual_dep_model = manual_dep_model
        st.session_state.manual_anx_model = manual_anx_model

    st.markdown("---")

    # 4. Best Models by Recall (pre-computed)
    if model_metrics and pipelines:
        st.markdown("### Top Models by Recall")

        def get_best_model(target):
            best_score = -1
            best_name = None
            for name, m in model_metrics.items():
                r = m.get('test_recall_per_target', {}).get(target, 0)
                if r > best_score:
                    best_score = r
                    best_name = name
            return best_name, best_score

        # Depression
        dep_name, dep_recall = get_best_model('Is_Depressed')
        if dep_name:
            acc = model_metrics[dep_name].get('test_accuracy_per_target', {}).get('Is_Depressed', 0)
            st.markdown(f"""
            <div class="best-model-card">
                <div style="font-size:0.9rem;opacity:0.9;">Depression</div>
                <div style="font-size:1.1rem;font-weight:bold;">{dep_name}</div>
                <div style="display:flex;justify-content:space-around;margin-top:0.5rem;">
                    <div><small>Recall</small><br><b>{dep_recall:.1%}</b></div>
                    <div><small>Accuracy</small><br><b>{acc:.1%}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Anxiety
        anx_name, anx_recall = get_best_model('Has_anxiety')
        if anx_name:
            acc = model_metrics[anx_name].get('test_accuracy_per_target', {}).get('Has_anxiety', 0)
            st.markdown(f"""
            <div class="best-model-card">
                <div style="font-size:0.9rem;opacity:0.9;">Anxiety</div>
                <div style="font-size:1.1rem;font-weight:bold;">{anx_name}</div>
                <div style="display:flex;justify-content:space-around;margin-top:0.5rem;">
                    <div><small>Recall</small><br><b>{anx_recall:.1%}</b></div>
                    <div><small>Accuracy</small><br><b>{acc:.1%}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 5. All Models Comparison
    with st.expander("Compare All Models", expanded=False):
        if model_metrics and len(model_metrics) > 1:
            view_metric = st.selectbox(
                "Metric:",
                ["Recall", "Accuracy"],
                key="compare_metric"
            )
            key = 'test_recall_per_target' if view_metric == "Recall" else 'test_accuracy_per_target'

            names, dep_scores, anx_scores = [], [], []
            for name, m in model_metrics.items():
                names.append(name)
                dep_scores.append(m[key].get('Is_Depressed', 0) * 100)
                anx_scores.append(m[key].get('Has_anxiety', 0) * 100)

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Depression', x=names, y=dep_scores,
                                 marker_color='#1f77b4', text=dep_scores,
                                 textposition='outside', texttemplate='%{y:.1f}%'))
            fig.add_trace(go.Bar(name='Anxiety', x=names, y=anx_scores,
                                 marker_color='#ff7f0e', text=anx_scores,
                                 textposition='outside', texttemplate='%{y:.1f}%'))
            fig.update_layout(
                barmode='group',
                height=380,
                title=f"{view_metric} Comparison (%)",
                yaxis=dict(range=[0, 100], title=f"{view_metric} (%)"),
                margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to compare.")

    st.markdown("---")

    # 6. Live Agreement with PHQ-8 / GAD-7 (After Submit)
    if st.session_state.get("submitted", False):
        st.markdown("### Agreement with PHQ-8 / GAD-7")
        dep_agree = st.session_state.get("live_agreement_dep", None)
        anx_agree = st.session_state.get("live_agreement_anx", None)

        if dep_agree is not None:
            st.caption(f"**Depression**: Model agrees with PHQ-8 (≥10) in **{dep_agree:.0%}** of this case.")
        if anx_agree is not None:
            st.caption(f"**Anxiety**: Model agrees with GAD-7 (≥10) in **{anx_agree:.0%}** of this case.")
        st.caption("*Agreement ≠ true diagnostic accuracy – it only shows consistency with the questionnaire.*")


# ----------------------------------------------------------------------
# Restart Button
# ----------------------------------------------------------------------
if st.session_state.submitted:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "Start New Screening",
            key="restart",
            help="Clear current results and start over",
            use_container_width=True
        ):
            # Keep only UI preferences
            keep_keys = {
                'background_style',
                'custom_bg_color',
                'custom_bg_image',
                'manual_dep_model',
                'manual_anx_model'
            }
            for key in list(st.session_state.keys()):
                if key not in keep_keys:
                    del st.session_state[key]
            st.session_state.submitted = False
            st.success("Ready for new screening!")
            st.rerun()


# ----------------------------------------------------------------------
# Screening Form – with live colour-changing scores
# ----------------------------------------------------------------------
if not st.session_state.submitted:
    st.markdown("## Complete the Screening")
    st.markdown("## Live Screening Scores")

    # Helper colour / label functions
    def get_phq_style(score):
        if score < 5:
            return "#27ae60", "Minimal", "rgba(39, 174, 96, 0.1)"
        elif score < 10:
            return "#f39c12", "Mild", "rgba(243, 156, 18, 0.1)"
        elif score < 15:
            return "#e67e22", "Moderate", "rgba(230, 126, 34, 0.1)"
        elif score < 20:
            return "#c0392b", "Moderately Severe", "rgba(192, 57, 43, 0.1)"
        else:
            return "#c0392b", "Severe", "rgba(192, 57, 43, 0.15)"

    def get_gad_style(score):
        if score < 5:
            return "#27ae60", "Minimal", "rgba(39, 174, 96, 0.1)"
        elif score < 10:
            return "#f39c12", "Mild", "rgba(243, 156, 18, 0.1)"
        elif score < 15:
            return "#e67e22", "Moderate", "rgba(230, 126, 34, 0.1)"
        else:
            return "#c0392b", "Severe", "rgba(192, 57, 43, 0.15)"

    phq_c, phq_l, phq_bg = get_phq_style(st.session_state.live_phq_total)
    gad_c, gad_l, gad_bg = get_gad_style(st.session_state.live_gad_total)

    col_live1, col_live2 = st.columns(2)

    with col_live1:
        st.markdown(f"""
        <div class="score-card" style="
            border-left:6px solid {phq_c};
            background:{phq_bg};
            padding:1.3rem;
            border-radius:12px;
            margin:0.8rem 0;
            box-shadow:0 4px 10px rgba(0,0,0,0.08);
            transition:all 0.3s ease;
        ">
            <h4 style="margin:0;color:{phq_c};font-weight:600;">PHQ-8 (Depression)</h4>
            <div style="font-size:2.5rem;font-weight:800;color:{phq_c};margin:0.4rem 0;">
                {st.session_state.live_phq_total}<span style="font-size:1.3rem;color:#666;"> / 24</span>
            </div>
            <div style="font-size:1rem;font-weight:600;color:{phq_c};
                 background:rgba(255,255,255,0.7);display:inline-block;
                 padding:0.2rem 0.6rem;border-radius:6px;">
                {phq_l}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_live2:
        st.markdown(f"""
        <div class="score-card" style="
            border-left:6px solid {gad_c};
            background:{gad_bg};
            padding:1.3rem;
            border-radius:12px;
            margin:0.8rem 0;
            box-shadow:0 4px 10px rgba(0,0,0,0.08);
            transition:all 0.3s ease;
        ">
            <h4 style="margin:0;color:{gad_c};font-weight:600;">GAD-7 (Anxiety)</h4>
            <div style="font-size:2.5rem;font-weight:800;color:{gad_c};margin:0.4rem 0;">
                {st.session_state.live_gad_total}<span style="font-size:1.3rem;color:#666;"> / 21</span>
            </div>
            <div style="font-size:1rem;font-weight:600;color:{gad_c};
                 background:rgba(255,255,255,0.7);display:inline-block;
                 padding:0.2rem 0.6rem;border-radius:6px;">
                {gad_l}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    with st.form(key="screening_form"):
        tab1, tab2, tab3 = st.tabs(["Demographics", "PHQ-8", "GAD-7"])

        # Demographics
        with tab1:
            st.markdown("### School & Personal Information")
            col1, col2 = st.columns(2)
            with col1:
                boarding_day = st.selectbox("School Type", ["Boarding", "Day"], key="boarding_day")
                school_type   = st.selectbox("School Gender", ["Boys", "Girls", "Mixed"], key="school_type")
                school_demo   = st.selectbox("School Level", ['Subcounty', 'Extracounty', 'County'], key="school_demo")
                school_county = st.selectbox("County", ["Nairobi","Kiambu","Makueni","Machakos"], key="school_county")
                age           = st.slider("Age", 12, 25, 16, key="age")
                gender        = st.selectbox("Gender", ["Male", "Female"], key="gender")
            with col2:
                form          = st.selectbox("Form", [1,2,3,4], key="form")
                religion      = st.selectbox("Religion", ["Christian", "Muslim", "Other"], key="religion")
                parents_home  = st.selectbox("Parents at Home", ["Both parents", "One parent", "None"], key="parents_home")
                parents_dead  = st.number_input("Deceased Parents", 0, 4, 0, key="parents_dead")
                fathers_edu   = st.selectbox("Father's Education", ["None","Primary","Secondary","Tertiary","University"], key="fathers_edu")
                mothers_edu   = st.selectbox("Mother's Education", ["None","Primary","Secondary","Tertiary","University"], key="mothers_edu")
            col3, col4, col5 = st.columns(3)
            with col3: co_curr = st.selectbox("Co-curricular", ["Yes", "No"], key="co_curr")
            with col4: sports  = st.selectbox("Sports", ["Yes", "No"], key="sports")
            with col5: acad_ability = st.slider("Academic Self-Rating", 1, 5, 3, key="acad_ability")

        # PHQ-8
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
                    phq[f'PHQ_{i}'] = st.select_slider(
                        f"p{i}", options=[0,1,2,3],
                        format_func=lambda x: likert[x],
                        label_visibility="collapsed",
                        key=f"phq_live_{i}",
                        on_change=update_live_phq
                    )
                st.markdown("---")

        # GAD-7
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
                    gad[f'GAD_{i}'] = st.select_slider(
                        f"g{i}", options=[0,1,2,3],
                        format_func=lambda x: likert[x],
                        label_visibility="collapsed",
                        key=f"gad_live_{i}",
                        on_change=update_live_gad
                    )
                st.markdown("---")

        # Submit
        st.markdown("---")
        _, c, _ = st.columns([1,2,1])
        with c:
            submitted = st.form_submit_button("Run Screening", use_container_width=True)

        # When the user clicks Submit – copy live totals
        if submitted:
            st.session_state.submitted = True
            st.session_state.phq_total = st.session_state.live_phq_total
            st.session_state.gad_total = st.session_state.live_gad_total

            # Build the final input dictionary
            st.session_state.input_data = {
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
                "Fathers_Education": {"None":0, "Primary":1, "Secondary":2, "Tertiary":3, "University":4}.get(fathers_edu, 0),
                "Mothers_Education": {"None":0, "Primary":1, "Secondary":2, "Tertiary":3, "University":4}.get(mothers_edu, 0),
                "Co_Curricular": 1 if co_curr == "Yes" else 0,
                "Sports": 1 if sports == "Yes" else 0,
                "Percieved_Academic_Abilities": int(acad_ability)
            }
            st.session_state.input_data.update(phq)
            st.session_state.input_data.update(gad)

            st.rerun()


# ----------------------------------------------------------------------
# Results Section (Only after submission)
# ----------------------------------------------------------------------
else:
    if st.session_state.submitted:
        st.markdown("---")
        st.markdown("## Detailed Risk Reports")

        # --------------------------------------------------------------
        # Determine which models to use
        # --------------------------------------------------------------
        if selection_mode == "Auto-Select Best":
            best_dep_model = dep_name
            best_anx_model = anx_name
        else:   # Manual or View-All
            best_dep_model = st.session_state.get("manual_dep_model")
            best_anx_model = st.session_state.get("manual_anx_model")

        dep_pipe = pipelines.get(best_dep_model)
        anx_pipe = pipelines.get(best_anx_model)

        if not dep_pipe or not anx_pipe:
            st.error("Selected models not found.")
        else:
            # Prepare input
            input_df = pd.DataFrame([st.session_state.input_data])
            X = input_df.copy()

            # ------------------------------------------------------------------
            # SHAP explanations
            # ------------------------------------------------------------------
            try:
                with st.spinner("Generating explanations..."):
                    # Depression
                    dep_explainer = shap.Explainer(
                        dep_pipe.named_steps['classifier'],
                        dep_pipe.named_steps['preprocessor'].transform(X)
                    )
                    dep_shap_values = dep_explainer(
                        dep_pipe.named_steps['preprocessor'].transform(X)
                    )

                    # Anxiety
                    anx_explainer = shap.Explainer(
                        anx_pipe.named_steps['classifier'],
                        anx_pipe.named_steps['preprocessor'].transform(X)
                    )
                    anx_shap_values = anx_explainer(
                        anx_pipe.named_steps['preprocessor'].transform(X)
                    )
            except Exception as e:
                st.warning(f"SHAP explanation failed: {e}")
                dep_shap_values = None
                anx_shap_values = None

            # ------------------------------------------------------------------
            # Predictions
            # ------------------------------------------------------------------
            dep_pred = dep_pipe.predict(X)[0]
            anx_pred = anx_pipe.predict(X)[0]

            # ------------------------------------------------------------------
            # Category labels
            # ------------------------------------------------------------------
            def get_dep_cat(score):
                if score < 5: return "Minimal"
                if score < 10: return "Mild"
                if score < 15: return "Moderate"
                if score < 20: return "Moderately Severe"
                return "Severe"

            def get_anx_cat(score):
                if score < 5: return "Minimal"
                if score < 10: return "Mild"
                if score < 15: return "Moderate"
                return "Severe"

            dep_cat = get_dep_cat(st.session_state.phq_total)
            anx_cat = get_anx_cat(st.session_state.gad_total)

            # ------------------------------------------------------------------
            # DEPRESSION REPORT
            # ------------------------------------------------------------------
            st.markdown("### Depression Risk Breakdown")
            col_d1, col_d2 = st.columns([1, 2])

            with col_d1:
                st.markdown(f"""
                <div class="score-card" style="text-align:center; padding:1.5rem;">
                    <h3 style="margin:0; color:#c0392b;">PHQ-8 Score</h3>
                    <div style="font-size:3rem; font-weight:800; color:#c0392b;">
                        {st.session_state.phq_total}<small style="font-size:1.5rem;">/24</small>
                    </div>
                    <div style="font-size:1.1rem; font-weight:600; color:#c0392b;">
                        {dep_cat}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Key Demographics
                st.markdown("**Key Risk Factors**")
                demo_items = [
                    ("Age", X.iloc[0]['Age']),
                    ("Gender", "Male" if X.iloc[0]['Gender'] == 1 else "Female"),
                    ("School Type", X.iloc[0]['Boarding_day']),
                    ("Parents at Home", {0:"None",1:"One",2:"Both"}.get(X.iloc[0]['Parents_Home'], "Unknown")),
                    ("Academic Self-Rating", X.iloc[0]['Percieved_Academic_Abilities'])
                ]
                for label, value in demo_items:
                    st.markdown(f"- **{label}**: {value}")

            with col_d2:
                if dep_shap_values is not None:
                    st.markdown("**How Each Factor Contributes**")
                    fig, ax = plt.subplots()
                    shap.plots.waterfall(dep_shap_values[0], max_display=10, show=False)
                    st.pyplot(fig, use_container_width=True)
                    plt.clf()
                else:
                    st.info("SHAP visualization unavailable.")

            # ------------------------------------------------------------------
            # ANXIETY REPORT
            # ------------------------------------------------------------------
            st.markdown("### Anxiety Risk Breakdown")
            col_a1, col_a2 = st.columns([1, 2])

            with col_a1:
                st.markdown(f"""
                <div class="score-card" style="text-align:center; padding:1.5rem;">
                    <h3 style="margin:0; color:#e67e22;">GAD-7 Score</h3>
                    <div style="font-size:3rem; font-weight:800; color:#e67e22;">
                        {st.session_state.gad_total}<small style="font-size:1.5rem;">/21</small>
                    </div>
                    <div style="font-size:1.1rem; font-weight:600; color:#e67e22;">
                        {anx_cat}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Key Risk Factors**")
                for label, value in demo_items:
                    st.markdown(f"- **{label}**: {value}")

            with col_a2:
                if anx_shap_values is not None:
                    st.markdown("**How Each Factor Contributes**")
                    fig, ax = plt.subplots()
                    shap.plots.waterfall(anx_shap_values[0], max_display=10, show=False)
                    st.pyplot(fig, use_container_width=True)
                    plt.clf()
                else:
                    st.info("SHAP visualization unavailable.")

            # ------------------------------------------------------------------
            # FINAL SUMMARY
            # ------------------------------------------------------------------
            st.markdown("---")
            st.markdown("## Final Interpretation")
            risk_level = "High" if (dep_pred == 1 or anx_pred == 1) else "Low"
            color = "#c0392b" if risk_level == "High" else "#27ae60"

            st.markdown(f"""
            <div style="text-align:center; padding:2rem; background:linear-gradient(135deg, {color}20, {color}10); border-radius:16px; border:2px solid {color};">
                <h2 style="color:{color}; margin:0;">Overall Risk: <strong>{risk_level}</strong></h2>
                <p style="font-size:1.1rem; color:#333; margin:1rem 0;">
                    { "Immediate support recommended." if risk_level == "High" else "Continue monitoring and support." }
                </p>
            </div>
            """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# FULL RISK REPORT + PDF DOWNLOAD + CRISIS HOTLINE
# ----------------------------------------------------------------------
if st.session_state.submitted:
    st.markdown("---")
    st.markdown("## Full Risk Report")

    st.markdown("### Download Report")
    if st.button("Generate PDF Report", key="pdf_gen"):
        with st.spinner("Creating PDF..."):
            import pdfkit
            from datetime import datetime

            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial; padding: 20px; }}
                    .header {{ text-align: center; color: #1f77b4; }}
                    .score {{ font-size: 2rem; font-weight: bold; }}
                    .risk-high {{ color: #c0392b; }} .risk-low {{ color: #27ae60; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>AdolescentMind Screening Report</h1>
                    <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
                <hr>
                <h2>Depression (PHQ-8)</h2>
                <p class="score">Score: {st.session_state.phq_total}/24 → <strong>{dep_cat}</strong></p>
                <h2>Anxiety (GAD-7)</h2>
                <p class="score">Score: {st.session_state.gad_total}/21 → <strong>{anx_cat}</strong></p>
                <h2>Final Risk Level</h2>
                <p class="score {'risk-high' if risk_level == 'High' else 'risk-low'}">
                    <strong>{risk_level} Risk</strong>
                </p>
                <hr>
                <p><em>This is a screening tool, not a diagnosis. Consult a professional.</em></p>
            </body>
            </html>
            """

            pdf = pdfkit.from_string(html_content, False)
            st.download_button(
                label="Download PDF",
                data=pdf,
                file_name=f"AdolescentMind_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )

    if risk_level == "High":
        st.markdown("---")
        st.markdown("""
        <div style="background:#ffebee; padding:1.5rem; border-radius:12px; border-left:6px solid #c0392b;">
            <h3 style="color:#c0392b; margin-top:0;">Immediate Support Needed</h3>
            <p><strong>Kenya Mental Health Helpline:</strong> <a href="tel:1199">1199</a></p>
            <p><strong>Befrienders Kenya:</strong> <a href="tel:+254722178177">0722 178 177</a></p>
            <p><em>You're not alone. Help is available 24/7.</em></p>
        </div>
        """, unsafe_allow_html=True)