import streamlit as st
import pandas as pd
import pickle, os, numpy as np
import shap, matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import base64

st.set_page_config(
    page_title="Mental Health Screening - Kenya", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Base directory
BASE = os.path.dirname(__file__)

# Function to load and encode image as base64
def get_base64_image(image_path):
    """Convert image to base64 for CSS background"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Image paths
BG_IMAGE_PATH = os.path.join(BASE, "images", "background.jpg")
DEPRESSION_CARD_BG = os.path.join(BASE, "images", "depression_bg.jpg")
ANXIETY_CARD_BG = os.path.join(BASE, "images", "anxiety_bg.jpg")

# Get base64 images
bg_image = get_base64_image(BG_IMAGE_PATH)
dep_card_bg = get_base64_image(DEPRESSION_CARD_BG)
anx_card_bg = get_base64_image(ANXIETY_CARD_BG)

# Custom CSS with background images
bg_style = ""
if bg_image:
    bg_style = f"""
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    """

dep_card_style = ""
if dep_card_bg:
    dep_card_style = f"""
    .depression-card {{
        background-image: url("data:image/jpg;base64,{dep_card_bg}") !important;
        background-size: cover;
        background-position: center;
    }}
    """

anx_card_style = ""
if anx_card_bg:
    anx_card_style = f"""
    .anxiety-card {{
        background-image: url("data:image/jpg;base64,{anx_card_bg}") !important;
        background-size: cover;
        background-position: center;
    }}
    """

st.markdown(f"""
    <style>
    {bg_style}
    
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        background: rgba(255,255,255,0.9);
        padding: 1rem;
        border-radius: 10px;
    }}
    .score-card {{
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }}
    
    {dep_card_style}
    {anx_card_style}
    
    .card-overlay {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255,255,255,0.85);
        z-index: 1;
    }}
    
    .card-content {{
        position: relative;
        z-index: 2;
    }}
    
    .score-number {{
        font-size: 4rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }}
    .score-label {{
        font-size: 1.2rem;
        font-weight: 600;
    }}
    .best-model-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }}
    .stButton>button {{
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: #155a8a;
        transform: scale(1.02);
    }}
    
    /* Make content boxes semi-transparent for background visibility */
    .block-container {{
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 2rem;
    }}
    </style>
""", unsafe_allow_html=True)

# Header with optional custom logo
LOGO_PATH = os.path.join(BASE, "images", "logo.png")

# Display logo if available
if os.path.exists(LOGO_PATH):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(LOGO_PATH, use_container_width=True)
else:
    st.markdown('<div class="main-header">🧠 Mental Health Screening Tool</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-header">Depression & Anxiety Assessment for Kenyan High School Students</div>', unsafe_allow_html=True)

PIPELINE_FILE = os.path.join(BASE, "trained_pipelines.pkl")
METRICS_FILE = os.path.join(BASE, "model_metrics.pkl")

if os.path.exists(PIPELINE_FILE):
    with open(PIPELINE_FILE, "rb") as f:
        pipelines = pickle.load(f)
else:
    pipelines = {}

if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "rb") as f:
        model_metrics = pickle.load(f)
else:
    model_metrics = {}

# Sidebar - Enhanced with best model display
with st.sidebar:
    st.title("⚙️ Settings")
    
    selection_mode = st.radio(
        "🤖 Model Selection:",
        ["Auto-Select Best", "Manual Selection", "View All Models"],
        help="Choose how models should be selected for predictions"
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
    
    # Show best models for each target
    if model_metrics:
        st.markdown("### 🏆 Best Models by Recall")
        
        # Find best depression model
        best_dep_recall = -1
        best_dep_model_info = None
        for name, metrics in model_metrics.items():
            if 'test_recall_per_target' in metrics:
                dep_recall = metrics['test_recall_per_target'].get('Is_Depressed', 0)
                if dep_recall > best_dep_recall:
                    best_dep_recall = dep_recall
                    best_dep_model_info = (name, metrics)
        
        # Find best anxiety model
        best_anx_recall = -1
        best_anx_model_info = None
        for name, metrics in model_metrics.items():
            if 'test_recall_per_target' in metrics:
                anx_recall = metrics['test_recall_per_target'].get('Has_anxiety', 0)
                if anx_recall > best_anx_recall:
                    best_anx_recall = anx_recall
                    best_anx_model_info = (name, metrics)
        
        # Display Depression Best Model
        if best_dep_model_info:
            name, metrics = best_dep_model_info
            st.markdown("#### 🧠 Depression")
            st.markdown(f"""
            <div class="best-model-card">
                <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">Selected Model</div>
                <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">{name}</div>
                <div style="display: flex; justify-content: space-around; margin-top: 0.5rem;">
                    <div>
                        <div style="font-size: 0.8rem; opacity: 0.9;">Recall</div>
                        <div style="font-size: 1.5rem; font-weight: bold;">{metrics['test_recall_per_target']['Is_Depressed']:.1%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; opacity: 0.9;">Accuracy</div>
                        <div style="font-size: 1.5rem; font-weight: bold;">{metrics['test_accuracy_per_target']['Is_Depressed']:.1%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Display Anxiety Best Model
        if best_anx_model_info:
            name, metrics = best_anx_model_info
            st.markdown("#### 😰 Anxiety")
            st.markdown(f"""
            <div class="best-model-card">
                <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">Selected Model</div>
                <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">{name}</div>
                <div style="display: flex; justify-content: space-around; margin-top: 0.5rem;">
                    <div>
                        <div style="font-size: 0.8rem; opacity: 0.9;">Recall</div>
                        <div style="font-size: 1.5rem; font-weight: bold;">{metrics['test_recall_per_target']['Has_anxiety']:.1%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; opacity: 0.9;">Accuracy</div>
                        <div style="font-size: 1.5rem; font-weight: bold;">{metrics['test_accuracy_per_target']['Has_anxiety']:.1%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Show technical details in expander
    with st.expander("📊 All Models Comparison", expanded=False):
        if model_metrics:
            view_metric = st.selectbox(
                "View Metric:",
                ["Recall", "Accuracy"],
                key="sidebar_metric"
            )
            
            metric_key = 'test_recall_per_target' if view_metric == "Recall" else 'test_accuracy_per_target'
            
            dep_scores = []
            anx_scores = []
            model_names = []
            
            for m, vals in model_metrics.items():
                if metric_key in vals:
                    model_names.append(m)
                    dep_scores.append(vals[metric_key].get('Is_Depressed', 0) * 100)
                    anx_scores.append(vals[metric_key].get('Has_anxiety', 0) * 100)
            
            if model_names:
                fig = go.Figure(data=[
                    go.Bar(name='Depression', x=model_names, y=dep_scores, marker_color='#1f77b4'),
                    go.Bar(name='Anxiety', x=model_names, y=anx_scores, marker_color='#ff7f0e')
                ])
                fig.update_layout(
                    barmode='group',
                    height=300,
                    title=f"Model {view_metric} (%)",
                    yaxis_title=f"{view_metric} (%)",
                    showlegend=True,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.info("💡 **Note:** Recall measures how well the model identifies people who need support.")

# Main content - Assessment Form
st.markdown("## 📝 Complete the Assessment")

tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🧠 Depression (PHQ-8)", "😰 Anxiety (GAD-7)"])

with tab1:
    st.markdown("### School & Personal Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        boarding_day = st.selectbox("🏫 School Type", ["Boarding", "Day"])
        school_type = st.selectbox("👥 School Gender", ["Boys", "Girls", "Mixed"])
        school_demo = st.selectbox("🏙️ School Location", ["Urban", "Rural", "Semi-urban"])
        school_county = st.selectbox("📍 County", ["Nairobi","Kiambu","Kisumu","Mombasa","Nakuru","Other"])
        age = st.slider("🎂 Age", min_value=12, max_value=25, value=16)
        gender = st.selectbox("⚧ Gender", ["Male", "Female"])
    
    with col2:
        form = st.selectbox("📚 Form", [1,2,3,4])
        religion = st.selectbox("✝️ Religion", ["Christian", "Muslim", "Other"])
        parents_home = st.selectbox("🏠 Parents Living Situation", ["Both parents", "One parent", "None"])
        parents_dead = st.number_input("💔 Deceased Parents", min_value=0, max_value=4, value=0)
        fathers_edu = st.selectbox("👨‍🎓 Father's Education", ["None","Primary","Secondary","Tertiary","University"])
        mothers_edu = st.selectbox("👩‍🎓 Mother's Education", ["None","Primary","Secondary","Tertiary","University"])
    
    col3, col4, col5 = st.columns(3)
    with col3:
        co_curr = st.selectbox("🎭 Co-curricular Activities", ["Yes", "No"])
    with col4:
        sports = st.selectbox("⚽ Sports Participation", ["Yes", "No"])
    with col5:
        acad_ability = st.slider("📖 Academic Self-Rating", 1, 5, 3, help="1=Low, 5=High")

with tab2:
    st.markdown("### Depression Screening (PHQ-8)")
    st.markdown("**Over the past 2 weeks, how often have you been bothered by the following?**")
    
    phq_qs = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself — or that you are a failure",
        "Trouble concentrating on things, such as homework or studying",
        "Moving or speaking so slowly, or being fidgety/restless"
    ]
    
    phq = {}
    likert_options = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
    likert_values = [0, 1, 2, 3]
    
    for i, q in enumerate(phq_qs, 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{i}.** {q}")
            with col2:
                phq[f'PHQ_{i}'] = st.select_slider(
                    f"PHQ_{i}",
                    options=likert_values,
                    format_func=lambda x: likert_options[x],
                    key=f"phq_{i}",
                    label_visibility="collapsed"
                )
        st.markdown("---")
    
    phq_total = sum(phq.values())
    st.markdown(f"### Current PHQ-8 Score: **{phq_total}** / 24")
    
    if phq_total < 5:
        st.success("✅ Minimal depression symptoms")
    elif phq_total < 10:
        st.info("ℹ️ Mild depression symptoms")
    elif phq_total < 15:
        st.warning("⚠️ Moderate depression symptoms")
    elif phq_total < 20:
        st.warning("⚠️ Moderately severe depression symptoms")
    else:
        st.error("🚨 Severe depression symptoms")

with tab3:
    st.markdown("### Anxiety Screening (GAD-7)")
    st.markdown("**Over the past 2 weeks, how often have you been bothered by the following?**")
    
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
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{i}.** {q}")
            with col2:
                gad[f'GAD_{i}'] = st.select_slider(
                    f"GAD_{i}",
                    options=likert_values,
                    format_func=lambda x: likert_options[x],
                    key=f"gad_{i}",
                    label_visibility="collapsed"
                )
        st.markdown("---")
    
    gad_total = sum(gad.values())
    st.markdown(f"### Current GAD-7 Score: **{gad_total}** / 21")
    
    if gad_total < 5:
        st.success("✅ Minimal anxiety symptoms")
    elif gad_total < 10:
        st.info("ℹ️ Mild anxiety symptoms")
    elif gad_total < 15:
        st.warning("⚠️ Moderate anxiety symptoms")
    else:
        st.error("🚨 Severe anxiety symptoms")

# Submit button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submitted = st.button("🔍 Analyze Mental Health Status", use_container_width=True)

def get_severity_info(score, max_score, assessment_type):
    """Get severity level, color, and recommendations based on validated clinical guidelines"""
    
    if assessment_type == 'depression':
        if score < 5:
            return {
                'level': 'Minimal',
                'color': '#28a745',
                'description': 'Your symptoms are minimal and not significantly impacting daily life.',
                'recommendations': [
                    'Maintain healthy lifestyle habits',
                    'Continue engaging in activities you enjoy',
                    'Stay connected with friends and family',
                    'Practice good sleep hygiene'
                ],
                'source': 'PHQ-8 Interpretation: Kroenke et al. (2009). The PHQ-8 as a measure of current depression in the general population. Journal of Affective Disorders.'
            }
        elif score < 10:
            return {
                'level': 'Mild',
                'color': '#ffc107',
                'description': 'You are experiencing mild symptoms that may benefit from self-care strategies.',
                'recommendations': [
                    'Engage in regular physical exercise',
                    'Practice relaxation techniques (meditation, deep breathing)',
                    'Maintain a regular sleep schedule',
                    'Talk to someone you trust about how you feel',
                    'Consider speaking with a school counselor'
                ],
                'source': 'PHQ-8 Interpretation: Kroenke et al. (2009) & WHO Mental Health Gap Action Programme (mhGAP) guidelines for mild depression management.'
            }
        elif score < 15:
            return {
                'level': 'Moderate',
                'color': '#fd7e14',
                'description': 'You are experiencing moderate symptoms. Professional support is recommended.',
                'recommendations': [
                    '🏥 Speak with a mental health professional or counselor',
                    'Continue self-care practices',
                    'Inform a trusted adult or family member',
                    'Consider therapy or counseling services',
                    'Avoid isolation - stay connected with others'
                ],
                'source': 'PHQ-8 Interpretation: Kroenke et al. (2009) & National Institute for Health and Care Excellence (NICE) guidelines for moderate depression.'
            }
        elif score < 20:
            return {
                'level': 'Moderately Severe',
                'color': '#dc3545',
                'description': 'You are experiencing moderately severe symptoms. Professional help is strongly recommended.',
                'recommendations': [
                    '🏥 Seek professional evaluation from a healthcare provider',
                    'Contact your school counselor or guidance office',
                    'Inform your parents or guardian',
                    'Professional therapy is recommended',
                    'Do not face this alone - reach out for support'
                ],
                'source': 'PHQ-8 Interpretation: Kroenke et al. (2009) & American Psychological Association (APA) practice guidelines for moderately severe depression.'
            }
        else:
            return {
                'level': 'Severe',
                'color': '#bd2130',
                'description': 'You are experiencing severe symptoms. Immediate professional evaluation is needed.',
                'recommendations': [
                    '🚨 Seek immediate professional evaluation',
                    'Contact a mental health professional or healthcare provider',
                    'Inform your parents/guardians immediately',
                    'Kenya Red Cross: 1199',
                    'Befrienders Kenya: +254 722 178 177'
                ],
                'source': 'PHQ-8 Interpretation: Kroenke et al. (2009) & WHO mhGAP guidelines for severe depression requiring immediate clinical attention.'
            }
    else:  # anxiety
        if score < 5:
            return {
                'level': 'Minimal',
                'color': '#28a745',
                'description': 'Your anxiety symptoms are minimal.',
                'recommendations': [
                    'Continue healthy stress management practices',
                    'Maintain regular exercise routine',
                    'Practice mindfulness or meditation',
                    'Get adequate sleep'
                ],
                'source': 'GAD-7 Interpretation: Spitzer et al. (2006). A brief measure for assessing generalized anxiety disorder. Archives of Internal Medicine.'
            }
        elif score < 10:
            return {
                'level': 'Mild',
                'color': '#ffc107',
                'description': 'You are experiencing mild anxiety that may respond to self-management strategies.',
                'recommendations': [
                    'Practice deep breathing exercises',
                    'Try progressive muscle relaxation',
                    'Limit caffeine intake',
                    'Maintain regular physical activity',
                    'Talk to someone you trust'
                ],
                'source': 'GAD-7 Interpretation: Spitzer et al. (2006) & NICE guidelines for mild anxiety management through psychoeducation and self-help.'
            }
        elif score < 15:
            return {
                'level': 'Moderate',
                'color': '#fd7e14',
                'description': 'You are experiencing moderate anxiety. Professional guidance is recommended.',
                'recommendations': [
                    '🏥 Consider speaking with a mental health professional',
                    'Learn and practice anxiety management techniques',
                    'Identify and address anxiety triggers',
                    'Maintain a worry journal',
                    'Join a support group if available'
                ],
                'source': 'GAD-7 Interpretation: Spitzer et al. (2006) & NICE guidelines recommending psychological interventions for moderate anxiety.'
            }
        else:
            return {
                'level': 'Severe',
                'color': '#dc3545',
                'description': 'You are experiencing severe anxiety. Professional evaluation is strongly recommended.',
                'recommendations': [
                    '🏥 Seek professional evaluation as soon as possible',
                    'Contact your school counselor immediately',
                    'Inform your parents or guardian',
                    'Professional evaluation and support is recommended',
                    'Practice grounding techniques during anxiety episodes',
                    'Kenya Red Cross: 1199'
                ],
                'source': 'GAD-7 Interpretation: Spitzer et al. (2006) & WHO mhGAP guidelines for severe anxiety requiring clinical evaluation.'
            }

if submitted:
    with st.spinner("🤖 Analyzing your responses..."):
        input_data = {
            "Boarding_day": boarding_day,
            "School_type": school_type,
            "School_Demographics": school_demo,
            "School_County": school_county,
            "Age": age,
            "Gender": 1 if gender == "Male" else 2,
            "Form": form,
            "Religion": 1 if religion == "Christian" else 2 if religion == "Muslim" else 3,
            "Parents_Home": {"None":0, "One parent":1, "Both parents":2}[parents_home],
            "Parents_Dead": parents_dead,
            "Fathers_Education": fathers_edu,
            "Mothers_Education": mothers_edu,
            "Co_Curricular": 1 if co_curr == "Yes" else 0,
            "Sports": 1 if sports == "Yes" else 0,
            "Percieved_Academic_Abilities": acad_ability
        }
        input_data.update(phq)
        input_data.update(gad)
        user_df = pd.DataFrame([input_data])

        # Make predictions with ALL models
        all_predictions = {}
        for name, pipe in pipelines.items():
            try:
                pred = pipe.predict(user_df)[0]
                all_predictions[name] = {
                    'depression_pred': pred[0],
                    'anxiety_pred': pred[1]
                }
            except Exception as e:
                all_predictions[name] = {
                    'depression_pred': None,
                    'anxiety_pred': None
                }

        # Model selection
        if selection_mode == "Manual Selection":
            best_dep_model = manual_dep_model
            best_anx_model = manual_anx_model
            best_dep_recall = model_metrics.get(best_dep_model, {}).get('test_recall_per_target', {}).get('Is_Depressed', 0)
            best_anx_recall = model_metrics.get(best_anx_model, {}).get('test_recall_per_target', {}).get('Has_anxiety', 0)
            best_dep_acc = model_metrics.get(best_dep_model, {}).get('test_accuracy_per_target', {}).get('Is_Depressed', 0)
            best_anx_acc = model_metrics.get(best_anx_model, {}).get('test_accuracy_per_target', {}).get('Has_anxiety', 0)
        else:
            # Auto-select best models
            best_dep_model = max(
                model_metrics.items(),
                key=lambda x: x[1].get('test_recall_per_target', {}).get('Is_Depressed', 0)
            )[0] if model_metrics else None
            
            best_anx_model = max(
                model_metrics.items(),
                key=lambda x: x[1].get('test_recall_per_target', {}).get('Has_anxiety', 0)
            )[0] if model_metrics else None
            
            best_dep_recall = model_metrics.get(best_dep_model, {}).get('test_recall_per_target', {}).get('Is_Depressed', 0)
            best_anx_recall = model_metrics.get(best_anx_model, {}).get('test_recall_per_target', {}).get('Has_anxiety', 0)
            best_dep_acc = model_metrics.get(best_dep_model, {}).get('test_accuracy_per_target', {}).get('Is_Depressed', 0)
            best_anx_acc = model_metrics.get(best_anx_model, {}).get('test_accuracy_per_target', {}).get('Has_anxiety', 0)

        depression_prediction = all_predictions.get(best_dep_model, {}).get('depression_pred', 'N/A')
        anxiety_prediction = all_predictions.get(best_anx_model, {}).get('anxiety_pred', 'N/A')

        # Get severity information
        dep_info = get_severity_info(phq_total, 24, 'depression')
        anx_info = get_severity_info(gad_total, 21, 'anxiety')

        # Results Display
        st.balloons()
        st.markdown("---")
        st.markdown("## 🎯 Your Mental Health Assessment Results")
        st.markdown("*Based on your responses, here's what we found:*")
        
        # Show which models were used
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"🧠 **Depression Model:** {best_dep_model} (Recall: {best_dep_recall:.1%}, Accuracy: {best_dep_acc:.1%})")
        with col_b:
            st.info(f"😰 **Anxiety Model:** {best_anx_model} (Recall: {best_anx_recall:.1%}, Accuracy: {best_anx_acc:.1%})")
        
        st.markdown("---")

        # Main Results Cards - WITHOUT Recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            card_class = "depression-card" if dep_card_bg else ""
            st.markdown(f"""
            <div class="score-card {card_class}">
                <div class="card-overlay"></div>
                <div class="card-content">
                    <h2 style="margin:0; color: {dep_info['color']}">🧠 Depression Assessment</h2>
                    <div class="score-number" style="color: {dep_info['color']}">{phq_total}<span style="font-size:2rem; color: #666">/24</span></div>
                    <div class="score-label" style="color: {dep_info['color']}">{dep_info['level']}</div>
                    <p style="margin-top: 1rem; font-size: 1rem; color: #333;">{dep_info['description']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📚 Clinical Guidelines Reference"):
                st.caption(f"**Source:** {dep_info['source']}")
                st.caption("**Note:** These are screening recommendations based on standardized PHQ-8 cutoff scores, not clinical diagnoses.")
        
        with col2:
            card_class = "anxiety-card" if anx_card_bg else ""
            st.markdown(f"""
            <div class="score-card {card_class}">
                <div class="card-overlay"></div>
                <div class="card-content">
                    <h2 style="margin:0; color: {anx_info['color']}">😰 Anxiety Assessment</h2>
                    <div class="score-number" style="color: {anx_info['color']}">{gad_total}<span style="font-size:2rem; color: #666">/21</span></div>
                    <div class="score-label" style="color: {anx_info['color']}">{anx_info['level']}</div>
                    <p style="margin-top: 1rem; font-size: 1rem; color: #333;">{anx_info['description']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📚 Clinical Guidelines Reference"):
                st.caption(f"**Source:** {anx_info['source']}")
                st.caption("**Note:** These are screening recommendations based on standardized GAD-7 cutoff scores, not clinical diagnoses.")

        # Recommendations Section - Separate from cards
        st.markdown("---")
        st.markdown("### 💡 Personalized Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧠 Depression Management")
            for rec in dep_info['recommendations']:
                st.markdown(f"- {rec}")
        
        with col2:
            st.markdown("#### 😰 Anxiety Management")
            for rec in anx_info['recommendations']:
                st.markdown(f"- {rec}")

        # SHAP Explanations
        st.markdown("---")
        st.markdown("### 🔍 Understanding Your Results")
        st.markdown("*These charts show which factors had the most influence on your assessment:*")
        
        tab1, tab2 = st.tabs(["🧠 Depression Factors", "😰 Anxiety Factors"])
        
        with tab1:
            try:
                sel_pipe = pipelines[best_dep_model]
                pre = sel_pipe.named_steps['preprocessor']
                clf = sel_pipe.named_steps['clf']
                
                # Transform data
                X_trans = pre.transform(user_df)
                if hasattr(X_trans, 'toarray'):
                    X_trans = X_trans.toarray()
                
                # Ensure float64 type
                X_trans = X_trans.astype(np.float64)
                
                # Get feature names
                try:
                    feature_names = list(pre.get_feature_names_out())
                except:
                    feature_names = [f"Feature_{i}" for i in range(X_trans.shape[1])]
                
                # Extract the depression estimator from MultiOutputClassifier
                if hasattr(clf, 'estimators_'):
                    # MultiOutputClassifier - get first estimator (depression)
                    base_model = clf.estimators_[0]
                else:
                    base_model = clf
                
                # Create explainer based on model type
                model_name = str(type(base_model).__name__).lower()
                
                if any(x in model_name for x in ['logistic', 'linear']):
                    # Linear models
                    explainer = shap.Explainer(base_model, X_trans, feature_names=feature_names)
                    shap_values = explainer(X_trans)
                    
                    # Extract values
                    if hasattr(shap_values, 'values'):
                        vals = shap_values.values
                    else:
                        vals = shap_values
                    
                    # Handle multi-dimensional output
                    if len(vals.shape) > 2:
                        vals = vals[:, :, 1]
                    elif len(vals.shape) == 3:
                        vals = vals[0, :, 1]
                    
                    mean_shap = np.abs(vals).mean(axis=0) if len(vals.shape) > 1 else np.abs(vals)
                    
                else:
                    # Tree-based models
                    explainer = shap.TreeExplainer(base_model)
                    shap_values = explainer.shap_values(X_trans)
                    
                    # Handle different return types
                    if isinstance(shap_values, list):
                        # Binary classification - use positive class
                        vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    else:
                        vals = shap_values
                    
                    # Get mean absolute SHAP values
                    if len(vals.shape) > 1:
                        mean_shap = np.abs(vals).mean(axis=0)
                    else:
                        mean_shap = np.abs(vals)
                
                # Ensure 1D array
                mean_shap = np.array(mean_shap).flatten()
                
                # Get top 10 features
                if len(mean_shap) > 10:
                    top_indices = np.argsort(mean_shap)[-10:][::-1]
                else:
                    top_indices = np.argsort(mean_shap)[::-1]
                
                top_features = [feature_names[i] for i in top_indices]
                top_values = mean_shap[top_indices]
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
                bars = ax.barh(range(len(top_features)), top_values, color=colors)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features, fontsize=10)
                ax.set_xlabel('Average Impact on Prediction', fontsize=11)
                ax.set_title('Top 10 Factors Influencing Depression Assessment', fontsize=13, pad=15, fontweight='bold')
                ax.invert_yaxis()
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, top_values)):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f'{val:.3f}', 
                           ha='left', va='center', fontsize=9, 
                           color='black', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.info("""
                📖 **How to read this chart:** 
                - Longer bars = stronger influence on your depression assessment
                - Higher values mean the feature had more impact on the prediction
                - Features include your survey responses, demographics, and symptoms
                """)
                
            except Exception as e:
                st.warning(f"⚠️ Could not generate explanation chart for depression model.")
                with st.expander("View technical error details"):
                    st.code(f"Error: {str(e)}\nModel type: {type(clf).__name__}")
                st.info("The AI model made predictions successfully, but we couldn't generate the visual explanation. This doesn't affect the accuracy of your results.")
        
        with tab2:
            try:
                sel_pipe = pipelines[best_anx_model]
                pre = sel_pipe.named_steps['preprocessor']
                clf = sel_pipe.named_steps['clf']
                
                # Transform data
                X_trans = pre.transform(user_df)
                if hasattr(X_trans, 'toarray'):
                    X_trans = X_trans.toarray()
                
                # Ensure float64 type
                X_trans = X_trans.astype(np.float64)
                
                # Get feature names
                try:
                    feature_names = list(pre.get_feature_names_out())
                except:
                    feature_names = [f"Feature_{i}" for i in range(X_trans.shape[1])]
                
                # Extract the anxiety estimator from MultiOutputClassifier
                if hasattr(clf, 'estimators_'):
                    # MultiOutputClassifier - get second estimator (anxiety)
                    base_model = clf.estimators_[1]
                else:
                    base_model = clf
                
                # Create explainer based on model type
                model_name = str(type(base_model).__name__).lower()
                
                if any(x in model_name for x in ['logistic', 'linear']):
                    # Linear models
                    explainer = shap.Explainer(base_model, X_trans, feature_names=feature_names)
                    shap_values = explainer(X_trans)
                    
                    # Extract values
                    if hasattr(shap_values, 'values'):
                        vals = shap_values.values
                    else:
                        vals = shap_values
                    
                    # Handle multi-dimensional output
                    if len(vals.shape) > 2:
                        vals = vals[:, :, 1]
                    elif len(vals.shape) == 3:
                        vals = vals[0, :, 1]
                    
                    mean_shap = np.abs(vals).mean(axis=0) if len(vals.shape) > 1 else np.abs(vals)
                    
                else:
                    # Tree-based models
                    explainer = shap.TreeExplainer(base_model)
                    shap_values = explainer.shap_values(X_trans)
                    
                    # Handle different return types
                    if isinstance(shap_values, list):
                        # Binary classification - use positive class
                        vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    else:
                        vals = shap_values
                    
                    # Get mean absolute SHAP values
                    if len(vals.shape) > 1:
                        mean_shap = np.abs(vals).mean(axis=0)
                    else:
                        mean_shap = np.abs(vals)
                
                # Ensure 1D array
                mean_shap = np.array(mean_shap).flatten()
                
                # Get top 10 features
                if len(mean_shap) > 10:
                    top_indices = np.argsort(mean_shap)[-10:][::-1]
                else:
                    top_indices = np.argsort(mean_shap)[::-1]
                
                top_features = [feature_names[i] for i in top_indices]
                top_values = mean_shap[top_indices]
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
                bars = ax.barh(range(len(top_features)), top_values, color=colors)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features, fontsize=10)
                ax.set_xlabel('Average Impact on Prediction', fontsize=11)
                ax.set_title('Top 10 Factors Influencing Anxiety Assessment', fontsize=13, pad=15, fontweight='bold')
                ax.invert_yaxis()
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, top_values)):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f'{val:.3f}', 
                           ha='left', va='center', fontsize=9, 
                           color='black', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.info("""
                📖 **How to read this chart:** 
                - Longer bars = stronger influence on your anxiety assessment
                - Higher values mean the feature had more impact on the prediction
                - Features include your survey responses, demographics, and symptoms
                """)
                
            except Exception as e:
                st.warning(f"⚠️ Could not generate explanation chart for anxiety model.")
                with st.expander("View technical error details"):
                    st.code(f"Error: {str(e)}\nModel type: {type(clf).__name__}")
                st.info("The AI model made predictions successfully, but we couldn't generate the visual explanation. This doesn't affect the accuracy of your results.")

        # Technical details in expander
        if selection_mode == "View All Models":
            st.markdown("---")
            with st.expander("📊 View All Model Predictions", expanded=False):
                st.markdown("### Comparison of All Model Predictions")
                
                dep_categories = {0: 'None', 1: 'Mild', 2: 'Moderate', 3: 'Moderately Severe', 4: 'Severe'}
                anx_categories = {0: 'Minimal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
                
                comparison_data = []
                for name in pipelines.keys():
                    metrics = model_metrics.get(name, {})
                    preds = all_predictions.get(name, {})
                    
                    comparison_data.append({
                        "Model": name,
                        "Depression": dep_categories.get(preds.get('depression_pred', 'N/A'), 'N/A'),
                        "Dep Recall": f"{metrics.get('test_recall_per_target', {}).get('Is_Depressed', 0):.1%}",
                        "Dep Accuracy": f"{metrics.get('test_accuracy_per_target', {}).get('Is_Depressed', 0):.1%}",
                        "Anxiety": anx_categories.get(preds.get('anxiety_pred', 'N/A'), 'N/A'),
                        "Anx Recall": f"{metrics.get('test_recall_per_target', {}).get('Has_anxiety', 0):.1%}",
                        "Anx Accuracy": f"{metrics.get('test_accuracy_per_target', {}).get('Has_anxiety', 0):.1%}"
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
                
                st.info("💡 **Note:** Different models may give different predictions. The selected 'best' models are highlighted at the top of the results.")

        # Download results
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 💾 Save Your Results")
            st.markdown("Download a comprehensive summary of your assessment to share with a healthcare provider or keep for your records.")
        
        with col2:
            results_summary = f"""MENTAL HEALTH SCREENING RESULTS
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Generated by: Mental Health Screening Tool - Kenya

═══════════════════════════════════════════════════════════

DEPRESSION ASSESSMENT (PHQ-8)
Score: {phq_total}/24
Severity Level: {dep_info['level']}
Model Used: {best_dep_model}
Model Performance: Recall {best_dep_recall:.1%}, Accuracy {best_dep_acc:.1%}

What This Means:
{dep_info['description']}

Recommended Actions:
{chr(10).join('• ' + rec for rec in dep_info['recommendations'])}

═══════════════════════════════════════════════════════════

ANXIETY ASSESSMENT (GAD-7)
Score: {gad_total}/21
Severity Level: {anx_info['level']}
Model Used: {best_anx_model}
Model Performance: Recall {best_anx_recall:.1%}, Accuracy {best_anx_acc:.1%}

What This Means:
{anx_info['description']}

Recommended Actions:
{chr(10).join('• ' + rec for rec in anx_info['recommendations'])}

═══════════════════════════════════════════════════════════

CRISIS RESOURCES (Available 24/7)
• Kenya Red Cross: 1199
• Befrienders Kenya: +254 722 178 177
• Your school counselor or guidance department

═══════════════════════════════════════════════════════════

IMPORTANT DISCLAIMER:
This is a screening tool for educational purposes only.
It is NOT a diagnostic instrument.

These results should be discussed with a qualified mental 
health professional. They do not replace professional 
clinical assessment.

If you are experiencing a mental health crisis, please 
seek immediate professional help or call the crisis lines 
listed above.

═══════════════════════════════════════════════════════════
"""
            
            st.download_button(
                label="📥 Download Full Report",
                data=results_summary,
                file_name=f"mental_health_screening_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        # Crisis resources - Always visible for high scores
        if phq_total >= 15 or gad_total >= 15:
            st.markdown("---")
            st.error("""
            ### 🚨 IMMEDIATE SUPPORT NEEDED
            
            Your scores indicate significant symptoms. **Please reach out for help immediately.**
            
            #### Crisis Resources in Kenya:
            
            - 🚑 **Kenya Red Cross:** 1199
            - 📞 **Befrienders Kenya:** +254 722 178 177
            - 🏥 **Your school counselor or guidance department**
            - 👨‍👩‍👧 **A trusted adult, teacher, or family member**
            - 🏥 **Nearest hospital emergency department**
            
            #### What to do right now:
            
            1. **Don't wait** - Reach out to one of the resources above
            2. **Tell someone** - Share how you're feeling with a trusted person
            3. **Stay safe** - If you're having thoughts of self-harm, go to the nearest emergency room
            4. **You are not alone** - Help is available and recovery is possible
            
            **Remember:** Seeking help is a sign of strength, not weakness. Many people have been where you are and have found their way to feeling better with the right support.
            """)

        # Final disclaimer - Always visible
        st.markdown("---")
        st.warning("""
        ### ⚠️ IMPORTANT DISCLAIMER
        
        **This tool is for screening and educational purposes only.** It is **NOT** a diagnostic instrument.
        
        #### Understanding Your Results:
        
        - ✅ These scores provide an **indication** of symptom severity based on validated screening questionnaires
        - ✅ Results follow **standardized** PHQ-8 and GAD-7 scoring interpretations
        - ✅ Should be **discussed** with a qualified mental health professional
        - ❌ Do **NOT replace** professional clinical assessment or diagnosis
        - ❌ Should **NOT** be used for self-diagnosis
        
        #### Recommended Next Steps:
        
        1. 🏥 **Share these results** with a healthcare provider or school counselor for proper evaluation
        2. 📋 **Use as a starting point** for a conversation about your mental health
        3. 🔄 **Re-screen periodically** to monitor symptom changes (recommended every 2-4 weeks)
        4. 💬 **Talk to someone** you trust about how you're feeling
        5. 📚 **Seek professional evaluation** if symptoms persist or worsen
        
        #### If You Are Experiencing a Crisis:
        
        - **Kenya Red Cross:** 1199 (24/7)
        - **Befrienders Kenya:** +254 722 178 177 (24/7)
        - **Your school counselor** or guidance department
        - **Nearest hospital** emergency department
        """)