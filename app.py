import streamlit as st
import pandas as pd
import pickle, os, numpy as np
import shap, matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="AdolescentMind - Kenya", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="icon.jpg"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .score-number {
        font-size: 4rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .score-label {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .contributing-factor {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        background: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #155a8a;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">AdolescentMind - Kenya</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Depression & Anxiety Assessment for Kenyan High School Students</div>', unsafe_allow_html=True)

BASE = os.path.dirname(__file__)
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

# Sidebar - Moved technical details here
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
    
    # Show technical details in expander
    with st.expander("📊 Technical Details", expanded=False):
        if model_metrics:
            view_metric = st.selectbox(
                "View Metric:",
                ["Recall", "Accuracy"]
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
    """Get severity level, color, and recommendations"""
    percentage = (score / max_score) * 100
    
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
                ]
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
                ]
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
                ]
            }
        elif score < 20:
            return {
                'level': 'Moderately Severe',
                'color': '#dc3545',
                'description': 'You are experiencing moderately severe symptoms. Professional help is strongly recommended.',
                'recommendations': [
                    '🏥 **Seek professional help immediately**',
                    'Contact your school counselor or guidance office',
                    'Inform your parents or guardian',
                    'Professional therapy is recommended',
                    'Do not face this alone - reach out for support'
                ]
            }
        else:
            return {
                'level': 'Severe',
                'color': '#bd2130',
                'description': 'You are experiencing severe symptoms. Immediate professional intervention is needed.',
                'recommendations': [
                    '🚨 **Seek immediate professional help**',
                    'Contact a mental health crisis line',
                    'Visit a healthcare facility',
                    'Inform your parents/guardians immediately',
                    'Kenya Red Cross: 1199',
                    'Befrienders Kenya: +254 722 178 177'
                ]
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
                ]
            }
        elif score < 10:
            return {
                'level': 'Mild',
                'color': '#ffc107',
                'description': 'You are experiencing mild anxiety that may respond to relaxation techniques.',
                'recommendations': [
                    'Practice deep breathing exercises',
                    'Try progressive muscle relaxation',
                    'Limit caffeine intake',
                    'Maintain regular physical activity',
                    'Talk to someone you trust'
                ]
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
                ]
            }
        else:
            return {
                'level': 'Severe',
                'color': '#dc3545',
                'description': 'You are experiencing severe anxiety. Professional support is strongly recommended.',
                'recommendations': [
                    '🏥 **Seek professional help as soon as possible**',
                    'Contact your school counselor immediately',
                    'Inform your parents or guardian',
                    'Professional therapy or treatment is needed',
                    'Practice grounding techniques during anxiety episodes',
                    'Kenya Red Cross: 1199'
                ]
            }
    
    return info

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
        st.markdown("---")

        # Main Results Cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="score-card" style="background: linear-gradient(135deg, {dep_info['color']}15 0%, {dep_info['color']}30 100%); border-left: 5px solid {dep_info['color']}">
                <h2 style="margin:0; color: {dep_info['color']}">🧠 Depression Assessment</h2>
                <div class="score-number" style="color: {dep_info['color']}">{phq_total}<span style="font-size:2rem; color: #666">/24</span></div>
                <div class="score-label" style="color: {dep_info['color']}">{dep_info['level']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**What this means:** {dep_info['description']}")
            
            st.markdown("#### 💡 Recommended Actions:")
            for rec in dep_info['recommendations']:
                st.markdown(f"- {rec}")
        
        with col2:
            st.markdown(f"""
            <div class="score-card" style="background: linear-gradient(135deg, {anx_info['color']}15 0%, {anx_info['color']}30 100%); border-left: 5px solid {anx_info['color']}">
                <h2 style="margin:0; color: {anx_info['color']}">😰 Anxiety Assessment</h2>
                <div class="score-number" style="color: {anx_info['color']}">{gad_total}<span style="font-size:2rem; color: #666">/21</span></div>
                <div class="score-label" style="color: {anx_info['color']}">{anx_info['level']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**What this means:** {anx_info['description']}")
            
            st.markdown("#### 💡 Recommended Actions:")
            for rec in anx_info['recommendations']:
                st.markdown(f"- {rec}")

        # SHAP Explanations - Fixed implementation
        st.markdown("---")
        st.markdown("### 🔍 Understanding Your Results")
        st.markdown("*These charts show which factors had the most influence on your assessment:*")
        
        tab1, tab2 = st.tabs(["🧠 Depression Factors", "😰 Anxiety Factors"])
        
        with tab1:
            try:
                sel_pipe = pipelines[best_dep_model]
                pre = sel_pipe.named_steps['preprocessor']
                clf = sel_pipe.named_steps['clf']
                
                # Transform input
                X_trans = pre.transform(user_df)
                
                # Get feature names after transformation
                try:
                    feature_names = pre.get_feature_names_out()
                except:
                    feature_names = [f"Feature_{i}" for i in range(X_trans.shape[1])]
                
                # Get the depression estimator
                if hasattr(clf, 'estimators_'):
                    base_model = clf.estimators_[0]  # Depression is first target
                else:
                    base_model = clf
                
                # Create SHAP explainer
                explainer = shap.TreeExplainer(base_model)
                shap_values = explainer.shap_values(X_trans)
                
                # Handle multi-class output
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use positive class
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values, 
                    X_trans, 
                    feature_names=feature_names,
                    plot_type="bar",
                    max_display=10,
                    show=False
                )
                plt.title("Factors Contributing to Depression Assessment", fontsize=14, pad=15)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.info("""
                📖 **How to read this chart:** 
                - Longer bars = stronger influence on your result
                - These are the top factors that contributed to your depression assessment
                - Factors can include your survey responses, demographics, and reported symptoms
                """)
                
            except Exception as e:
                st.warning(f"⚠️ Could not generate explanation chart: {str(e)}")
                st.info("The AI model still made predictions, but we couldn't show the contributing factors.")
        
        with tab2:
            try:
                sel_pipe = pipelines[best_anx_model]
                pre = sel_pipe.named_steps['preprocessor']
                clf = sel_pipe.named_steps['clf']
                
                X_trans = pre.transform(user_df)
                
                try:
                    feature_names = pre.get_feature_names_out()
                except:
                    feature_names = [f"Feature_{i}" for i in range(X_trans.shape[1])]
                
                # Get the anxiety estimator
                if hasattr(clf, 'estimators_'):
                    base_model = clf.estimators_[1]  # Anxiety is second target
                else:
                    base_model = clf
                
                explainer = shap.TreeExplainer(base_model)
                shap_values = explainer.shap_values(X_trans)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values, 
                    X_trans, 
                    feature_names=feature_names,
                    plot_type="bar",
                    max_display=10,
                    show=False
                )
                plt.title("Factors Contributing to Anxiety Assessment", fontsize=14, pad=15)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.info("""
                📖 **How to read this chart:** 
                - Longer bars = stronger influence on your result
                - These are the top factors that contributed to your anxiety assessment
                - Factors can include your survey responses, demographics, and reported symptoms
                """)
                
            except Exception as e:
                st.warning(f"⚠️ Could not generate explanation chart: {str(e)}")
                st.info("The AI model still made predictions, but we couldn't show the contributing factors.")

        # Technical details in expander
        if selection_mode == "View All Models":
            with st.expander("📊 View All Model Predictions", expanded=False):
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
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

        # Download results
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 💾 Save Your Results")
            st.markdown("Download a summary of your assessment to share with a healthcare provider.")
        
        with col2:
            results_summary = f"""MENTAL HEALTH SCREENING RESULTS
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

DEPRESSION ASSESSMENT (PHQ-8)
Score: {phq_total}/24
Severity: {dep_info['level']}
Description: {dep_info['description']}

Recommendations:
{chr(10).join('- ' + rec for rec in dep_info['recommendations'])}

ANXIETY ASSESSMENT (GAD-7)
Score: {gad_total}/21
Severity: {anx_info['level']}
Description: {anx_info['description']}

Recommendations:
{chr(10).join('- ' + rec for rec in anx_info['recommendations'])}

TECHNICAL DETAILS
Depression Model: {best_dep_model} (Recall: {best_dep_recall:.1%})
Anxiety Model: {best_anx_model} (Recall: {best_anx_recall:.1%})

IMPORTANT: This is a screening tool, not a diagnosis.
Please consult a healthcare professional for proper evaluation.
"""
            
            st.download_button(
                label="📥 Download Report",
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
            
            Your scores indicate significant symptoms. Please reach out for help immediately:
            
            **Crisis Resources in Kenya:**
            - 🚑 **Kenya Red Cross:** 1199
            - 📞 **Befrienders Kenya:** +254 722 178 177
            - 🏥 **Your school counselor or guidance department**
            - 👨‍👩‍👧 **A trusted adult, teacher, or family member**
            
            **You are not alone. Help is available and recovery is possible.**
            """)

        # Final disclaimer
        st.markdown("---")
        st.warning("""
        ### ⚠️ IMPORTANT DISCLAIMER
        
        This tool is for **screening and educational purposes only**. It is **NOT** a diagnostic instrument.
        
        **The results:**
        - Provide an indication of symptom severity
        - Are based on standardized mental health screening questionnaires (PHQ-8 and GAD-7)
        - Should be discussed with a qualified mental health professional
        - Do not replace professional clinical assessment
        
        **Next Steps:**
        - 🏥 Share these results with a healthcare provider or counselor
        - 📋 Use this as a starting point for a conversation about your mental health
        - 🔄 Consider re-taking this assessment periodically to track changes
        - 💬 Talk to someone you trust about how you're feeling
        
        **If you are experiencing a mental health crisis:**
        - Kenya Red Cross: **1199**
        - Befrienders Kenya: **+254 722 178 177**
        - Your school counselor or guidance department
        - Go to the nearest hospital emergency department
        """)