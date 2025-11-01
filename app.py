import streamlit as st
import pandas as pd
import pickle, os, numpy as np
import shap, matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Mental Health Screening - Kenya", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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

st.markdown('<div class="main-header">🧠 Mental Health Screening Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Depression & Anxiety Assessment for Kenyan High School Students</div>', unsafe_allow_html=True)

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

# Interactive Sidebar with Model Selection Option
st.sidebar.title("⚙️ Settings & Info")

# Model selection mode
selection_mode = st.sidebar.radio(
    "🤖 Model Selection Strategy:",
    ["Auto-Select Best (Recommended)", "Manual Selection", "View All Models"],
    help="Choose how models should be selected for predictions"
)

if selection_mode == "Manual Selection":
    st.sidebar.markdown("### Choose Models")
    manual_dep_model = st.sidebar.selectbox(
        "Depression Model:",
        list(pipelines.keys()) if pipelines else ["No models available"]
    )
    manual_anx_model = st.sidebar.selectbox(
        "Anxiety Model:",
        list(pipelines.keys()) if pipelines else ["No models available"]
    )

st.sidebar.markdown("---")

# Interactive model performance visualizations
if model_metrics:
    st.sidebar.markdown("### 📊 Model Performance")
    
    # Create interactive comparison chart
    view_metric = st.sidebar.selectbox(
        "View Metric:",
        ["Recall", "Accuracy"],
        help="Select which metric to visualize"
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
            title=f"Model {view_metric} Comparison (%)",
            yaxis_title=f"{view_metric} (%)",
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.sidebar.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** The tool automatically selects the best-performing model for each target based on recall scores.")

# Main content - Interactive Form with Progress
st.markdown("## 📝 Complete the Assessment")

# Progress tracking
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

# Create tabs for different sections
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
    
    # Show PHQ score
    phq_total = sum(phq.values())
    st.markdown(f"### Current PHQ-8 Score: **{phq_total}** / 24")
    
    # Score interpretation with color coding
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
    
    # Show GAD score
    gad_total = sum(gad.values())
    st.markdown(f"### Current GAD-7 Score: **{gad_total}** / 21")
    
    # Score interpretation with color coding
    if gad_total < 5:
        st.success("✅ Minimal anxiety symptoms")
    elif gad_total < 10:
        st.info("ℹ️ Mild anxiety symptoms")
    elif gad_total < 15:
        st.warning("⚠️ Moderate anxiety symptoms")
    else:
        st.error("🚨 Severe anxiety symptoms")

# Prominent submit button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submitted = st.button("🔍 Analyze Mental Health Status", use_container_width=True)

if submitted:
    with st.spinner("🤖 AI models are analyzing your responses..."):
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

        # Model selection based on mode
        if selection_mode == "Manual Selection":
            best_dep_model = manual_dep_model
            best_anx_model = manual_anx_model
            best_dep_recall = model_metrics.get(best_dep_model, {}).get('test_recall_per_target', {}).get('Is_Depressed', 0)
            best_anx_recall = model_metrics.get(best_anx_model, {}).get('test_recall_per_target', {}).get('Has_anxiety', 0)
        else:
            # Auto-select best models
            best_dep_model = None
            best_dep_recall = -1
            
            for name, metrics in model_metrics.items():
                if 'test_recall_per_target' in metrics:
                    dep_recall = metrics['test_recall_per_target'].get('Is_Depressed', 0)
                    if dep_recall > best_dep_recall:
                        best_dep_recall = dep_recall
                        best_dep_model = name

            best_anx_model = None
            best_anx_recall = -1
            
            for name, metrics in model_metrics.items():
                if 'test_recall_per_target' in metrics:
                    anx_recall = metrics['test_recall_per_target'].get('Has_anxiety', 0)
                    if anx_recall > best_anx_recall:
                        best_anx_recall = anx_recall
                        best_anx_model = name

        depression_prediction = all_predictions.get(best_dep_model, {}).get('depression_pred', 'N/A')
        anxiety_prediction = all_predictions.get(best_anx_model, {}).get('anxiety_pred', 'N/A')

        # Results Display
        st.balloons()
        st.markdown("---")
        st.markdown("## 🎯 Analysis Complete!")
        
        # Model info badges
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🧠 **Depression Model:** {best_dep_model} | Recall: {best_dep_recall:.1%}")
        with col2:
            st.info(f"😰 **Anxiety Model:** {best_anx_model} | Recall: {best_anx_recall:.1%}")
        
        st.markdown("---")
        
        # Results in cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 Depression Assessment")
            
            # Map prediction to category name
            dep_categories = {0: 'None', 1: 'Mild', 2: 'Moderate', 3: 'Moderately Severe', 4: 'Severe'}
            dep_category = dep_categories.get(depression_prediction, str(depression_prediction))
            
            if depression_prediction in [0, 'none', 'None']:
                st.success(f"### ✅ {dep_category}")
                color = "green"
            elif depression_prediction in [1, 'mild', 'Mild']:
                st.info(f"### ℹ️ {dep_category}")
                color = "blue"
            elif depression_prediction in [2, 'moderate', 'Moderate']:
                st.warning(f"### ⚠️ {dep_category}")
                color = "orange"
            else:
                st.error(f"### 🚨 {dep_category}")
                color = "red"
            
            dep_metrics = model_metrics.get(best_dep_model, {})
            col_a, col_b = st.columns(2)
            with col_a:
                if 'test_recall_per_target' in dep_metrics:
                    st.metric("Model Recall", f"{dep_metrics['test_recall_per_target'].get('Is_Depressed', 0):.1%}")
            with col_b:
                if 'test_accuracy_per_target' in dep_metrics:
                    st.metric("Model Accuracy", f"{dep_metrics['test_accuracy_per_target'].get('Is_Depressed', 0):.1%}")
        
        with col2:
            st.markdown("### 😰 Anxiety Assessment")
            
            # Map prediction to category name
            anx_categories = {0: 'Minimal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
            anx_category = anx_categories.get(anxiety_prediction, str(anxiety_prediction))
            
            if anxiety_prediction in [0, 'minimal', 'Minimal']:
                st.success(f"### ✅ {anx_category}")
            elif anxiety_prediction in [1, 'mild', 'Mild']:
                st.info(f"### ℹ️ {anx_category}")
            elif anxiety_prediction in [2, 'moderate', 'Moderate']:
                st.warning(f"### ⚠️ {anx_category}")
            else:
                st.error(f"### 🚨 {anx_category}")
            
            anx_metrics = model_metrics.get(best_anx_model, {})
            col_c, col_d = st.columns(2)
            with col_c:
                if 'test_recall_per_target' in anx_metrics:
                    st.metric("Model Recall", f"{anx_metrics['test_recall_per_target'].get('Has_anxiety', 0):.1%}")
            with col_d:
                if 'test_accuracy_per_target' in anx_metrics:
                    st.metric("Model Accuracy", f"{anx_metrics['test_accuracy_per_target'].get('Has_anxiety', 0):.1%}")

        # Interactive comparison if "View All Models" is selected
        if selection_mode == "View All Models":
            st.markdown("---")
            st.markdown("### 📊 Detailed Model Comparison")
            
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

        # SHAP explanations with interactive tabs
        st.markdown("---")
        st.markdown("### 🔍 AI Decision Explanation (SHAP)")
        st.markdown("*Understanding what influenced the predictions*")
        
        tab1, tab2 = st.tabs([f"🧠 Depression ({best_dep_model})", f"😰 Anxiety ({best_anx_model})"])
        
        with tab1:
            try:
                sel_pipe = pipelines[best_dep_model]
                pre = sel_pipe.named_steps['preprocessor']
                clf = sel_pipe.named_steps['clf']
                X_trans = pre.transform(user_df)
                base = clf.estimators_[0]
                
                explainer = shap.Explainer(base)
                shap_values = explainer(X_trans)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.bar(shap_values, max_display=12, show=False)
                plt.title(f"Top Features Influencing Depression Prediction", fontsize=14, pad=15)
                st.pyplot(fig)
                plt.close(fig)
                
                st.info("📖 **How to read this:** Features at the top had the strongest influence on the prediction. Red bars push toward higher severity, blue bars toward lower severity.")
                
            except Exception as e:
                st.warning(f"SHAP explanation unavailable for this model: {e}")
        
        with tab2:
            try:
                sel_pipe = pipelines[best_anx_model]
                pre = sel_pipe.named_steps['preprocessor']
                clf = sel_pipe.named_steps['clf']
                X_trans = pre.transform(user_df)
                base = clf.estimators_[1]
                
                explainer = shap.Explainer(base)
                shap_values = explainer(X_trans)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.bar(shap_values, max_display=12, show=False)
                plt.title(f"Top Features Influencing Anxiety Prediction", fontsize=14, pad=15)
                st.pyplot(fig)
                plt.close(fig)
                
                st.info("📖 **How to read this:** Features at the top had the strongest influence on the prediction. Red bars push toward higher severity, blue bars toward lower severity.")
                
            except Exception as e:
                st.warning(f"SHAP explanation unavailable for this model: {e}")

        # Download results option
        st.markdown("---")
        st.markdown("### 💾 Save Your Results")
        
        results_summary = f"""
        MENTAL HEALTH SCREENING RESULTS
        Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
        
        DEPRESSION ASSESSMENT (PHQ-8)
        Model Used: {best_dep_model}
        Category: {dep_category}
        PHQ-8 Score: {phq_total}/24
        Model Recall: {best_dep_recall:.1%}
        
        ANXIETY ASSESSMENT (GAD-7)
        Model Used: {best_anx_model}
        Category: {anx_category}
        GAD-7 Score: {gad_total}/21
        Model Recall: {best_anx_recall:.1%}
        
        DISCLAIMER: This is a screening tool, not a diagnosis. Please consult a healthcare professional for proper evaluation.
        """
        
        st.download_button(
            label="📥 Download Results as Text File",
            data=results_summary,
            file_name=f"mental_health_screening_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

        st.markdown("---")
        st.error("""
        ### ⚠️ IMPORTANT DISCLAIMER
        
        This tool is for **screening and educational purposes only**. It is **NOT** a diagnostic instrument.
        
        **If you are experiencing mental health concerns:**
        - 🏥 Contact a qualified mental health professional or counselor
        - 📞 Reach out to Kenya's mental health helpline
        - 👨‍👩‍👧 Talk to a trusted adult, teacher, or family member
        
        **Crisis Resources:**
        - Kenya Red Cross: 1199
        - Befrienders Kenya: +254 722 178 177
        - Your school counselor or guidance department
        """)