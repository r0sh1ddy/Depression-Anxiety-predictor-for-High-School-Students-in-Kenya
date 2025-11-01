import streamlit as st
import pandas as pd
import pickle, os, numpy as np
import shap, matplotlib.pyplot as plt

st.set_page_config(page_title="Mental Health Screening App - Kenya", layout="wide")
st.title("Mental Health Screening App - Kenya")
st.write("Predicts Depression (PHQ-8) and Anxiety (GAD-7) categories. The app evaluates multiple models and selects the best one based on recall (averaged across both targets). SHAP explanations are shown for the selected model.")

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

# Sidebar summary
st.sidebar.header("Model summary (validation)")
if model_metrics:
    for m, vals in model_metrics.items():
        st.sidebar.write(f"**{m}** — Recall: {vals['average_recall']:.3f}, Acc: {vals['average_accuracy']:.3f")

st.sidebar.markdown("---")
st.sidebar.write("This app selects the best model by highest stored recall (avg across depression & anxiety). If recall is not available it falls back to runtime confidence.")

with st.form("mh_form"):
    st.subheader("School & Demographics")
    boarding_day = st.selectbox("Do you attend a boarding or day school?", ["Boarding", "Day"])
    school_type = st.selectbox("Is your school a boys’, girls’, or mixed school?", ["Boys", "Girls", "Mixed"])
    school_demo = st.selectbox("What type of area best describes your school?", ["Urban", "Rural", "Semi-urban"])
    school_county = st.selectbox("Which county is your school located in?", ["Nairobi","Kiambu","Kisumu","Mombasa","Nakuru","Other"])

    age = st.number_input("Age", min_value=12, max_value=25, value=16)
    gender = st.selectbox("Gender", ["Male", "Female"])
    form = st.selectbox("Which form are you in?", [1,2,3,4])
    religion = st.selectbox("Religion", ["Christian", "Muslim", "Other"])
    parents_home = st.selectbox("Do your parents live together?", ["Both parents", "One parent", "None"])
    parents_dead = st.selectbox("How many parents are deceased?", [0,1,2,3,4])
    fathers_edu = st.selectbox("Father's highest education level:", ["None","Primary","Secondary","Tertiary","University"])
    mothers_edu = st.selectbox("Mother's highest education level:", ["None","Primary","Secondary","Tertiary","University"])
    co_curr = st.selectbox("Do you participate in co-curricular activities?", ["No","Yes"])
    sports = st.selectbox("Do you engage in sports?", ["No","Yes"])
    acad_ability = st.slider("Perceived academic abilities (1=low, 5=high)", 1, 5, 3)

    st.markdown("---")
    st.subheader("PHQ-8 (Depression) — Over the past 2 weeks, how often have you been bothered by:")
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
    likert_labels = ["Not at all","Several days","More than half the days","Nearly every day"]
    for i, q in enumerate(phq_qs, 1):
        phq[f'PHQ_{i}'] = st.select_slider(f"PHQ_{i}: {q}", options=[0,1,2,3], format_func=lambda x, labels=likert_labels: labels[x])

    st.markdown("---")
    st.subheader("GAD-7 (Anxiety) — Over the past 2 weeks, how often have you been bothered by:")
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
        gad[f'GAD_{i}'] = st.select_slider(f"GAD_{i}: {q}", options=[0,1,2,3], format_func=lambda x, labels=likert_labels: labels[x])

    submitted = st.form_submit_button("Predict")

if submitted:
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

    st.write("### Your input (preview)")
    st.dataframe(user_df)

    results = []
    for name, pipe in pipelines.items():
        try:
            pred = pipe.predict(user_df)[0]
            conf = 0.0
            try:
                proba = pipe.predict_proba(user_df)
                max_probs = [p[0].max() for p in proba]
                conf = float(sum(max_probs)/len(max_probs))
            except Exception:
                conf = 0.0
            recall = model_metrics.get(name, {}).get('average_recall', None)
            acc = model_metrics.get(name, {}).get('average_accuracy', None)
            results.append({"Model": name, "Prediction": pred, "Confidence": conf, "Recall": recall, "Accuracy": acc})
        except Exception as e:
            results.append({"Model": name, "Prediction": f"Error: {e}", "Confidence": 0.0, "Recall": model_metrics.get(name, {}).get('average_recall', None), "Accuracy": model_metrics.get(name, {}).get('average_accuracy', None)})

    with_recall = [r for r in results if r["Recall"] is not None]
    if with_recall:
        best = max(with_recall, key=lambda x: x["Recall"])
        reason = "highest stored recall (avg across both targets)"
    else:
        best = max(results, key=lambda x: x["Confidence"])
        reason = "highest runtime confidence"

    st.markdown("## Prediction Results")
    if isinstance(best["Prediction"], (list, tuple, pd.Series, np.ndarray)):
        dep_cat, anx_cat = best["Prediction"]
        st.success(f"Selected Model: **{best['Model']}** (selected by {reason})")
        st.write(f"- Depression (PHQ category): **{dep_cat}**")
        st.write(f"- Anxiety (GAD category): **{anx_cat}**")
        st.write(f"- Model Recall: {best['Recall']:.3f}, Accuracy: {best['Accuracy']:.3f}")
    else:
        st.success(f"Selected Model: **{best['Model']}** (selected by {reason})")
        st.write("Prediction:", best["Prediction"])

    st.markdown("---")
    st.subheader("All model results")
    st.dataframe(pd.DataFrame(results))

    # SHAP explanation (try to create explainer for first estimator inside MultiOutputClassifier)
    try:
        sel_pipe = pipelines[best["Model"]]
        pre = sel_pipe.named_steps['preprocessor']
        clf = sel_pipe.named_steps['clf']
        X_trans = pre.transform(user_df)
        base = clf.estimators_[0]
        try:
            explainer = shap.Explainer(base)
            shap_values = explainer(X_trans)
            st.subheader("SHAP — feature importance (first estimator)")
            shap.plots.bar(shap_values, max_display=12, show=False)
            st.pyplot(bbox_inches='tight')
        except Exception as e:
            st.write("SHAP could not compute using TreeExplainer:", e)
    except Exception as e:
        st.write("SHAP explanation not available for selected model:", e)

    st.markdown("---")
    st.write("Interpretation note: this tool is for screening and educational purposes only.")
