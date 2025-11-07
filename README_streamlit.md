AdolescentMind

# Mental Health Screening Tool - Kenya

## Overview

This is an **Interactive ensemble tool** for assessing depression (PHQ-8) and anxiety (GAD-7) symptoms among high school students in Kenya. The application uses machine learning models trained on real data to provide personalized mental health assessments.

> **IMPORTANT:** This is a **screening tool**, NOT a diagnostic instrument. If you are concerned about your mental health, please contact a qualified healthcare professional.

---

## What This Tool Does

- **Depression Screening:** Uses the PHQ-8 (Patient Health Questionnaire-8) to assess depression symptoms
- **Anxiety Screening:** Uses the GAD-7 (Generalized Anxiety Disorder-7) to assess anxiety symptoms
- **Smart Model Selection:** Automatically selects the best-performing AI model for each assessment
- **Explainable AI:** Shows you which factors influenced the predictions using SHAP analysis
- **Multi-Model Support:** Compare predictions from different machine learning models

---

## How to Use the Tool

### **Step 1: Choose Your Model Selection Mode (Sidebar)**

The tool offers three modes:
### Change background as desired (Sidebar)
###  View Model Performance (Sidebar)**

- Check the **interactive performance charts** to see how models compare
- Toggle between **Recall** and **Accuracy** metrics
- Understand which models perform best for Depression vs. Anxiety

---

## Completing the Assessment

### **Tab 1: Demographics **

Fill in your basic information:
- **School Information:** Type (Boarding/Day), Gender composition, Location, County
- **Personal Details:** Age, Gender, Form (grade level)
- **Family Background:** Religion, parents' living situation, parents' education
- **Activities:** Co-curricular participation, sports involvement
- **Academic Self-Rating:** Rate your perceived academic abilities (1-5)

*Why this matters:* Demographics help the model understand context and provide more accurate predictions.

---

### **Tab 2: Depression Screening - PHQ-8 **

Answer **8 questions** about how you've felt **over the past 2 weeks**.

For each statement, select how often you've experienced it:
- **Not at all** (0 points)
- **Several days** (1 point)
- **More than half the days** (2 points)
- **Nearly every day** (3 points)

**Questions cover:**
- Interest in activities
- Feeling down or hopeless
- Sleep problems
- Energy levels
- Appetite changes
- Self-worth
- Concentration issues
- Physical restlessness or slowness

**Real-time Score:**
- The tool shows your current PHQ-8 score (0-24)
- Color-coded severity: Green (Minimal) → Blue (Mild) → Orange (Moderate/Mod. Severe) → Red (Severe)

---

### **Tab 3: Anxiety Screening - GAD-7**

Answer **7 questions** about anxiety symptoms **over the past 2 weeks**.

Use the same frequency scale as PHQ-8.

**Questions cover:**
- Nervousness or anxiety
- Inability to stop worrying
- Excessive worry
- Trouble relaxing
- Restlessness
- Irritability
- Fear or dread

**Real-time Score:**
- Shows your current GAD-7 score (0-21)
- Color-coded severity: Green (Minimal) → Blue (Mild) → Orange (Moderate) → Red (Severe)


##  Understanding Your Results

### **Analysis Summary**

After clicking "** Analyze Mental Health Status**", you'll see:

1. **Model Information Badges**
   - Which AI model was used for Depression
   - Which AI model was used for Anxiety
   - Their recall scores (accuracy in detecting true cases)

2. **Depression Assessment Card** 
   - **Category:** None, Mild, Moderate, Moderately Severe, or Severe
   - **Color Coding:** Green (None) → Blue (Mild) → Orange (Moderate) → Red (Severe)
   - **Model Metrics:** Recall and Accuracy percentages

3. **Anxiety Assessment Card** 
   - **Category:** Minimal, Mild, Moderate, or Severe
   - **Color Coding:** Green (Minimal) → Blue (Mild) → Orange (Moderate) → Red (Severe)
   - **Model Metrics:** Recall and Accuracy percentages


##  Understanding the AI Explanations (SHAP)

### **What is SHAP?**

SHAP (SHapley Additive exPlanations) shows you **which factors** most influenced your prediction.

### **How to Read SHAP Charts:**

- **Vertical Axis:** Lists features (your responses and demographics)
- **Horizontal Axis:** Shows impact strength
- **Top Features:** Had the strongest influence on the prediction
- **Bar Colors:** 
  - 🔴 Red/Warm colors: Pushed toward higher severity
  - 🔵 Blue/Cool colors: Pushed toward lower severity

**Example Interpretation:**
```
PHQ_2 (Feeling down/hopeless)     ████████████
GAD_3 (Worrying too much)         ██████████
Age                               ████
```
This means:
- Your response to "Feeling down/hopeless" had the biggest impact
- "Worrying too much" was also very influential
- Age had a moderate influence

## Saving Your Results

Click the **" Download Results as Text File"** button to save:
- Date and time of assessment
- Your PHQ-8 and GAD-7 scores
- Predicted categories for both Depression and Anxiety
- Models used and their performance metrics
- Important disclaimer

**File format:** Plain text (.txt) file  
**Filename:** `mental_health_screening_YYYYMMDD_HHMM.txt`

## Comparing Multiple Models (Advanced)

If you selected **"View All Models"** mode:

You'll see a detailed comparison table showing:
- **All 4 AI models** (Logistic, RandomForest, XGBoost, LightGBM)
- Their predictions for YOUR specific case
- Individual recall and accuracy for Depression and Anxiety
- How models differ in their assessments

**Use this to:**
- Understand model variability
- See if there's consensus among models
- Learn about different AI approaches


## Understanding the Models

### **Available Models:**

1. **Logistic Regression** 
   - Simple, interpretable linear model
   - Good baseline performance
   - Fast predictions

2. **Random Forest** 
   - Ensemble of decision trees
   - Handles complex patterns
   - Robust to outliers

3. **XGBoost** ⚡
   - Advanced gradient boosting
   - High performance
   - Handles missing data well

4. **LightGBM**
   - Optimized gradient boosting
   - Very fast training/prediction
   - Often highest accuracy

### **Performance Metrics Explained:**

- **Recall:** Ability to correctly identify true cases (important for screening)
- **Accuracy:** Overall correctness of predictions
- **Higher is better** for both metrics


## When to Seek Help

### **Immediate Help Needed If:**

- You're having thoughts of self-harm or suicide
- You're unable to perform daily activities
- Symptoms are severely impacting your school or relationships
- You feel overwhelmed and can't cope

### **Crisis Resources in Kenya:**

 **Emergency Hotlines:**
- **Kenya Red Cross:** 1199
- **Befrienders Kenya:** +254 722 178 177
- **Lifeline Kenya:** +254 722 178 177

**Professional Help:**
- Visit your school counselor or guidance department
- Contact your local health facility
- Reach out to a mental health professional

 **Talk to Someone:**
- A trusted teacher or school administrator
- A family member or guardian
- A religious or community leader


##  Important Limitations

### **This Tool Cannot:**
- Provide a clinical diagnosis
- Replace professional mental health care
- Prescribe treatment or medication
- Handle emergency situations

### **This Tool Can:**
- Screen for depression and anxiety symptoms
- Provide an objective assessment of your responses
- Help you understand your mental health status
- Guide you on whether to seek professional help
- Track changes over time (if used regularly)

## Privacy & Data

- **No data is stored permanently** by this application
- Your responses are processed in real-time
- Downloaded results are saved **only on your device**
- No personal information is sent to external servers
- The tool runs independently in your browser


## About the Assessments

### **PHQ-8 (Depression)**
- **Full Name:** Patient Health Questionnaire-8
- **Purpose:** Screens for Major Depressive Disorder
- **Score Range:** 0-24
- **Categories:**
  - 0-4: Minimal/None
  - 5-9: Mild
  - 10-14: Moderate
  - 15-19: Moderately Severe
  - 20-24: Severe

### **GAD-7 (Anxiety)**
- **Full Name:** Generalized Anxiety Disorder-7
- **Purpose:** Screens for Generalized Anxiety Disorder
- **Score Range:** 0-21
- **Categories:**
  - 0-4: Minimal
  - 5-9: Mild
  - 10-14: Moderate
  - 15-21: Severe


## Best Practices

1. **Be Honest:** Answer based on how you truly feel, not how you think you should feel
2. **Take Your Time:** There's no rush - reflect on each question
3. **Private Space:** Complete the assessment in a quiet, private location
4. **Regular Screening:** Consider retaking every few months to track changes
5. **Seek Help:** If results concern you, don't hesitate to reach out for support
6. **Share Results:** Consider sharing downloaded results with a counselor or healthcare provider


##  Updates & Feedback

### **Model Training:**
- Models were trained on data from Kenyan high school students
- Training included demographic, academic, and symptom data
- Validation was performed to ensure accuracy and fairness

### **Continuous Improvement:**
- Models are periodically retrained with new data
- Performance metrics are continuously monitored
- User feedback helps improve the tool

## Additional Resources
### **Mental Health Education:**
- [Kenya Mental Health Policy](https://www.health.go.ke)
- [WHO Mental Health Resources](https://www.who.int/health-topics/mental-health)
- School guidance and counseling departments

### **Support Organizations:**
- **Kenya Psychological Association:** Professional mental health support
- **BasicNeeds Kenya:** Community mental health programs
- **Chiromo Lane Medical Centre:** Specialized mental health facility


## Frequently Asked Questions

**Q: How accurate is this tool?**  
A: The models achieve 80-90% accuracy on test data. However, accuracy varies by individual and should not replace professional assessment.

**Q: Can I use this for someone else?**  
A: The tool is designed for self-assessment. For assessing others (especially minors), consult a qualified professional.

**Q: How often should I use this tool?**  
A: Mental health screening can be done monthly or whenever you notice changes in your mood or behavior.

**Q: What if my score is high?**  
A: High scores suggest you should speak with a counselor or healthcare provider. The tool provides guidance, not diagnosis.

**Q: Why do different models give different results?**  
A: Each model learns patterns differently. The tool uses the most reliable model based on validation performance.

**Q: Is my data secure?**  
A: Yes. All processing happens in your browser. Nothing is stored or transmitted to external servers.

## Contact & Support

For technical issues or questions about the tool:
- Check the repository's GitHub Issues page
- Contact your school's IT or counseling department
- Refer to the project documentation

**Remember:** This tool is here to help, but it's not a substitute for professional care. Your mental health matters! 

---

*Last Updated: November 2025*  
*Developed for Kenyan High School Students*  
*Based on PHQ-8 and GAD-7 validated screening instruments*
