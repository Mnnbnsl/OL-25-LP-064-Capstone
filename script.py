import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mental_pipeline import MentalHealthPreprocessor, MentalHealthRegressionPreprocessor

clf_pipeline = joblib.load('pipelines/mental_health_clf_pipeline.pkl')
reg_pipeline = joblib.load('pipelines/mental_health_reg_pipeline.pkl')

st.set_page_config(page_title="Mental Health App", layout="wide")
st.sidebar.title("Mental Health App")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🤒 Treatment Prediction", "🎯 Age Prediction", "🧬 Cluster Persona"])
if page == "🏠 Home":
    st.title("🧠 Mental Health Prediction")
    
    st.markdown("""
### 💡 Welcome!

In the fast-paced world of technology, mental health often takes a backseat.  
This project aims to shine a light on the mental well-being of tech employees through data.

We’ve built an end-to-end machine learning application that:
- Predicts whether an individual is likely to seek treatment
- Estimates the respondent’s age based on mental and workplace responses
- Groups users into similar mental health personas for personalized insights

### 📚 Dataset Overview

The dataset used in this project comes from a mental health survey conducted among technology workers globally.  
It includes questions on:
- Treatment history
- Workplace policies and support systems
- Personal and family mental health conditions
- Attitudes toward discussing mental health

We cleaned, processed, and modeled this data to build both a classification and regression system, along with clustering for deeper user understanding.

### 🎯 Our Goal

To build a tool that can:
- Help organizations understand factors that affect employee mental health
- Provide insight into the kind of support employees might need
- Raise awareness and reduce stigma around mental health in tech workplaces

""")

if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.markdown("""
    ## 📈 What does the data tell us?

    Understanding the structure and patterns in the dataset is a crucial first step before model building. 
    This Exploratory Data Analysis (EDA) section gives us an opportunity to:

    - Uncover trends and anomalies
    - Identify the distribution of key features like age, gender, and treatment status
    - Understand how workplace factors relate to mental health
    - Guide feature selection and model development

    ### 🧠 Mental Health in Tech: Initial Observations
    The dataset used in this project is based on a survey of tech employees around the world. It contains insights on:
    - Whether employees have sought mental health treatment
    - Workplace policies around mental health
    - Ease of access to mental health resources
    - Demographic features like age and gender

    Let’s walk through some key findings below.
    """)

    # Age Distribution
    st.image("eda/age_dist.png", caption="Age Distribution of Survey Respondents", use_container_width=True)
    st.markdown("""
    **Observation:**  
    We filtered the dataset to retain participants aged 18 to 65. Most respondents are in their 20s and 30s, with a sharp drop after age 40.  
    This age distribution is typical of the tech industry and impacts how we interpret mental health patterns in younger vs older employees.
    """)

    # Treatment Histogram
    st.image("eda/treatment_hist.png", caption="Family History vs Seeking Treatment", use_container_width=True)
    st.markdown("""
    **Observation:**  
    Individuals with a family history of mental illness are significantly more likely to seek treatment themselves.  
    This confirms a known risk factor and is an important feature in our classification model.
    """)

    # Correlation Heatmap
    st.image("eda/heatmap.png", caption="Correlation Heatmap of Encoded Features", use_container_width=True)
    st.markdown("""
    **Observation:**  
    This heatmap shows the correlation between *Work Interference* and *Perceived Mental health consequences*. This justifies their inclusion in predictive modeling.

    ---

    ### 📤 Download Full EDA Notebook

    You can download the full notebook with detailed code, plots, and markdown insights.  
    It includes:
    - Missing value analysis
    - Distribution plots
    - Encoded categorical visualizations
    - Chi-square & ANOVA tests
    - Recommendations for feature selection

    """)

    with open("models/capstoneeda.ipynb", "rb") as file:
        btn = st.download_button(
            label="📥 Download EDA Report (Notebook)",
            data=file,
            file_name="eda_report.ipynb",
            mime="application/octet-stream"
        )

elif page == "🤒 Treatment Prediction":
    st.title("🩺 Mental Health Treatment Prediction")
    st.markdown("""
    ### 🔍 Can we predict if someone will seek mental health treatment?

    This classifier predicts whether a person is likely to **seek treatment**, based on:
    - Demographics
    - Workplace support
    - Personal and family history
    - Attitudes and comfort levels

    Fill in the form to get the prediction and model confidence.
    """)
    with st.form("classification_form"):
        age = st.slider("Age", 18, 65, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other/Non-Binary"])
        self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
        family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"])
        work_interfere = st.selectbox("How often does mental health interfere with your work?", ["Never", "Rarely", "Sometimes", "Often"])
        no_employees = st.selectbox("Number of employees in your company:", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
        remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
        tech_company = st.selectbox("Is your employer a tech company?", ["Yes", "No"])
        benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
        care_options = st.selectbox("Do you know the care options available?", ["Yes", "No", "Not sure"])
        wellness_program = st.selectbox("Is a wellness program provided?", ["Yes", "No", "Don't know"])
        seek_help = st.selectbox("Are resources provided to seek help?", ["Yes", "No", "Don't know"])
        anonymity = st.selectbox("Is anonymity guaranteed if you seek help?", ["Yes", "No", "Don't know"])
        leave = st.selectbox("How easy is it to take mental health leave?", ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"])
        mental_health_consequence = st.selectbox("Are there consequences for seeking MH treatment?", ["Yes", "No", "Maybe"])
        phys_health_consequence = st.selectbox("Are there consequences for seeking PH treatment?", ["Yes", "No", "Maybe"])
        coworkers = st.selectbox("Would you discuss MH with coworkers?", ["Yes", "No", "Some of them"])
        supervisor = st.selectbox("Would you discuss MH with your supervisor?", ["Yes", "No", "Some of them"])
        mental_health_interview = st.selectbox("Would you discuss MH in an interview?", ["Yes", "No", "Maybe"])
        phys_health_interview = st.selectbox("Would you discuss PH in an interview?", ["Yes", "No", "Maybe"])
        mental_vs_physical = st.selectbox("Which is more serious?", ["Yes", "No", "Don't know"])
        obs_consequence = st.selectbox("Have you observed mental health consequences in coworkers?", ["Yes", "No"])
        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "self_employed": self_employed,
            "family_history": family_history,
            "work_interfere": work_interfere,
            "no_employees": no_employees,
            "remote_work": remote_work,
            "tech_company": tech_company,
            "benefits": benefits,
            "care_options": care_options,
            "wellness_program": wellness_program,
            "seek_help": seek_help,
            "anonymity": anonymity,
            "leave": leave,
            "mental_health_consequence": mental_health_consequence,
            "phys_health_consequence": phys_health_consequence,
            "coworkers": coworkers,
            "supervisor": supervisor,
            "mental_health_interview": mental_health_interview,
            "phys_health_interview": phys_health_interview,
            "mental_vs_physical": mental_vs_physical,
            "obs_consequence": obs_consequence
        }])

        prediction = clf_pipeline.predict(input_data)[0]
        confidence = clf_pipeline.predict_proba(input_data).max()

        st.success(f"🧾 Prediction: **{'Yes' if prediction == 1 else 'No'}**")
        st.info(f"🔐 Model Confidence: **{confidence:.2%}**")

elif page == "🎯 Age Prediction":
    st.title("⏳ Predicting Respondent Age")

    st.markdown("""
    ### 🧮 Can we estimate someone's age from mental health responses?

    This model predicts a person's **age** based on:
    - Work environment
    - Comfort level
    - Support systems
    - Mental health experiences

    Use this to explore age-based patterns in mental health trends.
    """)
    with st.form("regression_form"):
        gender = st.selectbox("Gender", ["Male", "Female", "Other/Non-Binary"])
        self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
        family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"])
        work_interfere = st.selectbox("How often does mental health interfere with your work?", ["Never", "Rarely", "Sometimes", "Often"])
        treatment = st.selectbox("Have you seeken any treatment before?", ["Yes", "No"])
        no_employees = st.selectbox("Number of employees in your company:", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
        remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
        tech_company = st.selectbox("Is your employer a tech company?", ["Yes", "No"])
        benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
        care_options = st.selectbox("Do you know the care options available?", ["Yes", "No", "Not sure"])
        wellness_program = st.selectbox("Is a wellness program provided?", ["Yes", "No", "Don't know"])
        seek_help = st.selectbox("Are resources provided to seek help?", ["Yes", "No", "Don't know"])
        anonymity = st.selectbox("Is anonymity guaranteed if you seek help?", ["Yes", "No", "Don't know"])
        leave = st.selectbox("How easy is it to take mental health leave?", ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"])
        mental_health_consequence = st.selectbox("Are there consequences for seeking MH treatment?", ["Yes", "No", "Maybe"])
        phys_health_consequence = st.selectbox("Are there consequences for seeking PH treatment?", ["Yes", "No", "Maybe"])
        coworkers = st.selectbox("Would you discuss MH with coworkers?", ["Yes", "No", "Some of them"])
        supervisor = st.selectbox("Would you discuss MH with your supervisor?", ["Yes", "No", "Some of them"])
        mental_health_interview = st.selectbox("Would you discuss MH in an interview?", ["Yes", "No", "Maybe"])
        phys_health_interview = st.selectbox("Would you discuss PH in an interview?", ["Yes", "No", "Maybe"])
        mental_vs_physical = st.selectbox("Which is more serious?", ["Yes", "No", "Don't know"])
        obs_consequence = st.selectbox("Have you observed mental health consequences in coworkers?", ["Yes", "No"])
        submitted = st.form_submit_button("🔍 Predict")

        if submitted:
            input_data = {
                'Gender': gender,
                'self_employed': self_employed,
                'family_history': family_history,
                'treatment' : treatment,
                'work_interfere': work_interfere,
                'no_employees': no_employees,
                'remote_work': remote_work,
                'tech_company': tech_company,
                'benefits': benefits,
                'care_options': care_options,
                'wellness_program': wellness_program,
                'seek_help': seek_help,
                'anonymity': anonymity,
                'leave': leave,
                'mental_health_consequence': mental_health_consequence,
                'phys_health_consequence': phys_health_consequence,
                'coworkers': coworkers,
                'supervisor': supervisor,
                'mental_health_interview': mental_health_interview,
                'phys_health_interview': phys_health_interview,
                'mental_vs_physical': mental_vs_physical,
                'obs_consequence': obs_consequence
            }

            input_df = pd.DataFrame([input_data])

            # Predict age
            predicted_age = reg_pipeline.predict(input_df)[0]

            st.success(f"🧠 Estimated Age: {int(predicted_age)} years")

elif page == "🧬 Cluster Persona":
    st.title("👥 Mental Health Personas")
    st.markdown("""
---

## 🧠 Persona Descriptions

### 🔹 **Cluster 0 – Open Advocates**
- **Likely to work in supportive environments**
- **More comfortable discussing mental health**
- **Higher chances of having mental health benefits and policies in place**
- **Tends to seek treatment proactively**

These individuals are more engaged with mental health support systems and may benefit from continued wellness initiatives.

---

### 🔸 **Cluster 1 – Silent Sufferers**
- **Less likely to report available support at workplace**
- **May feel hesitant to disclose mental health issues**
- **Lower tendency to seek treatment**
- **Possibly facing stigma or organizational barriers**

This group may need **targeted outreach, policy changes, or culture shifts** to feel safer and more supported at work.

---

## 🧩 How can this help?
Understanding personas can:
- Inform HR and policy changes
- Personalize mental health programs
- Highlight gaps in organizational support
""")
    st.image("eda/cluster.png", caption="Cluster depiction using PCA", use_container_width=True)