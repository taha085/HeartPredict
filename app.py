"""
Run:
    streamlit run app.py

Required files (same folder):
    lgbm_heart_model.pkl
    feature_names.pkl
    threshold.pkl
    iqr_bounds.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Risk", page_icon="🫀")

# ── Load model artifacts ──
@st.cache_resource
def load_artifacts():
    model     = joblib.load("lgbm_heart_model.pkl")
    features  = joblib.load("feature_names.pkl")
    threshold = joblib.load("threshold.pkl")
    iqr       = joblib.load("iqr_bounds.pkl")
    return model, features, threshold, iqr

try:
    model, features, THRESHOLD, iqr_bounds = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}\nRun `python run_pipeline.py` first.")
    st.stop()

# ── Preprocessing (mirrors training pipeline exactly) ──
def preprocess(raw):
    def cap(val, col):
        return float(np.clip(val, iqr_bounds[col]["lb"], iqr_bounds[col]["ub"]))

    bmi   = cap(raw["BMI"],            "BMI")
    phys  = cap(raw["PhysicalHealth"], "PhysicalHealth")
    ment  = cap(raw["MentalHealth"],   "MentalHealth")
    sleep = cap(raw["SleepTime"],      "SleepTime")

    is_male         = 1 if raw["Sex"] == "Male" else 0
    is_smoking      = 1 if raw["Smoking"] == "Yes" else 0
    is_stroke       = 1 if raw["Stroke"] == "Yes" else 0
    is_diffwalking  = 1 if raw["DiffWalking"] == "Yes" else 0
    is_physical_act = 1 if raw["PhysicalActivity"] == "Yes" else 0
    is_asthma       = 1 if raw["Asthma"] == "Yes" else 0
    is_kidney       = 1 if raw["KidneyDisease"] == "Yes" else 0
    is_skin         = 1 if raw["SkinCancer"] == "Yes" else 0
    alcoholic       = 1 if raw["AlcoholDrinking"] == "Yes" else 0

    gen_health  = {"Poor":0,"Fair":1,"Good":2,"Very good":3,"Excellent":4}[raw["GenHealth"]]
    is_diabetic = {"No":0,"No, borderline diabetes":1,"Yes (during pregnancy)":2,"Yes":3}[raw["Diabetic"]]
    age_cat     = {"18-24":0,"25-29":1,"30-34":2,"35-39":3,"40-44":4,"45-49":5,
                   "50-54":6,"55-59":7,"60-64":8,"65-69":9,"70-74":10,"75-79":11,
                   "80 or older":12}[raw["AgeCategory"]]

    health_score   = phys + ment
    lifestyle_risk = is_smoking + alcoholic + (1 - is_physical_act)
    disease_count  = is_stroke + is_diabetic + is_asthma + is_kidney + is_skin
    bmi_risk       = 1 if bmi > 30 else 0
    sleep_risk     = 1 if (sleep < 5 or sleep > 9) else 0
    age_disease    = age_cat * disease_count
    total_risk     = lifestyle_risk + disease_count + bmi_risk + sleep_risk + is_diffwalking

    row = {
        "GenHealth": gen_health, "IS_Stroke": is_stroke,
        "PhysicalHealth": phys, "AgeCategory": age_cat,
        "DiseaseCount": disease_count, "Age_Disease": age_disease,
        "TotalRisk": total_risk, "IS_MALE": is_male,
        "IS_DiffWalking": is_diffwalking, "IS_Diabetic": is_diabetic,
        "IS_Smoking": is_smoking, "IS_PhysicalActivity": is_physical_act,
        "LifestyleRisk": lifestyle_risk, "BMI": bmi,
        "MentalHealth": ment, "SleepTime": sleep, "HealthScore": health_score,
    }
    return pd.DataFrame([row])[features]

# ── UI ──
st.title("🫀 Heart Disease Risk Screener")
st.write("Fill in your details below and click **Predict** to see your risk.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Info")
    age_cat   = st.selectbox("Age Group", ["18-24","25-29","30-34","35-39","40-44",
                              "45-49","50-54","55-59","60-64","65-69","70-74","75-79","80 or older"])
    sex       = st.selectbox("Sex", ["Male", "Female"])
    bmi       = st.number_input("BMI", min_value=10.0, max_value=100.0, value=27.5, step=0.1)
    sleep     = st.slider("Sleep Hours / Night", 1, 24, 7)

    st.subheader("General Health")
    gen_health  = st.selectbox("General Health", ["Poor","Fair","Good","Very good","Excellent"], index=2)
    phys_health = st.slider("Poor Physical Health Days (last 30)", 0, 30, 0)
    ment_health = st.slider("Poor Mental Health Days (last 30)", 0, 30, 0)

with col2:
    st.subheader("Lifestyle")
    smoking   = st.selectbox("Do you smoke?", ["No", "Yes"])
    alcohol   = st.selectbox("Heavy alcohol drinker?", ["No", "Yes"])
    phys_act  = st.selectbox("Physically active (last 30 days)?", ["Yes", "No"])
    diff_walk = st.selectbox("Difficulty walking / climbing stairs?", ["No", "Yes"])

    st.subheader("Medical History")
    stroke      = st.selectbox("Ever had a stroke?", ["No", "Yes"])
    diabetic    = st.selectbox("Diabetes", ["No","No, borderline diabetes","Yes (during pregnancy)","Yes"])
    asthma      = st.selectbox("Asthma?", ["No", "Yes"])
    kidney      = st.selectbox("Kidney disease?", ["No", "Yes"])
    skin_cancer = st.selectbox("Skin cancer?", ["No", "Yes"])

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    raw = {
        "AgeCategory": age_cat, "Sex": sex, "BMI": bmi, "SleepTime": sleep,
        "GenHealth": gen_health, "PhysicalHealth": phys_health, "MentalHealth": ment_health,
        "Smoking": smoking, "AlcoholDrinking": alcohol, "PhysicalActivity": phys_act,
        "DiffWalking": diff_walk, "Stroke": stroke, "Diabetic": diabetic,
        "Asthma": asthma, "KidneyDisease": kidney, "SkinCancer": skin_cancer,
    }

    X    = preprocess(raw)
    prob = float(model.predict_proba(X)[0, 1])
    pred = prob >= THRESHOLD

    st.divider()
    if pred:
        st.error(f"### ⚠️ High Risk — {prob*100:.1f}% probability")
        st.write("Several risk factors are elevated. Please consult a doctor.")
    else:
        st.success(f"### ✅ Low Risk — {prob*100:.1f}% probability")
        st.write("Your inputs suggest a lower risk. Keep up a healthy lifestyle!")

    st.progress(prob, text=f"Risk probability: {prob*100:.1f}%")

st.divider()
st.caption("⚕️ **Disclaimer:** This tool is for informational purposes only and is not a medical diagnosis. Always consult a qualified healthcare professional.")
