import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Risk", page_icon="🫀")

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = joblib.load("lgbm_heart_model.pkl")
    features  = joblib.load("feature_names.pkl")
    threshold = joblib.load("threshold.pkl")   # Fix #4: single source of truth
    iqr       = joblib.load("iqr_bounds.pkl")  # Fix #3: from x_train, not raw CSV
    return model, features, threshold, iqr

try:
    model, features, THRESHOLD, iqr_bounds = load_artifacts()
except Exception as e:
    st.error(
        f"Could not load model files: {e}\n\n"
        "Run the notebook **Heart1_improved_v2.ipynb** first to generate the pkl files."
    )
    st.stop()

# ── Preprocessing — PRESERVED from original, same logic ──────────────────────
def preprocess(raw):
    def cap(val, col):
        # Apply train-derived IQR bounds (Fix #3)
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
    age_cat     = {
        "18-24":0,"25-29":1,"30-34":2,"35-39":3,"40-44":4,"45-49":5,
        "50-54":6,"55-59":7,"60-64":8,"65-69":9,"70-74":10,"75-79":11,
        "80 or older":12
    }[raw["AgeCategory"]]

    health_score   = phys + ment
    lifestyle_risk = is_smoking + alcoholic + (1 - is_physical_act)
    disease_count  = is_stroke + is_diabetic + is_asthma + is_kidney + is_skin
    bmi_risk       = 1 if bmi > 30 else 0
    sleep_risk     = 1 if (sleep < 5 or sleep > 9) else 0
    age_disease    = age_cat * disease_count
    total_risk     = lifestyle_risk + disease_count + bmi_risk + sleep_risk + is_diffwalking

    row = {
        "GenHealth":            gen_health,
        "IS_Stroke":            is_stroke,
        "PhysicalHealth":       phys,
        "AgeCategory":          age_cat,
        "DiseaseCount":         disease_count,
        "Age_Disease":          age_disease,
        "TotalRisk":            total_risk,
        "IS_MALE":              is_male,
        "IS_DiffWalking":       is_diffwalking,
        "IS_Diabetic":          is_diabetic,
        "IS_Smoking":           is_smoking,
        "IS_PhysicalActivity":  is_physical_act,
        "LifestyleRisk":        lifestyle_risk,
        "BMI":                  bmi,
        "MentalHealth":         ment,
        "SleepTime":            sleep,
        "HealthScore":          health_score,
    }
    return pd.DataFrame([row])[features]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("❤️ Heart Disease Risk Screener")

# Fix: Prominent disclaimer before the form (was only a footer caption)
st.warning(
    "⚕️ **Medical Disclaimer:** This tool is for **informational and educational purposes only**. "
    "It does not constitute medical advice, diagnosis, or treatment. "
    "Results are based on statistical patterns in US survey data and may not apply to all populations. "
    "**Always consult a qualified healthcare professional** before making any health decisions.",
    icon="⚠️",
)

st.write("Fill in your details below and click **Predict** to see your estimated risk.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Info")
    age_cat = st.selectbox("Age Group", [
        "18-24","25-29","30-34","35-39","40-44",
        "45-49","50-54","55-59","60-64","65-69",
        "70-74","75-79","80 or older"
    ])
    sex   = st.selectbox("Sex", ["Male", "Female"])
    bmi   = st.number_input("BMI", min_value=10.0, max_value=100.0, value=27.5, step=0.1)
    sleep = st.slider("Sleep Hours / Night", 1, 24, 7)

    # Fix: warn on clinically implausible BMI
    if bmi < 14.0:
        st.warning("⚠️ BMI below 14 is extremely rare and clinically severe. Please verify your entry.")
    elif bmi > 60.0:
        st.warning("⚠️ BMI above 60 is uncommon. Please verify your entry.")

    st.subheader("General Health")
    gen_health  = st.selectbox("General Health", ["Poor","Fair","Good","Very good","Excellent"], index=2)
    phys_health = st.slider("Poor Physical Health Days (last 30)", 0, 30, 0)
    ment_health = st.slider("Poor Mental Health Days (last 30)", 0, 30, 0)

with col2:
    st.subheader("Lifestyle")
    smoking  = st.selectbox("Do you smoke?", ["No", "Yes"])
    alcohol  = st.selectbox("Heavy alcohol drinker?", ["No", "Yes"])
    phys_act = st.selectbox("Physically active (last 30 days)?", ["Yes", "No"])
    diff_walk= st.selectbox("Difficulty walking / climbing stairs?", ["No", "Yes"])

    st.subheader("Medical History")
    stroke      = st.selectbox("Ever had a stroke?", ["No", "Yes"])
    diabetic    = st.selectbox("Diabetes", ["No","No, borderline diabetes","Yes (during pregnancy)","Yes"])
    asthma      = st.selectbox("Asthma?", ["No", "Yes"])
    kidney      = st.selectbox("Kidney disease?", ["No", "Yes"])
    skin_cancer = st.selectbox("Skin cancer?", ["No", "Yes"])

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    raw = {
        "AgeCategory":    age_cat,
        "Sex":            sex,
        "BMI":            bmi,
        "SleepTime":      sleep,
        "GenHealth":      gen_health,
        "PhysicalHealth": phys_health,
        "MentalHealth":   ment_health,
        "Smoking":        smoking,
        "AlcoholDrinking":alcohol,
        "PhysicalActivity":phys_act,
        "DiffWalking":    diff_walk,
        "Stroke":         stroke,
        "Diabetic":       diabetic,
        "Asthma":         asthma,
        "KidneyDisease":  kidney,
        "SkinCancer":     skin_cancer,
    }

    X    = preprocess(raw)
    prob = float(model.predict_proba(X)[0, 1])

    # Fix #4: threshold loaded from pkl (data-driven, consistent with notebook)
    # Fix (UX): three-zone output — Low / Borderline / High
    BORDERLINE_LOW  = max(0.0, THRESHOLD - 0.10)
    BORDERLINE_HIGH = THRESHOLD

    st.divider()
    if prob >= BORDERLINE_HIGH:
        st.error(f"### ⚠️ High Risk — {prob*100:.1f}% probability")
        st.write("Several risk factors are elevated. Please consult a doctor.")
    elif prob >= BORDERLINE_LOW:
        st.warning(f"### 🟡 Borderline Risk — {prob*100:.1f}% probability")
        st.write(
            "Your inputs are in a borderline range. Consider discussing your results "
            "with a healthcare professional, especially if you have additional risk factors."
        )
    else:
        st.success(f"### ✅ Low Risk — {prob*100:.1f}% probability")
        st.write("Your inputs suggest a lower risk. Keep up a healthy lifestyle!")

    # Fix: st.progress() receives float 0.0–1.0 (Streamlit ≥ 1.18 compatible)
    st.progress(float(prob), text=f"Risk probability: {prob*100:.1f}%")

    st.caption(
        f"Model decision threshold: **{THRESHOLD:.2f}** "
        "(derived from validation data — optimised for F1 score on positive class)"
    )

st.divider()
# Preserve original footer disclaimer (now supplementary to the top banner)
st.caption(
    "⚕️ **Disclaimer:** This tool is for informational purposes only and is not a medical "
    "diagnosis. Always consult a qualified healthcare professional. "
    "This model was trained on US BRFSS survey data; performance may vary for other populations."
)
