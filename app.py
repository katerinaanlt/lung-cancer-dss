import streamlit as st
import requests

API_URL = "http://api:8000/predict"

st.set_page_config(page_title="Lung Cancer DSS", layout="wide")
st.title("Lung Cancer Risk Decision Support System")
st.markdown("Enter the patient's medical and demographic data to generate an XGBoost risk assessment.")

with st.form("patient_data_form"):
    st.subheader("Patient Parameters")
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        st.markdown("**Demographics & Vitals**")
        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        gender = st.selectbox("Gender", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        oxygen_sat = st.number_input("Oxygen Saturation (%)", min_value=70.0, max_value=100.0, value=98.0)
        education = st.number_input("Education (Years)", min_value=0, max_value=30, value=12)
        income = st.selectbox("Income Level (1-4)", options=[1, 2, 3, 4], index=1)
       
    with col2:
        st.markdown("**Smoking History**")
        smoker = st.selectbox("Current/Past Smoker", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        smoking_years = st.number_input("Smoking Years", min_value=0, max_value=int(age), value=0)
        cigs_per_day = st.number_input("Cigarettes per Day", min_value=0, max_value=100, value=0)
        pack_years = st.number_input("Pack Years", min_value=0.0, max_value=150.0, value=0.0)
        passive_smoking = st.selectbox("Passive Smoking", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
       
    with col3:
        st.markdown("**Environmental & Medical**")
        air_pollution = st.number_input("Air Pollution Index", min_value=0, max_value=300, value=50)
        occ_exposure = st.selectbox("Occupational Exposure", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        radon = st.selectbox("Radon Exposure", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        fam_history = st.selectbox("Family History of Cancer", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        asthma = st.selectbox("Asthma", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        prev_tb = st.selectbox("Previous TB", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
       
    with col4:
        st.markdown("**Symptoms & Lifestyle**")
        chronic_cough = st.selectbox("Chronic Cough", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        chest_pain = st.selectbox("Chest Pain", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        sob = st.selectbox("Shortness of Breath", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        fatigue = st.selectbox("Fatigue", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        exercise = st.number_input("Exercise (Hours/Week)", min_value=0.0, max_value=40.0, value=3.0)
        diet = st.selectbox("Diet Quality (1-4)", options=[1, 2, 3, 4], index=2)
        alcohol = st.number_input("Alcohol (Units/Week)", min_value=0.0, max_value=100.0, value=5.0)
        healthcare = st.selectbox("Healthcare Access (1-4)", options=[1, 2, 3, 4], index=2)

    submit_button = st.form_submit_button(label="Assess Risk")

if submit_button:
    payload = {
        "age": age, "gender": gender, "education_years": education, "income_level": income,
        "smoker": smoker, "smoking_years": smoking_years, "cigarettes_per_day": cigs_per_day,
        "pack_years": pack_years, "passive_smoking": passive_smoking, "air_pollution_index": air_pollution,
        "occupational_exposure": occ_exposure, "radon_exposure": radon, "family_history_cancer": fam_history,
        "asthma": asthma, "previous_tb": prev_tb, "chronic_cough": chronic_cough, "chest_pain": chest_pain,
        "shortness_of_breath": sob, "fatigue": fatigue, "bmi": bmi, "oxygen_saturation": oxygen_sat,
        "exercise_hours_per_week": exercise, "diet_quality": diet, "alcohol_units_per_week": alcohol,
        "healthcare_access": healthcare
    }
   
    with st.spinner('Analyzing patient data...'):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                risk_prob = result['risk_probability'] * 100
               
                st.divider()
                if result['assessment'] == "High Risk":
                    st.error(f"### Assessment: HIGH RISK ({risk_prob:.1f}%)")
                    st.warning("Recommendation: Immediate clinical follow-up and imaging advised.")
                else:
                    st.success(f"### Assessment: LOW RISK ({risk_prob:.1f}%)")
                    st.info("Recommendation: Standard preventative care.")
            else:
                st.error(f"Error from API: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend. Is the FastAPI server running?")
