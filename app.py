import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your saved model and scaler
model = joblib.load('best_academic_performance_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🎓 Student Academic Performance Prediction App")

st.write("Enter student behaviour details below to predict if they are likely to pass or fail.")

# Collect user inputs
study_hours = st.number_input("Study Hours per Day", min_value=0.0, step=0.5)
social_media = st.number_input("Social Media Hours per Day", min_value=0.0, step=0.5)
netflix_hours = st.number_input("Netflix Hours per Day", min_value=0.0, step=0.5)
part_time_job = st.selectbox("Part-time Job", ["Yes", "No"])
attendance = st.slider("Attendance Percentage", 0, 100)
sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0, step=0.5)
diet_quality = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
exercise_freq = st.number_input("Exercise Frequency (per week)", min_value=0)
extracurricular = st.selectbox("Extracurricular Participation", ["Yes", "No"])

# Encode categorical inputs manually to match your training
part_time_job_enc = 1 if part_time_job == "Yes" else 0
diet_mapping = {"Poor": 0, "Average": 1, "Good": 2}
diet_quality_enc = diet_mapping[diet_quality]
extracurricular_enc = 1 if extracurricular == "Yes" else 0

# Arrange the inputs into a DataFrame
input_data = pd.DataFrame({
    'study_hours_per_day': [study_hours],
    'social_media_hours': [social_media],
    'netflix_hours': [netflix_hours],
    'part_time_job': [part_time_job_enc],
    'attendance_percentage': [attendance],
    'sleep_hours': [sleep_hours],
    'diet_quality': [diet_quality_enc],
    'exercise_frequency': [exercise_freq],
    'extracurricular_participation': [extracurricular_enc]
})

# Scale the inputs
input_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_scaled)[0]

# Display result
if st.button("Predict"):
    if prediction == 1:
        st.success("✅ The student is likely to **PASS** the exam.")
    else:
        st.error("❌ The student is likely to **FAIL** the exam.")
