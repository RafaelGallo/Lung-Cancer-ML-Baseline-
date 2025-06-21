import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ==== Load trained model ====
model_path = r'C:\Users\rafae.RAFAEL_NOTEBOOK\Downloads\github_projetos_deeplearning_CNN\Lung Cancer\models\Logistic_Regression.pkl'
model = joblib.load(model_path)

# ==== App Title ====
st.title("🚑 Lung Cancer Risk Prediction App")
st.markdown("""
This app predicts the **risk level of lung cancer** based on patient lifestyle and clinical factors.  
Fill in the details below and click **Predict Lung Cancer Risk**.
""")

# ==== Input Form ====
with st.form("prediction_form"):
    st.subheader("Patient Information")

    age = st.slider('Age', 20, 90, 50)
    gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    air_pollution = st.slider('Air Pollution Level', 1, 8, 4)
    alcohol_use = st.slider('Alcohol Use Level', 1, 8, 4)
    dust_allergy = st.slider('Dust Allergy Level', 1, 8, 4)
    occupational_hazards = st.slider('Occupational Hazards Level', 1, 8, 4)
    genetic_risk = st.slider('Genetic Risk Level', 1, 8, 4)
    chronic_lung_disease = st.slider('Chronic Lung Disease Level', 1, 8, 4)
    balanced_diet = st.slider('Balanced Diet Level', 1, 8, 4)
    obesity = st.slider('Obesity Level', 1, 8, 4)
    smoking = st.slider('Smoking Level', 1, 8, 4)
    passive_smoker = st.slider('Passive Smoker Level', 1, 8, 4)
    chest_pain = st.slider('Chest Pain Level', 1, 8, 4)
    coughing_of_blood = st.slider('Coughing of Blood Level', 1, 8, 4)
    fatigue = st.slider('Fatigue Level', 1, 8, 4)
    weight_loss = st.slider('Weight Loss Level', 1, 8, 4)
    shortness_of_breath = st.slider('Shortness of Breath Level', 1, 8, 4)
    wheezing = st.slider('Wheezing Level', 1, 8, 4)
    swallowing_difficulty = st.slider('Swallowing Difficulty Level', 1, 8, 4)
    clubbing_of_finger_nails = st.slider('Clubbing of Finger Nails Level', 1, 8, 4)
    frequent_cold = st.slider('Frequent Cold Level', 1, 8, 4)
    dry_cough = st.slider('Dry Cough Level', 1, 8, 4)
    snoring = st.slider('Snoring Level', 1, 8, 4)

    submit = st.form_submit_button("Predict Lung Cancer Risk")

# ==== Prediction ====
if submit:
    # Build input dataframe
    input_data = pd.DataFrame([[  
        age, gender, air_pollution, alcohol_use, dust_allergy, occupational_hazards,
        genetic_risk, chronic_lung_disease, balanced_diet, obesity, smoking,
        passive_smoker, chest_pain, coughing_of_blood, fatigue, weight_loss,
        shortness_of_breath, wheezing, swallowing_difficulty,
        clubbing_of_finger_nails, frequent_cold, dry_cough, snoring
    ]], columns=model.feature_names_in_)

    # Handle missing values
    input_data = input_data.fillna(0).infer_objects(copy=False)

    # Prediction
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    # Map numeric class to human-readable label
    class_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    predicted_label = class_labels.get(prediction, prediction)

    # ==== Display results ====
    st.success(f"Predicted Cancer Risk Level: **{predicted_label}**")

    st.markdown(f"""
    ### Prediction Probabilities:
    - Class 0 (Low): {proba[0]*100:.2f}%
    - Class 1 (Medium): {proba[1]*100:.2f}%
    - Class 2 (High): {proba[2]*100:.2f}%
    """)

    # ==== Save log of predictions ====
    log_df = input_data.copy()
    log_df['predicted_class'] = predicted_label
    log_df.to_csv('user_predictions_log.csv', mode='a', index=False)
    st.info('Prediction logged to user_predictions_log.csv ✅')
