import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Tomato Soil Health Prediction", page_icon="🌱")

st.title("🌱 Tomato Soil Health Prediction System")
st.write("Enter the soil details below and click Predict.")

sample_id = st.number_input("Sample ID", value=101)
soil_ph = st.number_input("Soil pH", value=6.8)
moisture = st.number_input("Moisture (%)", value=65.0)
nitrogen = st.number_input("Nitrogen (mg/kg)", value=45.0)
phosphorus = st.number_input("Phosphorus (mg/kg)", value=30.0)
potassium = st.number_input("Potassium (mg/kg)", value=180.0)
temperature = st.number_input("Temperature (°C)", value=27.0)
organic_matter = st.number_input("Organic Matter (%)", value=4.2)

if st.button("Predict Soil Health"):

    new_data = pd.DataFrame({
        "Sample_ID": [sample_id],
        "Soil_pH": [soil_ph],
        "Moisture_%": [moisture],
        "Nitrogen_mgkg": [nitrogen],
        "Phosphorus_mgkg": [phosphorus],
        "Potassium_mgkg": [potassium],
        "Temperature_C": [temperature],
        "Organic_Matter_%": [organic_matter]
    })

    new_data = pd.get_dummies(new_data)
    new_data = new_data.reindex(columns=feature_columns, fill_value=0)
    new_data = scaler.transform(new_data)

    prediction = model.predict(new_data)

    st.success(f"Predicted Soil Health: {prediction[0]}")
