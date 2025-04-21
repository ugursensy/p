import streamlit as st
import pandas as pd
import numpy as np
import joblib  # ← EKLENMESİ GEREKEN SATIR

st.title("🏠 Ridge Regresyon ile Ev Fiyat Tahmini")

@st.cache_resource
def load_model():
    data = joblib.load("ridge_model_full.pkl")
    return data["model"], data["column_means"], data["columns"]

model, column_means, column_order = load_model()
# Girdi alanları
grlivarea = st.number_input("Yaşanabilir Alan (GrLivArea)", min_value=0)
garagecars = st.number_input("Garaj Kapasitesi", min_value=0)
fullbath = st.number_input("Tam Banyo Sayısı", min_value=0)
yearbuilt = st.number_input("İnşa Yılı", min_value=1800, max_value=2025)
overallqual = st.slider("Genel Kalite (1-10)", 1, 10, 5)

if st.button("Tahmin Et"):
    input_data = pd.DataFrame({
        "GrLivArea": [grlivarea],
        "GarageCars": [garagecars],
        "FullBath": [fullbath],
        "YearBuilt": [yearbuilt],
        "OverallQual": [overallqual]
    })
    
    prediction = model.predict(input_data)
    st.success(f"Tahmini Ev Fiyatı: ${prediction[0]:,.0f}")
