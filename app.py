import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Ev Fiyat Tahmini", page_icon="🏠")
st.title("🏠 Ridge Regresyon ile Ev Fiyat Tahmini")

# MODELİ YÜKLE
@st.cache_resource
def load_model():
    data = joblib.load("ridge_model_full.pkl")
    return data["model"], data["column_means"], data["columns"]

model, column_means, column_order = load_model()

# 🔍 Test – Model tipi
st.write("📦 Model tipi:", type(model))
st.write("🧩 Model kolonları:", model.feature_names_in_)

# FORM GİRİŞLERİ
st.subheader("Ev Özelliklerini Girin")

grlivarea = st.number_input("Yaşanabilir Alan (GrLivArea)", min_value=0)
garagecars = st.number_input("Garaj Kapasitesi (GarageCars)", min_value=0)
fullbath = st.number_input("Tam Banyo Sayısı (FullBath)", min_value=0)
yearbuilt = st.number_input("İnşa Yılı (YearBuilt)", min_value=1800, max_value=2025)
overallqual = st.slider("Genel Kalite (1–10)", 1, 10, 5)

# Ek alanlar
totalbsmt = st.number_input("Bodrum Alanı (TotalBsmtSF)", min_value=0)
garagearea = st.number_input("Garaj Alanı (GarageArea)", min_value=0)

exterqual = st.selectbox("Dış Kalite (ExterQual)", ["Po", "Fa", "TA", "Gd", "Ex"])
qual_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

if st.button("Tahmin Et"):
    # Ortalama satırı kopyala
    input_data = column_means.copy()

    # Kullanıcının verileri
    input_data["GrLivArea"] = grlivarea
    input_data["GarageCars"] = garagecars
    input_data["FullBath"] = fullbath
    input_data["YearBuilt"] = yearbuilt
    input_data["OverallQual"] = overallqual
    input_data["TotalBsmtSF"] = totalbsmt
    input_data["GarageArea"] = garagearea
    input_data["ExterQual"] = qual_map[exterqual]

    # DataFrame’e çevir
    input_df = pd.DataFrame([input_data])

    # Sıralamayı modele göre ayarla
    try:
        input_df = input_df[model.feature_names_in_]
    except Exception as e:
        st.error(f"❌ Kolon eşleşme hatası: {e}")
        st.stop()

    # 🔍 Tahmin öncesi veriyi göster
    st.write("📋 Modele gönderilen veri:")
    st.dataframe(input_df)

    # 🔍 Tahmin
    try:
        prediction_log = model.predict(input_df)
        prediction = np.expm1(prediction_log)  # log dönüşüm varsa doğru sonuç için bu

        st.success(f"🏷️ Tahmini Ev Fiyatı: ${prediction[0]:,.0f}")
    except Exception as e:
        st.error(f"⚠️ Tahmin sırasında hata: {e}")
