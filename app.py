import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Ev Fiyat Tahmini", page_icon="🏠")
st.title("🏠 Ridge Regresyon ile Ev Fiyat Tahmini")

# MODEL + ortalama + kolon listesini yükle
@st.cache_resource
def load_model():
    data = joblib.load("ridge_model_full.pkl")
    return data["model"], data["column_means"], data["columns"]

model, column_means, column_order = load_model()

# KULLANICIDAN ALINACAK DEĞERLER
st.subheader("Ev Özelliklerini Girin")

grlivarea = st.number_input("Yaşanabilir Alan (GrLivArea)", min_value=0)
garagecars = st.number_input("Garaj Kapasitesi (GarageCars)", min_value=0)
fullbath = st.number_input("Tam Banyo Sayısı (FullBath)", min_value=0)
yearbuilt = st.number_input("İnşa Yılı (YearBuilt)", min_value=1800, max_value=2025)
overallqual = st.slider("Genel Kalite (1–10)", 1, 10, 5)

# Kullanıcıdan alınacak yeni değişkenler
totalbsmt = st.number_input("Bodrum Alanı (TotalBsmtSF)", min_value=0)
firstflr = st.number_input("1. Kat Alanı (1stFlrSF)", min_value=0)
garagearea = st.number_input("Garaj Alanı (GarageArea)", min_value=0)
kitchen = st.number_input("Mutfak Sayısı (KitchenAbvGr)", min_value=0)
fireplaces = st.number_input("Şömine Sayısı (Fireplaces)", min_value=0)

exterqual = st.selectbox("Dış Kalite (ExterQual)", ["Po", "Fa", "TA", "Gd", "Ex"])
bsmtqual = st.selectbox("Bodrum Kalite (BsmtQual)", ["NA", "Po", "Fa", "TA", "Gd", "Ex"])

# TAHMİN
if st.button("Tahmin Et"):
    # Ortalama değerlerle başla
    input_data = column_means.copy()

    # Kullanıcıdan alınanları gir
    input_data["GrLivArea"] = grlivarea
    input_data["GarageCars"] = garagecars
    input_data["FullBath"] = fullbath
    input_data["YearBuilt"] = yearbuilt
    input_data["OverallQual"] = overallqual

    input_data["TotalBsmtSF"] = totalbsmt
    input_data["1stFlrSF"] = firstflr
    input_data["GarageArea"] = garagearea
    input_data["KitchenAbvGr"] = kitchen
    input_data["Fireplaces"] = fireplaces

    # Kategorik değişkenleri sayıya çevir (label encoding)
    qual_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "NA": 0}
    input_data["ExterQual"] = qual_map[exterqual]
    input_data["BsmtQual"] = qual_map[bsmtqual]
    
    # DataFrame ve sıralama
    input_df = pd.DataFrame([input_data])
    input_df = input_df[model.feature_names_in_]  # 🔥 EN ÖNEMLİ SATIR!

    # Tahmin
      # 🔥 Eğer log dönüşümlü eğitildiyse:
    prediction = np.expm1(model.predict(input_df))  # log dönüşüm varsa bu

    st.success(f"🏷️ Tahmini Ev Fiyatı: ${prediction[0]:,.0f}")
