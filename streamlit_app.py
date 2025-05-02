import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 🎯 Modeli yükle
model = pickle.load(open("best_model.pkl", "rb"))

# 📌 Kullanılacak giriş özellikleri (normalize edilmiş)
selected_features = {
    'Data_Value_Alt': "Obezite Oranı (Normalize)",
    'Low_Confidence_Limit': "Alt Güven Sınırı",
    'High_Confidence_Limit': "Üst Güven Sınırı",
    'Sample_Size': "Örneklem Büyüklüğü",
    'YearEnd': "Yıl (Eski: 0, Yeni: 1)"
}

st.set_page_config(page_title="Obezite Risk Tahmini", layout="centered")
st.title("🔍 Obezite Risk Tahmini")
st.markdown("Bu uygulama, girdiğiniz normalize edilmiş değerlere göre obezite riski taşıyıp taşımadığınızı tahmin eder.")

# 🧮 Kullanıcıdan veri al
input_data = {}
st.markdown("### 🎛 Değerleri Giriniz")
for key, label in selected_features.items():
    input_data[key] = st.slider(label, min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# 🎯 Tahmini hesapla
input_df = pd.DataFrame([input_data])

if st.button("📊 Tahmini Göster"):
    prediction = model.predict(input_df)[0]

    if prediction:
        st.error("⚠️ Obezite Riski Var", icon="⚠️")
    else:
        st.success("✅ Obezite Riski Yok", icon="✅")

    # Girdi özetini göster
    st.markdown("### 📋 Girdi Özeti")
    st.json(input_data)
