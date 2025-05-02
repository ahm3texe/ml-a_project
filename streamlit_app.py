import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Eğitimde kaydedilen modeli yükle
model = pickle.load(open("best_model.pkl", "rb"))

# Kullanılacak özellikler
selected_features = ['YearStart', 'YearEnd', 'Data_Value_Alt', 'Low_Confidence_Limit', 'High_Confidence_Limit']

st.title("🔍 Obezite Risk Tahmini")
st.markdown("Girdiğiniz değerlere göre obezite riski olup olmadığını tahmin eder.")

# Kullanıcıdan veri al
input_data = {}
for feature in selected_features:
    input_data[feature] = st.slider(f"{feature}", min_value=0.0, max_value=1.0, step=0.01)

input_df = pd.DataFrame([input_data])

if st.button("Tahmini Göster"):
    prediction = model.predict(input_df)[0]
    result = "⚠️ Risk Var" if prediction else "✅ Risk Yok"
    st.success(f"Tahmin Sonucu: {result}")
