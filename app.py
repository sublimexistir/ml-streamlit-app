import streamlit as st
import joblib
import numpy as np

# Cargar modelos (NUEVOS NOMBRES)
lr_model = joblib.load("modelos/prueba1_logistic_regression_model.pkl")
rf_model = joblib.load("modelos/prueba1_random_forest_model.pkl")

# Título
st.title("Predicción de Enfermedad Cardíaca")

# Información del estudiante
st.write("Nombre: Graciela Alzamora Gil")
st.write("NRC: 6817")

st.markdown(
    "[Ver Google Colab](https://colab.research.google.com/drive/1cYKEMHNFzubx86i5Nw1EuYu5EZbjoPKX?usp=sharing)"
)

st.subheader("Ingrese los datos del paciente")

# Selector de modelo
modelo = st.selectbox(
    "Seleccione el modelo",
    ["Logistic Regression", "Random Forest"]
)

# FEATURES (LAS 5 EXACTAS)
age = st.number_input("Age", min_value=1, max_value=120, value=30)
trestbps = st.number_input("Trestbps", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=700, value=200)
thalach = st.number_input("Max Heart Rate (thalach)", min_value=50, max_value=250, value=150)
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Botón de predicción
if st.button("Predecir"):

    # IMPORTANTE: orden EXACTO
    datos = np.array([[age, trestbps, chol, thalach, oldpeak]])

    # Selección de modelo
    if modelo == "Logistic Regression":
        pred = lr_model.predict(datos)
    else:
        pred = rf_model.predict(datos)

    # Resultado
    if pred[0] == 1:
        st.error("Predicción: posible enfermedad cardíaca")
    else:
        st.success("Predicción: no se detecta enfermedad cardíaca")
