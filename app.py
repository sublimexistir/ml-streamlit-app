import streamlit as st
import joblib
import numpy as np

# Cargar modelos
lr_model = joblib.load("modelos/logistic_regression.pkl")
rf_model = joblib.load("modelos/random_forest.pkl")

# Título
st.title("Predicción de Enfermedad Cardíaca")

# Información del estudiante
st.write("Nombre: Graciela Alzamora Gil")
st.write("NRC: 6817")

# Link del Colab
st.markdown(
    "[Ver Google Colab](https://colab.research.google.com/drive/1cYKEMHNFzubx86i5Nw1EuYu5EZbjoPKX?usp=sharing)"
)

st.subheader("Ingrese los datos del paciente")

# Selección de modelo
modelo = st.selectbox(
    "Seleccione el modelo",
    ["Logistic Regression", "Random Forest"]
)

# Inputs
age = st.number_input("Edad", min_value=1, max_value=120, value=30)

sex = st.selectbox(
    "Sexo",
    [0, 1],
    format_func=lambda x: "Femenino" if x == 0 else "Masculino"
)

cp = st.number_input("Tipo de dolor de pecho (cp)", min_value=0, max_value=3)

trestbps = st.number_input(
    "Presión arterial en reposo (trestbps)",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Colesterol (chol)",
    min_value=100,
    max_value=700,
    value=200
)

fbs = st.selectbox(
    "Azúcar en sangre en ayunas > 120 mg/dl (fbs)",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Sí"
)

restecg = st.number_input(
    "Resultado electrocardiograma (restecg)",
    min_value=0,
    max_value=2
)

thalach = st.number_input(
    "Frecuencia cardíaca máxima (thalach)",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Angina inducida por ejercicio (exang)",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Sí"
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.number_input(
    "Slope",
    min_value=0,
    max_value=2
)

ca = st.number_input(
    "Número de vasos principales (ca)",
    min_value=0,
    max_value=4
)

thal = st.number_input(
    "Thal",
    min_value=0,
    max_value=3
)

# Botón de predicción
if st.button("Predecir"):

    datos = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    # Selección del modelo
    if modelo == "Logistic Regression":
        prediccion = lr_model.predict(datos)
    else:
        prediccion = rf_model.predict(datos)

    # Resultado
    if prediccion[0] == 1:
        st.error("El modelo predice posible enfermedad cardíaca.")
    else:
        st.success("El modelo NO predice enfermedad cardíaca.")