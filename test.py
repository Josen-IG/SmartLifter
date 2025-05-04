# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# --- 1) Configuración de la página
st.set_page_config(page_title="SmartLifter Demo", layout="wide")

# --- 2) Carga datos de ejercicios y modelo
@st.cache(allow_output_mutation=True)
def load_data_and_model():
    # Ejercicios limpios (asegúrate de que contiene: exercise_id, Title, Type, Level, Equipment)
    gym = pd.read_csv("C:/Users/josen/Documents/MASTER/TFM/Datasets/gym.csv")
    # Modelo pipeline: StandardScaler + XGBRegressor
    import os
    model_path = os.getenv("MODEL_PATH", "xgb_frequency_model_regularizado.pkl")
    model = joblib.load(model_path)
    return gym, model

gym, pipe = load_data_and_model()

# Prepara codificadores para las categorías
le_gender = LabelEncoder().fit(["Male","Female"])
le_type   = LabelEncoder().fit(gym["Type"])
le_level  = LabelEncoder().fit(["Beginner","Intermediate","Advanced"])

# --- 3) Sidebar: datos «crudos»
st.sidebar.header("Perfil de usuario")

age       = st.sidebar.number_input("Edad",           min_value=10,  max_value=100, value=30)
gender    = st.sidebar.selectbox("Género",           ["Male","Female"])
height_cm = st.sidebar.number_input("Altura (cm)",    min_value=120, max_value=220, value=170)
weight_kg = st.sidebar.number_input("Peso (kg)",      min_value=30,  max_value=200, value=70)
freq_w    = st.sidebar.number_input("Días/semana gym", 1,       7,        3)

# --- 4) Cálculos automáticos
bmi = weight_kg / (height_cm/100)**2
st.sidebar.markdown(f"**BMI:** {bmi:.1f}")

# Nivel estimado según frecuencia
if freq_w <= 2:
    level = "Beginner"
elif freq_w <= 4:
    level = "Intermediate"
else:
    level = "Advanced"
st.sidebar.markdown(f"**Nivel estimado:** {level}")

# --- 5) Preparar datos para predecir frecuencia de TODOS los ejercicios
# Creamos un DataFrame con la fila de usuario y luego lo «broadcast» a cada ejercicio
user_df = pd.DataFrame({
    "Age":    [age],
    "Gender": [le_gender.transform([gender])[0]],
    "BMI":    [bmi],
})
# Repetimos esa fila para cada ejercicio, añadiendo Type y Level codificados
recs = []
for _, ex in gym.iterrows():
    row = user_df.copy()
    row["Type"]  = le_type.transform([ex["Type"]])[0]
    row["Level"] = le_level.transform([ex["Level"]])[0]
    recs.append(row)
X_all = pd.concat(recs, ignore_index=True)

# --- 6) Predecir frecuencia
freq_pred = pipe.predict(X_all)
# Añadimos resultados al DataFrame de ejercicios
gym["Estimated_freq"] = freq_pred

# --- 7) Seleccionar top‐5
top5 = gym.nlargest(5, "Estimated_freq")

# --- 8) Mostrar resultados
st.title("💪 Exercise Routine Recommender Demo")
st.write("Basado en tus datos, aquí tienes 5 ejercicios recomendados:")

for _, ex in top5.iterrows():
    st.subheader(ex["Title"])
    st.markdown(f"- **Tipo:** {ex['Type']}   ·   **Nivel:** {ex['Level']}")
    st.markdown(f"- **Equipamiento:** {ex['Equipment']}   ·   **Frecuencia/sem:** {ex['Estimated_freq']:.2f}")
    st.write("---")
