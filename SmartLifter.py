import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ——————————————————————————————————————————————————————
# 1) Carga de datos y modelo
# ——————————————————————————————————————————————————————
@st.cache_data
def load_exercises():
    return pd.read_csv("C:/Users/josen/Documents/MASTER/TFM/SmartLifter.csv")

@st.cache_resource
def load_model():
    return joblib.load("xgb_frequency_model_regularizado.pkl")

# Cargar datos y modelo
exercises = load_exercises()
model = load_model()

# mapa para transformar Level a número
level_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}

# ——————————————————————————————————————————————————————
# 2) Sidebar: inputs de usuario
# ——————————————————————————————————————————————————————
st.sidebar.header("Perfil de usuario")

age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30, step=1)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

height_cm = st.sidebar.number_input("Height (cm)", min_value=100, max_value=230, value=170)
weight_kg = st.sidebar.number_input("Weight (kg)", min_value=40, max_value=200, value=70)

# calculamos el BMI sobre la marcha
bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
st.sidebar.markdown(f"**BMI:** {bmi}")

level = st.sidebar.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])

frequency = st.sidebar.number_input(
    "Gym sessions per week", min_value=1, max_value=7, value=3, step=1
)

# ——————————————————————————————————————————————————————
# 3) Preparamos el DataFrame de predicción
# ——————————————————————————————————————————————————————
df_pred = exercises.copy()

# asignamos los inputs fijos
df_pred["Age"] = age
df_pred["Gender"] = gender
df_pred["BMI"] = bmi
df_pred["Level"] = level
df_pred["Frequency"] = frequency

# nuevas features
df_pred["Age_x_Level"] = df_pred["Age"] * level_map[level]

# BMI de categorías
bmi_bins  = [0, 18.5, 25, 30, np.inf]
bmi_labels = [0, 1, 2, 3]  # Underweight, Normal, Overweight, Obese
df_pred["BMI_cat"] = (
    pd.cut(df_pred["BMI"], bins=bmi_bins, labels=bmi_labels)
      .astype(int)
)

# Frequency de categorías
freq_bins  = [0, 2, 4, np.inf]
freq_labels = [0, 1, 2]  # Low, Medium, High
df_pred["Freq_cat"] = (
    pd.cut(df_pred["Frequency"], bins=freq_bins, labels=freq_labels)
      .astype(int)
)

# ——————————————————————————————————————————————————————
# 4) Codificar variables categóricas
# ——————————————————————————————————————————————————————
# Pre-entrenamos tres LabelEncoders para género, nivel y tipo de ejercicio
le_gender = LabelEncoder().fit(["Male", "Female"])
le_level  = LabelEncoder().fit(["Beginner", "Intermediate", "Advanced"])
le_type   = LabelEncoder().fit(exercises["Type"].unique())
le_bmi    = LabelEncoder().fit([0,1,2,3])
le_freq   = LabelEncoder().fit([0,1,2])

# aplicamos la transformación
df_pred["Gender"]  = le_gender.transform(df_pred["Gender"])
df_pred["Level"]   = le_level.transform(df_pred["Level"])
df_pred["Type"]    = le_type.transform(df_pred["Type"])
df_pred["BMI_cat"] = le_bmi.transform(df_pred["BMI_cat"])
df_pred["Freq_cat"]= le_freq.transform(df_pred["Freq_cat"])

# ——————————————————————————————————————————————————————
# 5) Predicción de la frecuencia deseada por ejercicio
# ——————————————————————————————————————————————————————
features = [
    "Age", "Gender", "BMI", "Type", "Level",
    "Age_x_Level", "BMI_cat", "Freq_cat"
]
df_pred["score"] = model.predict(df_pred[features])

# ——————————————————————————————————————————————————————
# 6) Selección de TOP-5 y display en Streamlit
# ——————————————————————————————————————————————————————
st.title("Recomendador de rutinas")
st.write("En base a tu perfil, estos son los 5 mejores ejercicios para ti:")

top5 = df_pred.sort_values("score", ascending=False).head(5)

for _, row in top5.iterrows():
    st.subheader(f"{row['Title']}")
    # deshacemos el encoding rápido para mostrar texto legible
    tipo   = le_type.inverse_transform([int(row["Type"])])[0]
    niv    = le_level.inverse_transform([int(row["Level"])])[0]
    equip  = row["Equipment"]
    freq_e = row["score"]
    st.write(f"- **Type:** {tipo}  |  **Level:** {niv}")
    st.write(f"- **Equipment:** {equip}  |  **Est. sessions/week:** {freq_e:.2f}")