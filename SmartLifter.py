import os
import streamlit as st
import pandas as pd
import joblib
import groq
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict, deque

# ────────────────────────────────────────────────────────────────────────────
# 1) Explicación de la rutina (usa Groq solo para la explicación)
# ────────────────────────────────────────────────────────────────────────────
load_dotenv()  # Carga automáticamente las variables definidas en .env
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("La variable de entorno GROQ_API_KEY no está definida.")
client = groq.Client(api_key=GROQ_API_KEY)

def explicar_rutina(schedule: dict, user_profile: dict) -> str:
    rutina_str = ""
    for día, ejercicios in schedule.items():
        rutina_str += f"{día}: {', '.join(ejercicios)}\n"
    prompt = f"""
    Eres un entrenador personal virtual. Basándote en el siguiente perfil y rutina, crea una explicación en español 
    orientada al usuario que explique:
    - Por qué esta rutina se adapta a su perfil (nivel: {user_profile['Level_es']}, edad: {user_profile['Age']} años, IMC: {user_profile['BMI']}). 
    - Cómo distribuye el trabajo de grupos musculares para evitar sobrecargas.
    - La lógica de la cantidad de ejercicios por día según el tiempo que indicó ({user_profile['Workout_minutes']} min).
    - Qué beneficios generales obtendrá si sigue esta rutina.

    Perfil del usuario:
    • Género: {user_profile['Gender_es']}
    • Nivel: {user_profile['Level_es']}
    • Edad: {user_profile['Age']}
    • Peso: {user_profile['weight_kg']} kg
    • Altura: {user_profile['Height_m']} m
    • Frecuencia semanal: {user_profile['Frequency']} días
    • IMC aproximado: {user_profile['BMI']}
    • Minutos por sesión: {user_profile['Workout_minutes']}

    Rutina generada:
    {rutina_str}

    Responde en párrafos claros, en español, sin listas numeradas: solo texto que explique en un lenguaje amigable para el usuario.
    """
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Eres un entrenador personal experto."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# ────────────────────────────────────────────────────────────────────────────
# 2) Cargar modelo y encoders (desde SmartLifter.pkl) + leer CSV de ejercicios
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_smartlifter_pickle(path: str):
    contenido = joblib.load(path)
    if not isinstance(contenido, dict):
        raise ValueError("SmartLifter.pkl no contiene un dict con 'model' y 'encoders'.")
    for key in ["model", "encoders"]:
        if key not in contenido:
            raise ValueError(f"Falta la clave '{key}' en el pickle.")
    return contenido["model"], contenido["encoders"]

@st.cache_data
def load_exercises_from_csv(path: str, _encoders: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {
        "Title", "Gender", "Level", "Age", "Weight_kg", "Height_m",
        "Frequency", "BMI", "BodyGroup", "BodyPart", "Equipment"
    }
    faltantes = expected_cols - set(df.columns)
    if faltantes:
        raise ValueError(f"Algunos campos faltan en SmartLifter.csv: {faltantes}")
    df["Gender_code"]    = _encoders["Gender"].transform(df["Gender"])
    df["Level_code"]     = _encoders["Level"].transform(df["Level"])
    df["BodyGroup_code"] = _encoders["BodyGroup"].transform(df["BodyGroup"])
    df["BodyPart_code"]  = _encoders["BodyPart"].transform(df["BodyPart"])
    df["Equipment_code"] = _encoders["Equipment"].transform(df["Equipment"])
    df = df.drop_duplicates(subset="Title").reset_index(drop=True)
    return df

# ────────────────────────────────────────────────────────────────────────────
# 3) Función para pedir datos al usuario en la barra lateral 
# ────────────────────────────────────────────────────────────────────────────
def get_user_input():
    try:
        st.sidebar.image("id.png", width=150)
    except:
        st.sidebar.error("Imagen 'id.png' no encontrada. Verifique la ruta y el nombre del archivo.")
    st.sidebar.header("Parámetros del usuario")
    gender_map = {"Hombre": "Male", "Mujer": "Female"}
    level_map  = {"Principiante": "Beginner", "Intermedio": "Intermediate", "Avanzado": "Advanced"}
    gender_es = st.sidebar.selectbox("Género", list(gender_map.keys()), index=0)
    level_es  = st.sidebar.selectbox("Nivel de entrenamiento", list(level_map.keys()), index=1)
    age             = st.sidebar.slider("Edad", 17, 70, 26, 1)
    weight_kg       = st.sidebar.number_input("Peso (kg)", 40.0, 150.0, 70.0, 0.5)
    height_m        = st.sidebar.number_input("Altura (m)", 1.40, 2.20, 1.75, 0.01)
    frequency       = st.sidebar.slider("Frecuencia semanal (días)", 1, 7, 3, 1)
    workout_minutes = st.sidebar.slider("Minutos por sesión", 30, 120, 60, 5)
    bmi = weight_kg / (height_m ** 2)
    return {
        "Gender": gender_map[gender_es],
        "Level": level_map[level_es],
        "Age": age,
        "weight_kg": weight_kg,
        "Height_m": height_m,
        "Frequency": frequency,
        "Workout_minutes": workout_minutes,
        "BMI": round(bmi, 1),
        "Gender_es": gender_es,
        "Level_es": level_es,
    }

# ────────────────────────────────────────────────────────────────────────────
# 4) Función generate_routine 
# ────────────────────────────────────────────────────────────────────────────
def generate_routine(user_profile, all_ex_df: pd.DataFrame, model, encoders, seed=42):
    import numpy as np

    # 4.1) Validar edad
    age = user_profile["Age"]
    if not 17 <= age <= 70:
        raise ValueError("La edad debe estar entre 17 y 70 años.")

    # 4.2) Calcular cuántos ejercicios por día según tiempo disponible
    total_time       = user_profile.get("Workout_minutes", 60)
    warmup           = 10
    rest_per_ex      = 2
    available_time   = total_time - warmup
    approx_ex_time   = 5
    exercises_per_day = max(3, min(6, available_time // (approx_ex_time + rest_per_ex)))

    # 4.3) Codificar género y nivel
    g = encoders["Gender"].transform([user_profile["Gender"]])[0]
    level_input = user_profile["Level"]
    levels_priority = ["Advanced", "Intermediate", "Beginner"]
    available_levels = [lvl for lvl in levels_priority if lvl in encoders["Level"].classes_]
    if level_input not in available_levels:
        level_input = available_levels[0]

    # 4.4) Pre-filtrado general por género, nivel y características físicas
    df_filtered_base = all_ex_df[
        (all_ex_df["Gender_code"] == g) &
        (all_ex_df["Level_code"] == encoders["Level"].transform([level_input])[0]) &
        (abs(all_ex_df["Age"] - age) <= 3) &
        (abs(all_ex_df["Weight_kg"] - user_profile["weight_kg"]) <= 5) &
        (abs(all_ex_df["Height_m"] - user_profile["Height_m"]) <= 0.05)
    ].copy()
    if df_filtered_base.empty:
        df_filtered_base = all_ex_df.copy()

    # 4.5) Mapeo de BodyGroup 
    split_map = {
        "Chest":      "Push",
        "Shoulders":  "Push",
        "Triceps":    "Push",
        "Biceps":     "Pull",
        "Lats":       "Pull",
        "Middle Back":"Pull",
        "Lower Back": "Pull",
        "Traps":      "Pull",
        "Forearms":   "Pull",
        "Legs":       "Legs",
        "Quadriceps": "Legs",
        "Hamstrings": "Legs",
        "Glutes":     "Legs",
        "Calves":     "Legs",
        "Adductors":  "Legs",
        "Abductors":  "Legs",
        "Abdominals": "Core",
        "Neck":       "Otros",
        "Otros":      "Otros"
    }

    # 4.6) Generar secuencia de splits según frecuencia
    def build_day_sequence(freq):
        if freq == 1:
            return ["Full"]
        elif freq == 2:
            return ["Upper", "Lower"]
        elif freq == 3:
            return ["Push", "Legs", "Pull"]
        elif freq == 4:
            return ["Push", "Legs", "Pull", "Core"]
        elif freq == 5:
            return ["Push", "Legs", "Pull", "Legs", "Core"]
        elif freq == 6:
            return ["Push", "Legs", "Pull", "Legs", "Upper", "Core"]
        else:
            return ["Push", "Legs", "Pull", "Legs", "Upper", "Core", "Lower"]

    days_splits = build_day_sequence(user_profile["Frequency"])

    # 4.7) Columnas de features para la predicción
    feature_cols = [
        "Age", "Gender_code", "Weight_kg", "Height_m", "Frequency", "BMI",
        "Level_code", "Type_x_code", "BodyGroup_code", "BodyPart_code", "Equipment_code"
    ]

    # 4.8) Variables auxiliares para evitar repeticiones
    rng = np.random.RandomState(seed)
    used_titles_week = set()
    used_bodyparts_week = set()
    last_trained = {k: -99 for k in ["Push", "Pull", "Legs", "Core", "Full", "Upper", "Lower"]}

    schedule = {}

    # 4.9) Iterar por cada día y asignar ejercicios
    for i, split in enumerate(days_splits):
        día_label = f"Día {i+1}"

        # 4.9.1) Forzar descanso si no han pasado 2 días desde la última vez que entrenamos este split
        if i - last_trained.get(split, -99) < 2:
            schedule[día_label] = []
            last_trained[split] = i
            continue

        # 4.9.2) Filtrar ejercicios según el split, usando BodyPart
        if split == "Full":
            df_day = df_filtered_base.copy()

        elif split == "Upper":
            # Upper = todos los BodyPart que no sean de piernas
            upper_parts = ["Chest", "Shoulders", "Triceps",
                           "Biceps", "Lats", "Middle Back",
                           "Lower Back", "Traps", "Forearms",
                           "Neck", "Abdominals"]
            df_day = df_filtered_base[df_filtered_base["BodyPart"].isin(upper_parts)].copy()

        elif split == "Lower":
            # Lower = todos los BodyPart que pertenezcan a piernas
            lower_parts = ["Legs", "Quadriceps", "Hamstrings",
                           "Glutes", "Calves", "Adductors", "Abductors"]
            df_day = df_filtered_base[df_filtered_base["BodyPart"].isin(lower_parts)].copy()

        elif split == "Core":
            # Incluir tanto abdominales como lumbares
            df_day = df_filtered_base[
                df_filtered_base["BodyPart"].isin(["Abdominals", "Lower Back"])
            ].copy()

        elif split == "Otros":
            # Otros = partes accesorias (cuello, antebrazos, etc.)
            otros_parts = ["Neck", "Forearms"]
            df_day = df_filtered_base[df_filtered_base["BodyPart"].isin(otros_parts)].copy()

        else:
            # Push, Pull o Legs
            if split == "Push":
                push_parts = ["Chest", "Shoulders", "Triceps"]
                df_day = df_filtered_base[df_filtered_base["BodyPart"].isin(push_parts)].copy()
            elif split == "Pull":
                pull_parts = ["Biceps", "Lats", "Middle Back", "Lower Back", "Traps", "Forearms"]
                df_day = df_filtered_base[df_filtered_base["BodyPart"].isin(pull_parts)].copy()
            elif split == "Legs":
                legs_parts = ["Legs", "Quadriceps", "Hamstrings", "Glutes", "Calves", "Adductors", "Abductors"]
                df_day = df_filtered_base[df_filtered_base["BodyPart"].isin(legs_parts)].copy()
            else:
                # Si aparece algún split inesperado, caer a full-body
                df_day = df_filtered_base.copy()

        # 4.9.3) Si no hay ejercicios de ese split, caer a full-body
        if df_day.empty:
            df_day = df_filtered_base.copy()

        # 4.9.4) Preparar X_pred_day con las columnas de features y códigos
        X_pred_day = df_day.copy()
        X_pred_day["Gender_code"]    = g
        X_pred_day["Level_code"]     = encoders["Level"].transform([level_input])[0]
        X_pred_day["Age"]            = age
        X_pred_day["Weight_kg"]      = user_profile["weight_kg"]
        X_pred_day["Height_m"]       = user_profile["Height_m"]
        X_pred_day["Frequency"]      = user_profile["Frequency"]
        X_pred_day["Type_x_code"]    = encoders["Type_x"].transform(["Strength"])[0]
        X_pred_day["BMI"]            = X_pred_day["Weight_kg"] / (X_pred_day["Height_m"]**2)
        X_pred_day["BodyGroup_code"] = encoders["BodyGroup"].transform(X_pred_day["BodyGroup"])
        X_pred_day["BodyPart_code"]  = encoders["BodyPart"].transform(X_pred_day["BodyPart"])

        # 4.9.5) Calcular proba y score
        proba_day     = model.predict_proba(X_pred_day[feature_cols])
        title_classes = model.classes_
        scores = []
        for idx, (_, row) in enumerate(X_pred_day.iterrows()):
            title = row["Title"]
            if title in title_classes:
                class_idx = list(title_classes).index(title)
                scores.append(proba_day[idx][class_idx])
            else:
                scores.append(0)
        X_pred_day["score"] = scores

        # 4.9.6) Ordenar por score descendente
        ranked_day = X_pred_day.sort_values("score", ascending=False).reset_index(drop=True)

        # 4.9.7) Selección de ejercicios en 3 fases
        chosen = []
        bodyparts_day = set()

        # Fase 1: no repetir BodyPart local ni Title en la semana
        for _, row in ranked_day.iterrows():
            title    = row["Title"]
            bodypart = row["BodyPart"]
            if title not in used_titles_week and bodypart not in bodyparts_day:
                chosen.append((title, bodypart))
                used_titles_week.add(title)
                used_bodyparts_week.add(bodypart)
                bodyparts_day.add(bodypart)
            if len(chosen) == exercises_per_day:
                break

        # Fase 2: si faltan, solo evitamos repetir Title en la semana
        ptr = 0
        while len(chosen) < exercises_per_day and ptr < len(ranked_day):
            title = ranked_day.loc[ptr, "Title"]
            if title not in used_titles_week:
                chosen.append((title, ranked_day.loc[ptr, "BodyPart"]))
                used_titles_week.add(title)
            ptr += 1

        # Fase 3: si aún faltan, rellenar desde top 200 sin repetir Title en la semana
        if len(chosen) < exercises_per_day:
            top200 = ranked_day["Title"].head(200).tolist()
            candidates = [t for t in top200 if t not in used_titles_week]
            needed = exercises_per_day - len(chosen)
            extras = rng.choice(candidates, size=needed, replace=False).tolist()
            for ex in extras:
                grp = ranked_day[ranked_day["Title"] == ex]["BodyPart"].iloc[0]
                chosen.append((ex, grp))
                used_titles_week.add(ex)

        # ─────────────────────────────────────────────────────────────────────────
        # 4.9.8) INTERCALAR BODYPARTS
        grupo_por_bodypart = defaultdict(list)
        for título, bp in chosen:
            grupo_por_bodypart[bp].append((título, bp))

        colas = {bp: deque(lista) for bp, lista in grupo_por_bodypart.items()}
        manejo_colas = [(bp, colas[bp]) for bp in colas]
        reordenado = []
        idx_cola = 0

        while manejo_colas:
            bp_actual, cola_actual = manejo_colas[idx_cola]
            if cola_actual:
                reordenado.append(cola_actual.popleft())
                if cola_actual:
                    idx_cola = (idx_cola + 1) % len(manejo_colas)
                else:
                    manejo_colas.pop(idx_cola)
                    if manejo_colas:
                        idx_cola %= len(manejo_colas)
            else:
                manejo_colas.pop(idx_cola)
                if manejo_colas:
                    idx_cola %= len(manejo_colas)

        chosen = reordenado
        # ─────────────────────────────────────────────────────────────────────────

        # 4.9.9) Guardar ejercicios del día, formateados
        schedule[día_label] = [f"{ex} ({grp})" for ex, grp in chosen]
        last_trained[split] = i

    return schedule

# ────────────────────────────────────────────────────────────────────────────
# 5) Función principal de Streamlit
# ────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #121212 !important;
            color: #FFFFFF !important;
        }
        .main, .block-container {
            background-color: #121212 !important;
            color: #FFFFFF !important;
        }
        h1, h2, h3, h4, h5, h6, p, label, div, span {
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] {
            background-color: #1C1F2E !important;
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        .stButton>button {
            background-color: #2A2D3E !important;
            color: #FFFFFF !important;
        }
        .stButton>button:hover {
            background-color: #3A3E50 !important;
            color: #FFFFFF !important;
        }
        .stTextInput>div>div>input,
        .stSlider>div>div>div input {
            color: #FFFFFF !important;
            background-color: transparent !important;
            border: 1px solid #FFFFFF !important;
        }
        .stSlider>div>label {
            color: #FFFFFF !important;
        }
        ::-webkit-scrollbar {
            width: 8px;
            background-color: #1C1F2E;
        }
        ::-webkit-scrollbar-thumb {
            background-color: #3A3E50;
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center; color:#FFFFFF;'>SmartLifter</h1>", unsafe_allow_html=True)
    st.write("""
        Ajusta tus parámetros personales en la barra lateral y pulsa “Generar rutina” 
        para ver qué ejercicios tienes asignados cada día según tu perfil.
    """)

    try:
        model, encoders = load_smartlifter_pickle("SmartLifter.pkl")
    except Exception as e:
        st.error(f"❌ Error al cargar SmartLifter.pkl: {e}")
        st.stop()

    try:
        all_ex_df = load_exercises_from_csv("SmartLifter.csv", encoders)
    except Exception as e:
        st.error(f"❌ Error al cargar SmartLifter.csv: {e}")
        st.stop()

    user_input = get_user_input()

    st.subheader("Tus datos:")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(f"- Género: **{user_input['Gender_es']}**")
        st.write(f"- Nivel: **{user_input['Level_es']}**")
        st.write(f"- Edad: **{user_input['Age']}** años")
        st.write(f"- Peso: **{user_input['weight_kg']} kg**")
        st.write(f"- Altura: **{user_input['Height_m']} m**")
        st.write(f"- Frecuencia: **{user_input['Frequency']} días/semana**")
        st.write(f"- Minutos por sesión: **{user_input['Workout_minutes']} min**")
        st.write(f"- IMC aproximado: **{user_input['BMI']}**")
    with col2:
        try:
            st.image("bench.png", use_container_width=True)
        except:
            st.error("Imagen 'bench.png' no encontrada. Verifique la ruta y el nombre del archivo.")
    st.markdown("---")

    if st.button("🏋️‍♂️ Generar rutina"):
        try:
            schedule = generate_routine(
                user_profile=user_input,
                all_ex_df=all_ex_df,
                model=model,
                encoders=encoders,
                seed=42
            )
            dias = list(schedule.keys())
            max_ej = max(len(ejs) for ejs in schedule.values())
            tabla = {dia: [] for dia in dias}
            for dia in dias:
                lista_ejs = schedule[dia]
                for e in lista_ejs:
                    tabla[dia].append(e)
                while len(tabla[dia]) < max_ej:
                    tabla[dia].append("")
            df_tabla = pd.DataFrame(tabla)
            st.subheader("Rutina semanal:")
            st.dataframe(df_tabla)
            st.markdown("---")
            with st.spinner("Generando explicación de la rutina…"):
                explicación = explicar_rutina(schedule, user_input)
            st.subheader("¿Por qué esta rutina es adecuada para ti?")
            st.markdown(f"<div style='text-align: justify; color:#FFFFFF;'>{explicación}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ No se pudo generar la rutina: {e}")

if __name__ == "__main__":
    main()