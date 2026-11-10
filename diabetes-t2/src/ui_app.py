# src/ui_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# -----------------------
# Config general
# -----------------------
st.set_page_config(page_title="Predicción de Diabetes Tipo 2", page_icon="🩺", layout="wide")
st.title("Pronóstico de diabetes tipo 2")
st.caption("Centro de Salud Huaycán · 2025 – Herramienta de apoyo (sin diagnóstico)")

# -----------------------
# Selector de modelo
# -----------------------
st.sidebar.header("del modelo")
opcion_modelo = st.sidebar.selectbox(
    "Elegir modelo",
    ["NHANES - Random Forest", "NHANES - Regresión Logística"],
    index=0
)

MAP_PATH = {
    "NHANES - Random Forest": (
        "models/model_nhanes_rf.joblib",
        "models/metrics_model_nhanes_rf.txt",
        "models/cache_nhanes_rf_ytest.npy",
        "models/cache_nhanes_rf_ptest.npy",
    ),
    "NHANES - Regresión Logística": (
        "models/model_nhanes_logreg.joblib",
        "models/metrics_model_nhanes_logreg.txt",
        "models/cache_nhanes_logreg_ytest.npy",
        "models/cache_nhanes_logreg_ptest.npy",
    ),
}
MODEL_PATH_STR, METRICS_PATH_STR, YTEST_PATH_STR, PTEST_PATH_STR = MAP_PATH[opcion_modelo]
MODEL_PATH, METRICS_PATH = Path(MODEL_PATH_STR), Path(METRICS_PATH_STR)
YTEST_PATH, PTEST_PATH   = Path(YTEST_PATH_STR), Path(PTEST_PATH_STR)

if not MODEL_PATH.exists():
    st.error(f"No se encontró el modelo: {MODEL_PATH_STR}. Ejecuta:  python src/train_nhanes.py")
    st.stop()
if not (YTEST_PATH.exists() and PTEST_PATH.exists()):
    st.error("Faltan caches de prueba (y_test/proba_test). Vuelve a entrenar:  python src/train_nhanes.py")
    st.stop()

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

bundle = load_model(MODEL_PATH)
pipe = bundle["pipeline"]
feature_order = bundle["features"]

st.info(f"Modelo: **{opcion_modelo}** · Variables: {', '.join(feature_order)}")

# -----------------------
# Estado y botones de reset
# -----------------------
DEFAULTS = {
    "k_edad": 45, "k_sexo": "Masculino", "k_imc": 27.5, "k_glucosa": 100.0,
    "k_pas": 120, "k_pad": 80, "k_cintura": 90.0, "k_actividad": "Baja",
    "k_tabaquismo": "No", "k_fam": "No", "k_umbral": 0.50, "k_yreal": 0
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# buffer demo (y_true, proba)
st.session_state.setdefault("demo_y",  np.array([], dtype=int))
st.session_state.setdefault("demo_p",  np.array([], dtype=float))

# >>> NUEVO: guardar última predicción para que no desaparezca
st.session_state.setdefault("last_pred", None)

def _reset_form():
    st.session_state.update(DEFAULTS)
    st.session_state["last_pred"] = None   # limpiar también el resultado persistente
    st.rerun()

def _clear_demo():
    st.session_state["demo_y"] = np.array([], dtype=int)
    st.session_state["demo_p"] = np.array([], dtype=float)
    st.rerun()

# Umbral + botones
cols_head = st.columns([4,1,1])
with cols_head[0]:
    umbral = st.slider("Umbral de clasificación (opcional)", 0.10, 0.90, key="k_umbral", step=0.01)
with cols_head[1]:
    st.button("Limpiar formulario", on_click=_reset_form)
with cols_head[2]:
    st.button("Limpiar cohorte demo", on_click=_clear_demo)

# -----------------------
# KPIs dinámicos (TEST + DEMO)
# -----------------------
y_test = np.load(YTEST_PATH)
p_test = np.load(PTEST_PATH)

def metrics_from_probs(y_true, p, thr):
    y_pred = (p >= thr).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    try:
        auc = roc_auc_score(y_true, p)
    except ValueError:
        auc = float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return auc, acc, prec, rec, spec, f1, (tn, fp, fn, tp)

# Combinar prueba fija + casos demo (si existen)
y_all = y_test
p_all = p_test
if len(st.session_state["demo_y"]) > 0:
    y_all = np.concatenate([y_test, st.session_state["demo_y"]])
    p_all = np.concatenate([p_test, st.session_state["demo_p"]])

auc, acc, prec, rec, spec, f1, (tn, fp, fn, tp) = metrics_from_probs(y_all, p_all, umbral)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("AUC", f"{auc:0.3f}")
k2.metric("Exactitud", f"{acc:0.3f}")
k3.metric("Precisión", f"{prec:0.3f}")
k4.metric("Sensibilidad", f"{rec:0.3f}")
k5.metric("Especificidad", f"{spec:0.3f}")
k6.metric("F1", f"{f1:0.3f}")
st.caption(f"Matriz de confusión (dinámica): TN {tn} · FP {fp} · FN {fn} · TP {tp}")

st.markdown("---")

# >>> NUEVO: si hay una predicción guardada, mostrarla SIEMPRE (antes del formulario)
if st.session_state["last_pred"] is not None:
    lp = st.session_state["last_pred"]
    st.subheader("Datos ingresados")
    st.dataframe(pd.DataFrame([lp["entrada"]]), width='stretch')
    # franja verde
    st.success(f"Probabilidad estimada de Diabetes T2: **{lp['proba']:.2%}**")
    # tarjeta de resultado
    bg = "#ffe5e5" if lp["clase"] == "Positivo" else "#e7f7ec"
    border = "#ff6b6b" if lp["clase"] == "Positivo" else "#2ecc71"
    icon = "🩸" if lp["clase"] == "Positivo" else "✅"
    msg  = (f"{icon} <strong>Resultado:</strong> {lp['clase']} — "
            f"{'Riesgo de diabetes tipo 2 detectado' if lp['clase']=='Positivo' else 'Sin indicios de diabetes tipo 2'} "
            f"(umbral {lp['umbral']:.2f}). "
            f"<br><em>Clasificación de riesgo general:</em> <strong>{lp['riesgo']}</strong>.")
    st.markdown(
        f"""
        <div style="background:{bg};border:2px solid {border};padding:14px 16px;
                    border-radius:14px;margin:8px 0 4px 0;font-size:1.05rem;">
            {msg}
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------
# Formulario
# -----------------------
with st.form("form-d2"):
    c1, c2 = st.columns(2)
    with c1:
        edad = st.number_input("Edad (años)", 1, 120, key="k_edad")
        sexo = st.selectbox("Sexo", ["Masculino","Femenino"], key="k_sexo")
        imc  = st.number_input("IMC (kg/m²)", 10.0, 70.0, key="k_imc")
        glu  = st.number_input("Glucosa en ayunas (mg/dL)", 40.0, 400.0, key="k_glucosa")
        pas  = st.number_input("Presión sistólica (mmHg)", 60, 250, key="k_pas")
    with c2:
        pad  = st.number_input("Presión diastólica (mmHg)", 40, 140, key="k_pad")
        cin  = st.number_input("Circunferencia de cintura (cm)", 40.0, 200.0, key="k_cintura")
        act  = st.selectbox("Actividad física", ["Baja","Moderada","Alta"], key="k_actividad")
        tab  = st.selectbox("Tabaquismo", ["No","Exfumador","Actual"], key="k_tabaquismo")
        fam  = st.selectbox("Historial familiar de DM2", ["No","Sí"], key="k_fam")

    st.subheader("Vista demo: etiqueta real para métricas")
    yreal = st.selectbox("Etiqueta real (demo)", ["Negativo (0)", "Positivo (1)"],
                         index=st.session_state.get("k_yreal",0))
    st.session_state["k_yreal"] = 0 if yreal.startswith("Negativo") else 1

    colp1, colp2 = st.columns([1,1])
    predecir = colp1.form_submit_button("Predecir")
    limpiar_local = colp2.form_submit_button("Limpiar campos (solo arriba)")

if limpiar_local:
    for key, val in DEFAULTS.items():
        if key.startswith("k_") and key not in ("k_umbral",):
            st.session_state[key] = val
    # NO tocamos last_pred aquí (para que persista si quieres)
    st.rerun()

# -----------------------
# Predicción y agregado a cohorte (y guardar para persistir)
# -----------------------
if predecir:
    entrada = pd.DataFrame([{
        "edad": st.session_state["k_edad"],
        "sexo": st.session_state["k_sexo"],
        "imc": st.session_state["k_imc"],
        "glucosa_ayunas": st.session_state["k_glucosa"],
        "presion_sistolica": st.session_state["k_pas"],
        "presion_diastolica": st.session_state["k_pad"],
        "cintura_cm": st.session_state["k_cintura"],
        "actividad_fisica": st.session_state["k_actividad"],
        "tabaquismo": st.session_state["k_tabaquismo"],
        "antecedente_familiar": st.session_state["k_fam"],
    }])

    entrada_modelo = entrada.reindex(columns=feature_order)
    proba = float(pipe.predict_proba(entrada_modelo)[:, 1][0])

    clase = "Positivo" if proba >= st.session_state["k_umbral"] else "Negativo"
    riesgo = "Bajo" if proba < 0.33 else ("Medio" if proba < 0.66 else "Alto")

    # Guardar para que persista tras el rerun
    st.session_state["last_pred"] = {
        "entrada": entrada.to_dict(orient="records")[0],
        "proba": proba,
        "umbral": st.session_state["k_umbral"],
        "clase": clase,
        "riesgo": riesgo,
    }

    # Agregar a cohorte demo para mover KPIs
    st.session_state["demo_y"] = np.append(st.session_state["demo_y"], st.session_state["k_yreal"])
    st.session_state["demo_p"] = np.append(st.session_state["demo_p"], proba)

    st.rerun()

# -----------------------
# Métricas fijas del entrenamiento
# -----------------------
st.markdown("---")
st.subheader(f"Métricas del modelo (conjunto de prueba fijo): {opcion_modelo}")
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        st.text(f.read())
else:
    st.caption("No se encontraron métricas.")
