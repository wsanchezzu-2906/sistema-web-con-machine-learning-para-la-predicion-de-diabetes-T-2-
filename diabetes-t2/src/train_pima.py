# src/train_pima.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# Rutas
DATA = Path("data/pima.csv")
OUT_DIR = Path("models"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Carga
if not DATA.exists():
    raise FileNotFoundError(f"No se encontró {DATA.resolve()}.\nPon pima.csv dentro de la carpeta 'data/'.")

df = pd.read_csv(DATA)

# 2) Limpieza: en Pima hay '0' que significan 'desconocido'
for col in ["Glucose", "BloodPressure", "BMI"]:
    df[col] = df[col].replace(0, np.nan).fillna(df[col].median())

# 3) Renombrar y seleccionar features
df = df.rename(columns={
    "Age": "edad",
    "BMI": "imc",
    "Glucose": "glucosa_ayunas",
    "BloodPressure": "presion_diastolica",
    "Outcome": "diabetes"
})
features = ["edad", "imc", "glucosa_ayunas", "presion_diastolica"]
target = "diabetes"

X = df[features].copy()
y = df[target].astype(int).copy()

# 4) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

def evaluar_y_guardar(modelo, nombre_archivo_base: str):
    """Entrena, evalúa, guarda modelo (.joblib) y métricas (.txt)."""
    modelo.fit(X_train, y_train)

    proba_test = modelo.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)  # sensibilidad
    auc = roc_auc_score(y_test, proba_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)           # especificidad
    report = classification_report(y_test, y_pred, digits=3)

    # Guardar modelo y métricas
    out_model = OUT_DIR / f"{nombre_archivo_base}.joblib"
    out_metrics = OUT_DIR / f"metrics_{nombre_archivo_base}.txt"
    joblib.dump({"pipeline": modelo, "features": features}, out_model)

    with open(out_metrics, "w", encoding="utf-8") as f:
        f.write(f"AUC: {auc:.3f}\nAccuracy: {acc:.3f}\nPrecision: {prec:.3f}\n")
        f.write(f"Sensibilidad: {recall:.3f}\nEspecificidad: {specificity:.3f}\n")
        f.write(f"Matriz de confusión: [[{tn} {fp}] [{fn} {tp}]]\n\n")
        f.write(report)

    print(f"\n[{nombre_archivo_base}] guardado:")
    print(f"  Modelo:   {out_model.resolve()}")
    print(f"  Métricas: {out_metrics.resolve()}")

# 5) Modelo 1: Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42
)
evaluar_y_guardar(rf, "model_rf")

# 6) Modelo 2: Regresión Logística (prob. lineales, interpretable)
logreg = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
evaluar_y_guardar(logreg, "model_logreg")

print("\n Entrenamiento finalizado de ambos modelos.")
