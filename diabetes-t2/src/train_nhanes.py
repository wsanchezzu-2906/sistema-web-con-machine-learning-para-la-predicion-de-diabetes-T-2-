# src/train_nhanes.py
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# -----------------------------
# Rutas y salida
# -----------------------------
DATA_5 = Path("data/nhanes-5vari.csv")
XPT_BMX = Path("data/BMX_J.xpt")
XPT_PAQ = Path("data/PAQ_J.xpt")
XPT_SMQ = Path("data/SMQ_J.xpt")
XPT_DIQ = Path("data/diabetes.DIQ_J.xpt")

OUT = Path("models")
OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Cargar base CSV (5 variables)
# -----------------------------
if not DATA_5.exists():
    raise FileNotFoundError(f"No se encontró {DATA_5.resolve()}")

df = pd.read_csv(DATA_5)

expected = ["Source", "Gender", "Age", "BMI", "Glucose", "BloodPressure", "Diabetes"]
if list(df.columns) != expected:
    if len(df.columns) == 1:
        df = df.iloc[:, 0].str.split(",", expand=True)
        df.columns = expected
    else:
        raise ValueError(f"Encabezados inesperados. Esperado: {expected}, obtenido: {list(df.columns)}")

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
df["Glucose"] = pd.to_numeric(df["Glucose"], errors="coerce")
df["BloodPressure"] = pd.to_numeric(df["BloodPressure"], errors="coerce")
df["Diabetes"] = pd.to_numeric(df["Diabetes"], errors="coerce").astype("Int64")

df["sexo"] = df["Gender"].map({"Male": "Masculino", "Female": "Femenino"}).fillna("Masculino")

df_base = pd.DataFrame({
    "edad": df["Age"],
    "sexo": df["sexo"],
    "imc": df["BMI"],
    "glucosa_ayunas": df["Glucose"],
    "presion_diastolica": df["BloodPressure"],
})
df_base["presion_sistolica"] = df_base["presion_diastolica"] + 40  # aproximación

y = df["Diabetes"].rename("diabetes").astype(int)

# -----------------------------
# 2) Leer XPT (si existen) para distribuciones
# -----------------------------
def safe_read_xpt(path):
    try:
        if path.exists():
            return pd.read_sas(path, format="xport")
    except Exception:
        pass
    return None

bmx = safe_read_xpt(XPT_BMX)
paq = safe_read_xpt(XPT_PAQ)
smq = safe_read_xpt(XPT_SMQ)
diq = safe_read_xpt(XPT_DIQ)

if bmx is not None and "BMXWAIST" in bmx.columns:
    cintura_series = pd.to_numeric(bmx["BMXWAIST"], errors="coerce").dropna()
    cintura_mu = float(cintura_series.mean())
    cintura_sd = float(cintura_series.std(ddof=1)) if len(cintura_series) > 1 else 12.0
else:
    cintura_mu, cintura_sd = 95.0, 12.0

def actividad_from_paq(df_paq):
    if df_paq is None:
        return {"Alta":0.25, "Moderada":0.35, "Baja":0.40}
    vig = None
    for c in ["PAQ650","PAQ605"]:
        if c in df_paq.columns: vig = df_paq[c]; break
    mod = None
    for c in ["PAQ665","PAQ620"]:
        if c in df_paq.columns: mod = df_paq[c]; break
    if vig is None and mod is None:
        return {"Alta":0.25, "Moderada":0.35, "Baja":0.40}
    def to_cat(v, m):
        a_yes = (v == 1) if v is not None else False
        m_yes = (m == 1) if m is not None else False
        if a_yes: return "Alta"
        elif m_yes: return "Moderada"
        return "Baja"
    cats = [to_cat(v, m) for v, m in zip(
        vig.to_numpy() if vig is not None else [np.nan]*len(df_paq),
        mod.to_numpy() if mod is not None else [np.nan]*len(df_paq)
    )]
    vc = pd.Series(cats).value_counts(normalize=True)
    p = {"Alta": vc.get("Alta", 0.25), "Moderada": vc.get("Moderada", 0.35), "Baja": vc.get("Baja", 0.40)}
    s = sum(p.values())
    return {k: v/s for k, v in p.items()}

def tabaquismo_from_smq(df_smq):
    if df_smq is None:
        return {"No":0.65, "Exfumador":0.15, "Actual":0.20}
    smq020 = df_smq["SMQ020"] if "SMQ020" in df_smq.columns else None
    smq040 = df_smq["SMQ040"] if "SMQ040" in df_smq.columns else None
    if smq020 is None and smq040 is None:
        return {"No":0.65, "Exfumador":0.15, "Actual":0.20}
    def to_cat(ever, now):
        if now in [1,2]: return "Actual"
        elif ever == 1 and now == 3: return "Exfumador"
        return "No"
    cats = [to_cat(e, n) for e, n in zip(
        smq020.to_numpy() if smq020 is not None else [np.nan]*len(df_smq),
        smq040.to_numpy() if smq040 is not None else [np.nan]*len(df_smq)
    )]
    vc = pd.Series(cats).value_counts(normalize=True)
    p = {"No": vc.get("No", 0.65), "Exfumador": vc.get("Exfumador", 0.15), "Actual": vc.get("Actual", 0.20)}
    s = sum(p.values())
    return {k: v/s for k, v in p.items()}

def fam_from_diq(df_diq):
    if df_diq is None or "DIQ170" not in df_diq.columns:
        return {"Sí":0.20, "No":0.80}
    fam = pd.to_numeric(df_diq["DIQ170"], errors="coerce")
    p_si = float((fam == 1).mean()); p_no = 1.0 - p_si
    if p_si == 0 and p_no == 0: p_si, p_no = 0.20, 0.80
    return {"Sí":p_si, "No":p_no}

p_act = actividad_from_paq(paq)
p_tab = tabaquismo_from_smq(smq)
p_fam = fam_from_diq(diq)

# -----------------------------
# 3) Completar variables faltantes
# -----------------------------
n = len(df_base)
rng = np.random.default_rng(42)

base_cint = rng.normal(loc=cintura_mu, scale=cintura_sd, size=n)
df_base["cintura_cm"] = np.clip(base_cint + 0.8*(df_base["imc"] - 27), 60, 150)

df_base["actividad_fisica"] = rng.choice(["Alta","Moderada","Baja"], size=n,
                                         p=[p_act["Alta"], p_act["Moderada"], p_act["Baja"]])
df_base["tabaquismo"] = rng.choice(["No","Exfumador","Actual"], size=n,
                                   p=[p_tab["No"], p_tab["Exfumador"], p_tab["Actual"]])
df_base["antecedente_familiar"] = rng.choice(["No","Sí"], size=n,
                                             p=[p_fam["No"], p_fam["Sí"]])

feature_order = [
    "edad","sexo","imc","glucosa_ayunas",
    "presion_sistolica","presion_diastolica",
    "cintura_cm","actividad_fisica",
    "tabaquismo","antecedente_familiar"
]
X = df_base[feature_order].copy()

# -----------------------------
# 4) Imputaciones mínimas
# -----------------------------
num_cols = ["edad","imc","glucosa_ayunas","presion_sistolica","presion_diastolica","cintura_cm"]
cat_cols = ["sexo","actividad_fisica","tabaquismo","antecedente_familiar"]

for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
for c in cat_cols:
    X[c] = X[c].astype(str).fillna("Desconocido")

# -----------------------------
# 5) Split, preprocesamiento y entrenamiento
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

def entrenar_evaluar_guardar(modelo, nombre):
    pipe = Pipeline([("pre", pre), ("clf", modelo)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn/(tn+fp)
    rep = classification_report(y_test, y_pred, digits=3)

    model_path   = OUT / f"model_{nombre}.joblib"
    metrics_path = OUT / f"metrics_model_{nombre}.txt"

    joblib.dump({"pipeline": pipe, "features": feature_order}, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"AUC: {auc:.3f}\nAccuracy: {acc:.3f}\nPrecision: {prec:.3f}\n")
        f.write(f"Sensibilidad: {rec:.3f}\nEspecificidad: {spec:.3f}\n")
        f.write(f"Matriz de confusión: [[{tn} {fp}] [{fn} {tp}]]\n\n")
        f.write(rep)

    # NEW: guarda y_test y proba_test para métricas dinámicas en la app
    np.save(OUT / f"cache_{nombre}_ytest.npy",  y_test.to_numpy())
    np.save(OUT / f"cache_{nombre}_ptest.npy", proba)

    print(f"[{nombre}] modelo:   {model_path.resolve()}")
    print(f"[{nombre}] métricas: {metrics_path.resolve()}")

# Modelos (balanceados para mejor recall)
rf     = RandomForestClassifier(n_estimators=400, max_depth=10,
                               min_samples_leaf=2, class_weight="balanced",
                               random_state=42)
logreg = LogisticRegression(max_iter=3000, solver="liblinear",
                            class_weight="balanced", random_state=42)

entrenar_evaluar_guardar(rf,     "nhanes_rf")
entrenar_evaluar_guardar(logreg, "nhanes_logreg")

print("\nEntrenamiento NHANES (CSV+XPT) terminado.")
