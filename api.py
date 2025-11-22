from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import joblib
import json

app = FastAPI()

# --- Модель ---
MODEL_PATH = "catboost_pro.cbm"
CATBOOST_MODEL = None
CAT_FEATURES = [
    'last_phone_model_categorical', 'last_os_categorical', 'login_frequency_30d',
    'freq_change_7d_vs_mean', 'var_login_interval_30d', 'hour', 'is_night', 'iso_anomaly', 'cluster'
]

# --- Схема входных данных ---
class Transaction(BaseModel):
    amount: float
    monthly_os_changes: int
    monthly_phone_model_changes: int
    last_phone_model_categorical: str
    last_os_categorical: str
    logins_last_7_days: int
    logins_last_30_days: int
    login_frequency_7d: float
    login_frequency_30d: float
    freq_change_7d_vs_mean: float
    logins_7d_over_30d_ratio: float
    avg_login_interval_30d: float
    std_login_interval_30d: float
    var_login_interval_30d: float
    ewm_login_interval_7d: float
    burstiness_login_interval: float
    fano_factor_login_interval: float
    zscore_avg_login_interval_7d: float
    hour: int
    is_night: int
    direction_freq: float
    iso_anomaly: int
    cluster: int
    dist_to_center: float

# --- Загрузка модели ---
def load_model():
    global CATBOOST_MODEL
    from catboost import CatBoostClassifier
    CATBOOST_MODEL = CatBoostClassifier()
    CATBOOST_MODEL.load_model(MODEL_PATH)

@app.on_event("startup")
def startup_event():
    load_model()

@app.post("/predict")
async def predict(transaction: Transaction):
    data = transaction.dict()
    df = pd.DataFrame([data])
    # Привести категориальные к строкам
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)
    # Предсказание
    proba = CATBOOST_MODEL.predict_proba(df)[0][1]
    verdict = "fraud" if proba > 0.3 else "clean"
    return {"verdict": verdict, "probability": proba}

# --- Эндпоинт для обратной связи (дообучение) ---
class Feedback(BaseModel):
    transaction: dict
    label: int  # 1 - fraud, 0 - clean

@app.post("/feedback")
async def feedback(feedback: Feedback):
    # Сохраняем обратную связь для дообучения
    with open("feedback_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"transaction": feedback.transaction, "label": feedback.label}, ensure_ascii=False) + "\n")
    return {"status": "feedback saved"}
