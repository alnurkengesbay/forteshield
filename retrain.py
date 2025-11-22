import pandas as pd
from catboost import CatBoostClassifier
import os

DATA_PATH = "corrected_data.csv"
MODEL_PATH = "catboost_pro.cbm"

# --- Симуляция дообучения ---
def retrain_model():
    if not os.path.exists(DATA_PATH):
        print("Нет новых данных для дообучения.")
        return
    df = pd.read_csv(DATA_PATH)
    if 'label' not in df.columns:
        print("Нет метки 'label' в данных.")
        return
    X = df.drop(columns=['label'])
    y = df['label']
    model = CatBoostClassifier(iterations=100, auto_class_weights='Balanced', verbose=0)
    model.fit(X, y)
    model.save_model(MODEL_PATH)
    print("Модель дообучена и сохранена.")

if __name__ == "__main__":
    retrain_model()
