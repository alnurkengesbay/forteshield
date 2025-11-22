import pandas as pd
import matplotlib.pyplot as plt
from src.model import FraudModel
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("Initializing model...")
    fraud_model = FraudModel()
    
    try:
        fraud_model.load()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Access the underlying CatBoost model
    model = fraud_model.model

    # Получаем важность признаков
    feature_importance = model.get_feature_importance(prettified=True)

    # Выводим топ-20 в консоль цифрами (для точности)
    print("\n--- TOP 20 FEATURES ---")
    print(feature_importance.head(20))

    # Строим график (для наглядности)
    plt.figure(figsize=(12, 8))
    
    # CatBoost 'prettified' output usually has columns "Feature Id" and "Importances"
    # We use iloc to be safe regardless of exact column name spacing
    features = feature_importance.iloc[:15, 0][::-1]
    importances = feature_importance.iloc[:15, 1][::-1]
    
    plt.barh(features, importances)
    plt.title('Что на самом деле важно для модели? (Feature Importance)')
    plt.xlabel('Важность (%)')
    
    # Save to file instead of show() since we are in a terminal environment
    output_path = 'feature_importance.png'
    plt.savefig(output_path)
    print(f"\nГрафик сохранен в файл: {output_path}")

if __name__ == "__main__":
    main()
