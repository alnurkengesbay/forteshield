import pandas as pd
import numpy as np
import logging
import shap
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report

from src import config

logger = logging.getLogger(__name__)

class FraudModel:
    """
    Wrapper around CatBoostClassifier with SHAP explanation capabilities.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params if params else config.CATBOOST_PARAMS
        self.model = CatBoostClassifier(**self.params)
        self.cat_features: List[str] = []
        self.feature_names: List[str] = []

    def train(self, df: pd.DataFrame):
        """
        Trains the model on the provided DataFrame.
        """
        logger.info("Preparing data for training...")
        
        # Identify features
        self.feature_names = [c for c in df.columns if c not in config.DROP_COLS]
        X = df[self.feature_names]
        y = df[config.TARGET_COL]

        # Identify categorical features present in X
        self.cat_features = [c for c in config.CATEGORICAL_FEATURES if c in X.columns]
        logger.info(f"Categorical features: {self.cat_features}")

        # Check target distribution
        target_counts = y.value_counts()
        logger.info(f"Target distribution:\n{target_counts}")
        
        if len(target_counts) < 2:
             logger.warning("Target has less than 2 classes. Training might fail or be meaningless.")
        elif target_counts.min() < 2:
             logger.warning("One of the classes has fewer than 2 samples. Stratified split will fail.")
             # Fallback to non-stratified split if strictly necessary, or just warn.
             # For now, let's just log it.

        # Split
        logger.info(f"Splitting data (Test size: {config.TEST_SIZE})...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=config.TEST_SIZE, 
                random_state=config.RANDOM_STATE, 
                stratify=y
            )
        except ValueError as e:
            logger.error(f"Stratified split failed: {e}. Falling back to random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=config.TEST_SIZE, 
                random_state=config.RANDOM_STATE
            )

        logger.info(f"Training CatBoost model on {len(X_train)} samples...")
        self.model.fit(
            X_train, y_train,
            cat_features=self.cat_features,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            verbose=100
        )
        logger.info("Training complete.")

        # Evaluate immediately on test set
        self.evaluate(X_test, y_test)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Calculates and logs metrics.
        """
        logger.info("Evaluating model...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "ROC-AUC": roc_auc_score(y_test, y_pred_proba),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        }

        logger.info("=== Evaluation Results (Default Threshold 0.5) ===")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")
        
        logger.info("\n" + classification_report(y_test, y_pred))

        # Custom Threshold Logic for Imbalanced Data
        threshold = 0.1
        y_pred_custom = (y_pred_proba > threshold).astype(int)
        
        logger.info(f"--- Metrics with Threshold {threshold} ---")
        logger.info(f"Recall: {recall_score(y_test, y_pred_custom):.4f}")
        logger.info(f"Precision: {precision_score(y_test, y_pred_custom):.4f}")
        logger.info(f"F1-Score: {f1_score(y_test, y_pred_custom):.4f}")
        logger.info("\n" + classification_report(y_test, y_pred_custom))

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts fraud probability.
        """
        if not self.feature_names:
             # If feature names are missing, try to infer them or use all valid columns
             self.feature_names = [c for c in df.columns if c not in config.DROP_COLS]
             
        # Ensure all features exist in df, fill missing with 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
                
        X = df[self.feature_names]
        return self.model.predict_proba(X)[:, 1]

    def explain_prediction(self, df: pd.DataFrame, index: int = 0) -> None:
        """
        Generates a SHAP explanation for a specific row in the dataframe.
        """
        logger.info("Generating SHAP explanation...")
        X = df[self.feature_names]
        
        # Create Explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        # Get specific instance
        instance_values = X.iloc[index]
        instance_shap = shap_values[index]
        
        # Text explanation
        explanation_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': instance_values.values,
            'shap_impact': instance_shap
        })
        
        # Filter for positive contributors (pushing towards Fraud)
        risk_factors = explanation_df[explanation_df['shap_impact'] > 0].sort_values(by='shap_impact', ascending=False).head(3)
        
        print("\n--- FRAUD EXPLANATION REPORT ---")
        print(f"Transaction Index: {index}")
        print("Top Risk Factors:")
        for i, row in risk_factors.iterrows():
            print(f"- Feature '{row['feature']}' = '{row['value']}' (+{row['shap_impact']:.2f} risk)")
            
        # Note: In a CLI environment, we can't show plots easily, but we can save them
        # shap.plots.waterfall(...) 

    def save(self, filename: str = "model.cbm"):
        """
        Saves the model to disk.
        """
        path = config.MODELS_DIR / filename
        self.model.save_model(str(path))
        logger.info(f"Model saved to {path}")

    def load(self, filename: str = "model.cbm"):
        """
        Loads the model from disk.
        """
        path = config.MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model.load_model(str(path))
        
        # Restore feature names from the loaded model
        if hasattr(self.model, 'feature_names_'):
            self.feature_names = self.model.feature_names_
        else:
            # Fallback if feature_names_ is not available (older versions)
            logger.warning("Model feature names not found. Using all columns except dropped ones.")
            # This is risky if the dataframe has extra columns, but better than crashing
            pass
            
        logger.info(f"Model loaded from {path}")
