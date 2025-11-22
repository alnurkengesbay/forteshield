import pandas as pd
import numpy as np
import logging
from typing import List
from src import config

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Encapsulates all feature engineering logic.
    """

    def __init__(self):
        pass

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts hour, day of week, and is_night flag.
        """
        logger.info("Generating time-based features...")
        if config.DATETIME_COL not in df.columns:
            logger.warning(f"{config.DATETIME_COL} missing. Skipping time features.")
            return df

        df = df.copy()
        df['hour'] = df[config.DATETIME_COL].dt.hour
        df['day_of_week'] = df[config.DATETIME_COL].dt.dayofweek
        
        # Night transaction: 00:00 to 06:00
        df['is_night'] = df['hour'].apply(lambda x: 1 if 0 <= x < 6 else 0)
        
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills NaNs: 'Unknown' for categorical, 0 for numerical.
        """
        logger.info("Handling missing values...")
        df = df.copy()

        # Identify categorical columns present in the dataframe
        cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in df.columns]
        
        # Fill categorical
        for col in cat_cols:
            df[col] = df[col].fillna('Unknown').astype(str)

        # Identify numerical columns
        # We explicitly exclude categorical columns, ID, target, and date columns
        exclude_cols = set(cat_cols) | {config.ID_COL, config.TARGET_COL, config.DATE_COL, config.DATETIME_COL}
        num_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Force numeric conversion
        for col in num_cols:
            # Coerce errors to NaN (e.g. "04.фев" -> NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill numerical
        df[num_cols] = df[num_cols].fillna(0)
        
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates advanced interaction features and encodings.
        """
        logger.info("Generating interaction features...")
        df = df.copy()
        
        # 1. Log Amount (Smoothing)
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            
        # 2. Direction Frequency (Count Encoding)
        # Note: In a real production pipeline, these counts should be saved from training set
        # and applied to inference. For this setup, we calculate on the loaded batch.
        if 'direction' in df.columns:
            direction_counts = df['direction'].value_counts()
            df['direction_freq'] = df['direction'].map(direction_counts)
            # Fill NaNs for new directions seen in inference (if any) with 1
            df['direction_freq'] = df['direction_freq'].fillna(1)

        # 3. New Device Flag
        if 'monthly_phone_model_changes' in df.columns:
            df['new_device_flag'] = (df['monthly_phone_model_changes'] > 0).astype(int)
            
        # 4. Night High Amount Interaction
        if 'is_night' in df.columns and 'amount' in df.columns:
            df['night_high_amount'] = df['is_night'] * df['amount']

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the full preprocessing pipeline.
        """
        df = self.create_time_features(df)
        # Clean data (convert to numeric) BEFORE creating interaction features
        df = self.handle_missing_values(df)
        df = self.create_interaction_features(df)
        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Returns the list of feature names used for training.
        """
        return [c for c in df.columns if c not in config.DROP_COLS]
