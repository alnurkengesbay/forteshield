import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional
from src import config

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and merging of transaction and behavioral data.
    """

    def __init__(self, data_dir: Path = config.DATA_DIR):
        self.data_dir = data_dir

    def _load_csv(self, filename: str, sep: str = ';') -> pd.DataFrame:
        """
        Helper to load a CSV file with error handling.
        """
        file_path = self.data_dir / filename
        # Fallback to check if file exists in current directory or Downloads if not in data_dir
        if not file_path.exists():
            # Try looking in common locations for the sake of the environment
            potential_paths = [
                Path(filename),
                Path.home() / "Downloads" / filename,
                Path(r"c:\Users\alnur\Downloads") / filename
            ]
            for p in potential_paths:
                if p.exists():
                    file_path = p
                    break
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Could not find file: {filename}")

        try:
            logger.info(f"Loading data from {file_path}...")
            # Try default encoding first (utf-8)
            try:
                df = pd.read_csv(file_path, sep=sep)
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for {filename}, trying cp1251...")
                df = pd.read_csv(file_path, sep=sep, encoding='cp1251')
            
            logger.info(f"Successfully loaded {len(df)} rows from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise

    def load_and_merge(self) -> pd.DataFrame:
        """
        Loads transactions and patterns, converts dates, and merges them.
        """
        df_trans = self._load_csv(config.TRANSACTIONS_FILENAME)
        df_patterns = self._load_csv(config.PATTERNS_FILENAME)

        # Rename columns
        logger.info("Renaming columns from Russian to English...")
        df_trans.rename(columns=config.TRANS_COL_MAP, inplace=True)
        df_patterns.rename(columns=config.PATTERNS_COL_MAP, inplace=True)

        # Clean up: Remove rows where target is 'target' (duplicate headers)
        if config.TARGET_COL in df_trans.columns:
             df_trans = df_trans[df_trans[config.TARGET_COL] != config.TARGET_COL]
             # Convert target to numeric immediately
             df_trans[config.TARGET_COL] = pd.to_numeric(df_trans[config.TARGET_COL], errors='coerce')
             df_trans.dropna(subset=[config.TARGET_COL], inplace=True)
             df_trans[config.TARGET_COL] = df_trans[config.TARGET_COL].astype(int)

        # Date Conversion
        logger.info("Converting date columns...")
        # Ensure datetime format
        df_trans[config.DATETIME_COL] = pd.to_datetime(df_trans[config.DATETIME_COL], errors='coerce')
        df_trans[config.DATE_COL] = pd.to_datetime(df_trans[config.DATE_COL], errors='coerce')
        df_patterns[config.DATE_COL] = pd.to_datetime(df_patterns[config.DATE_COL], errors='coerce')

        # Merge
        logger.info("Merging datasets on ID and Date (Left Join)...")
        try:
            df_merged = df_trans.merge(
                df_patterns, 
                on=[config.ID_COL, config.DATE_COL], 
                how='left'
            )
            logger.info(f"Merged dataset shape: {df_merged.shape}")
            return df_merged
        except KeyError as e:
            logger.error(f"Merge failed. Check column names in CSVs. Missing key: {e}")
            raise
