import os
from pathlib import Path
from typing import List, Dict, Any

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# File Paths
# Note: In a real env, these might be absolute paths or S3 buckets
TRANSACTIONS_FILENAME = "транзакции_в_Мобильном_интернет_Банкинге.csv"
PATTERNS_FILENAME = "поведенческие_паттерны_клиентов_3.csv"

# Column Definitions
ID_COL = "cst_dim_id"
DATE_COL = "transdate"
DATETIME_COL = "transdatetime"
TARGET_COL = "target"

# Column Renaming Map (Russian to English)
TRANS_COL_MAP = {
    'Уникальный идентификатор клиента': ID_COL,
    'Дата совершенной транзакции': DATE_COL,
    'Дата и время совершенной транзакции': DATETIME_COL,
    'Сумма совершенного перевода': 'amount',
    'Размеченные транзакции(переводы), где 1 - мошенническая операция , 0 - чистая': TARGET_COL,
    'Уникальный идентификатор транзакции': 'docno',
    'Зашифрованный идентификатор получателя/destination транзакции': 'direction'
}

PATTERNS_COL_MAP = {
    'Уникальный идентификатор клиента': ID_COL,
    'Дата совершенной транзакции': DATE_COL,
    'Модель телефона из самой последней сессии (по времени) перед transdate': 'last_phone_model',
    'Версия ОС из самой последней сессии перед transdate': 'last_os',
    'Количество разных версий ОС (os_ver) за последние 30 дней до transdate — сколько разных ОС/версий использовал клиент': 'monthly_os_changes',
    'Количество разных моделей телефона (phone_model) за последние 30 дней — насколько часто клиент “менял устройство” по логам': 'monthly_phone_model_changes'
}

# Feature Configuration
CATEGORICAL_FEATURES: List[str] = [
    "last_phone_model", 
    "last_os", 
    "direction", 
    "docno",
    "last_phone_model_categorical", # From patterns file if exists
    "last_os_categorical"           # From patterns file if exists
]

# Columns to drop during training
DROP_COLS: List[str] = [
    ID_COL, 
    DATE_COL, 
    DATETIME_COL, 
    TARGET_COL,
    "docno",      # ID, noise
    "direction"   # Avoid overfitting on specific wallets
]

# Model Hyperparameters
CATBOOST_PARAMS: Dict[str, Any] = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3,
    "eval_metric": "AUC",
    "scale_pos_weight": 10, # Manual balance for precision
    "verbose": 100,
    "allow_writing_files": False,
    "random_seed": 42,
    "early_stopping_rounds": 100,
    # Uncomment 'task_type': 'GPU' if you have a GPU configured
    # "task_type": "GPU" 
}

# Random Seed
RANDOM_STATE = 42
TEST_SIZE = 0.2
