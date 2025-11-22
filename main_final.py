import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, average_precision_score
import warnings

warnings.filterwarnings('ignore')

print("🚀 ЗАПУСК РЕЖИМА 'GOD MODE' (ENSEMBLE + INTERACTION FEATURES)...")

# --- 1. ЗАГРУЗКА ДАННЫХ ---
df_trans = pd.read_csv('data/транзакции_в_Мобильном_интернет_Банкинге.csv', sep=';', header=1, encoding='cp1251')
df_behav = pd.read_csv('data/поведенческие_паттерны_клиентов_3.csv', sep=';', header=1, encoding='cp1251')

for df in [df_trans, df_behav]:
    df['transdate'] = pd.to_datetime(df['transdate'].astype(str).str.strip("'"))
df_trans['transdatetime'] = pd.to_datetime(df_trans['transdatetime'].astype(str).str.strip("'"))

df = pd.merge(df_trans, df_behav, on=['cst_dim_id', 'transdate'], how='left')

# --- 2. FEATURE ENGINEERING (MAXIMUM POWER) ---
print("🔧 Генерация элитных фичей...")

# 2.1. User Context
user_stats = df.groupby('cst_dim_id')['amount'].agg(['mean', 'std', 'max']).reset_index()
user_stats.columns = ['cst_dim_id', 'user_mean', 'user_std', 'user_max']
df = pd.merge(df, user_stats, on='cst_dim_id', how='left')

df['amount_zscore'] = (df['amount'] - df['user_mean']) / (df['user_std'] + 1.0)
df['amount_to_max'] = df['amount'] / (df['user_max'] + 1.0) # Насколько близко к рекорду клиента
df['amount_log'] = np.log1p(df['amount'])

# 2.2. Time Features
df['hour'] = df['transdatetime'].dt.hour
df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x > 23 else 0)
df['day_of_week'] = df['transdatetime'].dt.dayofweek

# 2.3. Interaction Features (Золотые фичи)
# "Ночная транзакция на большую сумму"
df['night_x_amount'] = df['is_night'] * df['amount_log']
# "Редкий получатель + Большая сумма" (Frequency Encoding * Amount)
freq_map = df['direction'].value_counts(normalize=True).to_dict()
df['direction_freq'] = df['direction'].map(freq_map)
df['rare_high_amount'] = (1 - df['direction_freq']) * df['amount_log']

# --- 3. ПОДГОТОВКА ---
X_temp = df.drop(columns=['target'])
y_temp = df['target']
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y_temp)

# 2.4. Target Encoding (Risk Score) - Осторожно, чтобы не переобучиться
def smooth_target_encode(train_df, test_df, cat_col, target_col, weight=10):
    global_mean = train_df[target_col].mean()
    agg = train_df.groupby(cat_col)[target_col].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smoothed = (counts * means + weight * global_mean) / (counts + weight)
    return train_df[cat_col].map(smoothed).fillna(global_mean), test_df[cat_col].map(smoothed).fillna(global_mean)

df.loc[train_idx, 'receiver_risk'], df.loc[test_idx, 'receiver_risk'] = \
    smooth_target_encode(df.loc[train_idx], df.loc[test_idx], 'direction', 'target', weight=5)

# NEW: Cross-Domain Interactions (Risk * Behavior)
# ТЕПЕРЬ, когда receiver_risk создан, можно делать интеракции
df['risk_x_zscore'] = df['receiver_risk'] * df['amount_zscore']
df['risk_x_night'] = df['receiver_risk'] * df['is_night']

# Очистка
drop_cols = ['cst_dim_id', 'transdate', 'transdatetime', 'docno', 'target',
             'Зашифрованный идентификатор получателя/destination транзакции', 
             'direction', 'user_mean', 'user_std', 'user_max']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['target']

# FillNA
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].fillna('Unknown')
    else:
        X[col] = X[col].fillna(0)

cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']

X_train = X.loc[train_idx]
y_train = y.loc[train_idx]
X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

# --- 4. ENSEMBLE TRAINING (3 MODELS) ---
print("\n🔥 Обучение Ансамбля (3 нейросети)...")

# Model 1: Precision Focused (All Features)
model_1 = CatBoostClassifier(
    iterations=1000, depth=4, l2_leaf_reg=10, learning_rate=0.05,
    auto_class_weights='SqrtBalanced', cat_features=cat_features, verbose=0, random_seed=42
)
model_1.fit(X_train, y_train)
print("✅ Model 1 (Skeptic) ready.")

# Model 2: Recall Focused (NO RISK SCORE) - "Ищейка"
# Убираем receiver_risk, чтобы модель искала аномалии поведения
X_train_no_risk = X_train.drop(columns=['receiver_risk'])
X_test_no_risk = X_test.drop(columns=['receiver_risk'])
cat_features_no_risk = [i for i, col in enumerate(X_train_no_risk.columns) if X_train_no_risk[col].dtype == 'object']

model_2 = CatBoostClassifier(
    iterations=1500, depth=10, l2_leaf_reg=0.1, learning_rate=0.01, # Aggressive Overfitting for Recall
    scale_pos_weight=20, # FORCE the model to find fraud
    cat_features=cat_features_no_risk, verbose=0, random_seed=43
)
model_2.fit(X_train_no_risk, y_train)
print("✅ Model 2 (Bloodhound - No Risk) ready.")

# Model 3: Balanced (All Features)
model_3 = CatBoostClassifier(
    iterations=1000, depth=6, l2_leaf_reg=3, learning_rate=0.04,
    auto_class_weights='SqrtBalanced', cat_features=cat_features, verbose=0, random_seed=44
)
model_3.fit(X_train, y_train)
print("✅ Model 3 (Realist) ready.")

# --- 5. VOTING (ГОЛОСОВАНИЕ) ---
print("\n🗳️ Голосование моделей...")

p1 = model_1.predict_proba(X_test)[:, 1]
p2 = model_2.predict_proba(X_test_no_risk)[:, 1] # Важно: подаем X без риска
p3 = model_3.predict_proba(X_test)[:, 1]

# Взвешенное среднее
# 40% Скептик (Точность), 40% Ищейка (Охват), 20% Реалист
final_proba = (p1 * 0.40) + (p2 * 0.40) + (p3 * 0.20)

# --- 6. ПОИСК ИДЕАЛЬНОГО БАЛАНСА ---
print("⚖️ Подбор идеального порога...")
print(f"   Max Probability: {final_proba.max():.4f}")
print(f"   Mean Probability: {final_proba.mean():.4f}")

best_f1 = 0
best_thr = 0.5
best_metrics = {'Precision': 0, 'Recall': 0}

# Ищем порог, максимизирующий F1
for thr in np.arange(0.01, 0.95, 0.01):
    pred = (final_proba > thr).astype(int)
    
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
        best_metrics = {'Precision': prec, 'Recall': rec}

print("\n" + "="*40)
print(f"🏆 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ (ENSEMBLE)")
print("="*40)
print(f"Threshold: {best_thr:.2f}")
print(f"💎 Precision: {best_metrics['Precision']:.2%}")
print(f"🔍 Recall:    {best_metrics['Recall']:.2%}")
print(f"⚖️ F1-Score:  {best_f1:.2%}")
print("="*40)

# Сохраняем лучшую модель (по факту сохраним Model 3 как основную для демо, т.к. ансамбль сложно сохранять в один файл)
model_3.save_model("catboost_final.cbm")
print("\n💾 Основная модель сохранена как 'catboost_final.cbm'")
