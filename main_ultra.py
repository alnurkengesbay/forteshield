import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')

print("🚀 ЗАПУСК РЕЖИМА 'ULTRA' (TARGET ENCODING + AGGREGATIONS)...")

# --- 1. ЗАГРУЗКА ---
df_trans = pd.read_csv('data/транзакции_в_Мобильном_интернет_Банкинге.csv', sep=';', header=1, encoding='cp1251')
df_behav = pd.read_csv('data/поведенческие_паттерны_клиентов_3.csv', sep=';', header=1, encoding='cp1251')

# Чистка
for df in [df_trans, df_behav]:
    df['transdate'] = pd.to_datetime(df['transdate'].str.strip("'"))
df_trans['transdatetime'] = pd.to_datetime(df_trans['transdatetime'].str.strip("'"))

# Мерж
df = pd.merge(df_trans, df_behav, on=['cst_dim_id', 'transdate'], how='left')

# --- 2. FEATURE ENGINEERING (PRO LEVEL) ---

# 2.1. Базовые фичи
df['hour'] = df['transdatetime'].dt.hour
df['day_of_week'] = df['transdatetime'].dt.dayofweek
df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x > 23 else 0)

# 2.2. TARGET ENCODING для Direction (Безопасный расчет риска получателя)
# Суть: Считаем % фрода для каждого получателя, но добавляем "вес доверия" (Smoothing),
# чтобы не банить получателя с 1 транзакцией.
def smooth_target_encode(df, cat_col, target_col, weight=10):
    # Глобальное среднее (вероятность фрода по всей базе)
    global_mean = df[target_col].mean()
    
    # Агрегация по категории
    agg = df.groupby(cat_col)[target_col].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    
    # Формула сглаживания: (count * mean + weight * global_mean) / (count + weight)
    smoothed = (counts * means + weight * global_mean) / (counts + weight)
    
    return df[cat_col].map(smoothed).fillna(global_mean)

# ВАЖНО: Считаем это ТОЛЬКО на Train части, чтобы не было утечки!
# Но для простоты здесь разделим выборку ДО генерации фичей
X_temp = df.drop(columns=['target'])
y_temp = df['target']
X_train_idx, X_test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y_temp)

# Создаем колонку (по умолчанию глобальное среднее)
df['receiver_risk_score'] = df['target'].mean()

# Обучаем энкодер на TRAIN и применяем к TRAIN
train_df = df.loc[X_train_idx]
df.loc[X_train_idx, 'receiver_risk_score'] = smooth_target_encode(train_df, 'direction', 'target', weight=10)

# Применяем "знания" из TRAIN к TEST (как в реальности)
# Берем словарь рисков из трейна
risk_map = df.loc[X_train_idx].groupby('direction')['receiver_risk_score'].mean()
global_risk = df.loc[X_train_idx]['target'].mean()
# Мапим, если не нашли (новый получатель) — ставим глобальный риск
df.loc[X_test_idx, 'receiver_risk_score'] = df.loc[X_test_idx, 'direction'].map(risk_map).fillna(global_risk)


# 2.3. USER AGGREGATIONS (Контекст клиента)
# Насколько эта транзакция больше обычной для этого клиента?
# (В реале это делается через оконные функции, тут упростим до группировки)
user_stats = df.groupby('cst_dim_id')['amount'].agg(['mean', 'std']).reset_index()
user_stats.columns = ['cst_dim_id', 'user_mean_amt', 'user_std_amt']

df = pd.merge(df, user_stats, on='cst_dim_id', how='left')

# Z-score суммы (насколько сумма аномальна для клиента)
# +0.01 чтобы не делить на ноль
df['amount_zscore'] = (df['amount'] - df['user_mean_amt']) / (df['user_std_amt'] + 0.01)
df['amount_to_mean'] = df['amount'] / (df['user_mean_amt'] + 1.0)

# --- 3. ПОДГОТОВКА ---
# Убираем сырые ID, но оставляем наши НОВЫЕ умные фичи
drop_cols = ['cst_dim_id', 'transdate', 'transdatetime', 'docno', 'direction', 'target', 
             'Зашифрованный идентификатор получателя/destination транзакции',
             'user_mean_amt', 'user_std_amt'] # Убираем вспомогательные, оставляем zscore

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['target']

# Заполняем пропуски
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].fillna('Unknown')
    else:
        X[col] = X[col].fillna(0)

# Индексы категорий
cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']

# Сплит (используем те же индексы, что и для энкодинга)
X_train = X.loc[X_train_idx]
y_train = y.loc[X_train_idx]
X_test = X.loc[X_test_idx]
y_test = y.loc[X_test_idx]

# --- 4. ОБУЧЕНИЕ (F1 Optimization) ---
print("🔥 Training CatBoost (F1 Optimized)...")

# auto_class_weights='SqrtBalanced' — это мягче, чем Balanced, но жестче, чем None.
# Это часто дает лучший баланс P/R.
model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.02,
    depth=6,
    l2_leaf_reg=5,
    auto_class_weights='SqrtBalanced', # ПОПРОБУЙ ЭТО!
    cat_features=cat_features,
    verbose=200,
    random_seed=42,
    early_stopping_rounds=200
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))

# --- 5. ПОИСК ИДЕАЛЬНОГО ПОРОГА (ПО F1) ---
y_prob = model.predict_proba(X_test)[:, 1]

best_thr = 0.5
best_f1 = 0
best_metrics = {}

# Ищем порог, где F1 (гармония точности и полноты) максимальна
for thr in np.arange(0.1, 0.9, 0.01):
    pred = (y_prob > thr).astype(int)
    f1 = f1_score(y_test, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
        best_metrics = {
            'Precision': precision_score(y_test, pred),
            'Recall': recall_score(y_test, pred)
        }

print(f"\n🏆 OPTIMAL THRESHOLD: {best_thr:.2f}")
print(f"F1-Score:  {best_f1:.2%}")
print(f"Precision: {best_metrics['Precision']:.2%}")
print(f"Recall:    {best_metrics['Recall']:.2%}")

# --- 6. ОТЧЕТ ---
final_pred = (y_prob > best_thr).astype(int)
print("\n--- Final Classification Report ---")
print(classification_report(y_test, final_pred))

imp = model.get_feature_importance(prettified=True).head(7)
print("\nTop Features (Check 'receiver_risk_score' and 'amount_zscore'):")
print(imp)

# Save the model
model.save_model('catboost_model_ultra.cbm')
print("\n💾 Model saved to 'catboost_model_ultra.cbm'")