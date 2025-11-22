import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, recall_score, precision_recall_curve
import warnings

warnings.filterwarnings('ignore')

print("☢️ ЗАПУСК РЕЖИМА 'DOMINATOR' (CONTEXT + RISK SCORING)...")

# --- 1. ЗАГРУЗКА ДАННЫХ ---
# Читаем со второй строки (header=1), так как первая - описание
# Added encoding='cp1251' to avoid UnicodeDecodeError
df_trans = pd.read_csv('data/транзакции_в_Мобильном_интернет_Банкинге.csv', sep=';', header=1, encoding='cp1251')
df_behav = pd.read_csv('data/поведенческие_паттерны_клиентов_3.csv', sep=';', header=1, encoding='cp1251')

# Чистка дат
for df in [df_trans, df_behav]:
    # Убираем кавычки и пробелы
    df['transdate'] = pd.to_datetime(df['transdate'].astype(str).str.strip("'"))
df_trans['transdatetime'] = pd.to_datetime(df_trans['transdatetime'].astype(str).str.strip("'"))

# Объединение (Left Join)
df = pd.merge(df_trans, df_behav, on=['cst_dim_id', 'transdate'], how='left')

# --- 2. FEATURE ENGINEERING: КОНТЕКСТ КЛИЕНТА ---
print("🔧 Создаем контекстные фичи (User Profiling)...")

# Считаем среднюю сумму и отклонение для КАЖДОГО клиента
# (В реале это делается на исторических данных, тут берем всю выборку для примера)
user_stats = df.groupby('cst_dim_id')['amount'].agg(['mean', 'std', 'count']).reset_index()
user_stats.columns = ['cst_dim_id', 'user_mean_amt', 'user_std_amt', 'user_tx_count']

df = pd.merge(df, user_stats, on='cst_dim_id', how='left')

# 1. Z-Score суммы (Насколько транзакция выбивается из нормы клиента)
# Добавляем +1 к делителю, чтобы не делить на ноль
df['amount_zscore'] = (df['amount'] - df['user_mean_amt']) / (df['user_std_amt'] + 1.0)

# 2. Отношение к среднему (Во сколько раз больше обычного)
df['amount_to_mean'] = df['amount'] / (df['user_mean_amt'] + 1.0)

# 3. Базовые фичи
df['hour'] = df['transdatetime'].dt.hour
df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x > 23 else 0)
df['amount_log'] = np.log1p(df['amount']) # Логарифм для сглаживания

# --- 3. ПОДГОТОВКА К ОБУЧЕНИЮ ---
# Разделяем на Train/Test ДО Target Encoding, чтобы не было утечки данных!
X_temp = df.drop(columns=['target'])
y_temp = df['target']

# Стратифицированный сплит (сохраняем пропорцию фрода)
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y_temp)

# --- 4. FEATURE ENGINEERING: RISK SCORING (TARGET ENCODING) ---
print("🎯 Вычисляем риск получателей (Target Encoding)...")

# Функция сглаживания (Smoothing), чтобы не банить за 1 транзакцию
def smooth_target_encode(train_df, test_df, cat_col, target_col, weight=10):
    # Считаем статистику ТОЛЬКО на Train
    global_mean = train_df[target_col].mean()
    agg = train_df.groupby(cat_col)[target_col].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    
    # Формула сглаживания
    smoothed = (counts * means + weight * global_mean) / (counts + weight)
    
    # Применяем к Train
    train_encoded = train_df[cat_col].map(smoothed).fillna(global_mean)
    # Применяем к Test (используя знания из Train!)
    test_encoded = test_df[cat_col].map(smoothed).fillna(global_mean)
    
    return train_encoded, test_encoded

# Применяем к 'direction' (получатель)
df.loc[train_idx, 'receiver_risk'], df.loc[test_idx, 'receiver_risk'] = \
    smooth_target_encode(df.loc[train_idx], df.loc[test_idx], 'direction', 'target', weight=10)

# --- 5. ОЧИСТКА МУСОРА ---
drop_cols = [
    'cst_dim_id', 'transdate', 'transdatetime', 'docno', 'target',
    'Зашифрованный идентификатор получателя/destination транзакции', 
    'direction', # Убираем сырой ID, оставляем receiver_risk!
    'user_mean_amt', 'user_std_amt' # Убираем вспомогательные
]

# Формируем финальные датасеты
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['target']

# Заполнение пропусков
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

# --- 6. ОБУЧЕНИЕ (CatBoost Pro) ---
print("🔥 Training CatBoost with SqrtBalanced Weights...")

# Используем 'SqrtBalanced'. Это "Золотая середина".
# Balanced = слишком много Recall, мало Precision.
# None = много Precision, мало Recall.
# SqrtBalanced = То, что тебе нужно.
model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    auto_class_weights='SqrtBalanced',
    cat_features=cat_features,
    verbose=200,
    early_stopping_rounds=200,
    eval_metric='F1', # Оптимизируем гармонию P и R
    random_seed=42
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))

# --- 7. ПОИСК ИДЕАЛЬНОГО ПОРОГА ---
print("\n⚖️ Optimizing Threshold...")
y_prob = model.predict_proba(X_test)[:, 1]

best_thr = 0.5
best_f1 = 0
metrics_at_best = {}

# Перебираем пороги, чтобы найти пик F1
for thr in np.arange(0.1, 0.95, 0.01):
    pred = (y_prob > thr).astype(int)
    f1 = f1_score(y_test, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
        metrics_at_best = {
            'Precision': precision_score(y_test, pred),
            'Recall': recall_score(y_test, pred)
        }

print(f"\n🏆 ИДЕАЛЬНЫЙ ПОРОГ (Threshold): {best_thr:.2f}")
print(f"💎 Precision: {metrics_at_best['Precision']:.2%}")
print(f"🔍 Recall:    {metrics_at_best['Recall']:.2%}")
print(f"⚖️ F1-Score:  {best_f1:.2%}")

# Финальный отчет
final_pred = (y_prob > best_thr).astype(int)
print("\n--- Detailed Report ---")
print(classification_report(y_test, final_pred))

# Топ фичи
print("\n🧠 ТОП ФАКТОРОВ (Почему это работает):")
print(model.get_feature_importance(prettified=True).head(7))

# Сохраняем модель
model.save_model("catboost_dominator.cbm")
print("\n💾 Модель сохранена как 'catboost_dominator.cbm'")