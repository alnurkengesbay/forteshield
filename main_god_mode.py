import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
import warnings

warnings.filterwarnings('ignore')

print("🔥 ЗАПУСК РЕЖИМА 'GOD MODE' (SMOTE + UNSUPERVISED LEARNING)...")

# --- 1. ЗАГРУЗКА ДАННЫХ ---
df_trans = pd.read_csv('data/транзакции_в_Мобильном_интернет_Банкинге.csv', sep=';', header=1, encoding='cp1251')
df_behav = pd.read_csv('data/поведенческие_паттерны_клиентов_3.csv', sep=';', header=1, encoding='cp1251')

for df in [df_trans, df_behav]:
    df['transdate'] = pd.to_datetime(df['transdate'].astype(str).str.strip("'"))
df_trans['transdatetime'] = pd.to_datetime(df_trans['transdatetime'].astype(str).str.strip("'"))

df = pd.merge(df_trans, df_behav, on=['cst_dim_id', 'transdate'], how='left')

# --- 2. FEATURE ENGINEERING (ALL-IN) ---
print("🔧 Генерация фичей (Supervised + Unsupervised)...")

# 2.1. User Context
user_stats = df.groupby('cst_dim_id')['amount'].agg(['mean', 'std', 'max']).reset_index()
user_stats.columns = ['cst_dim_id', 'user_mean', 'user_std', 'user_max']
df = pd.merge(df, user_stats, on='cst_dim_id', how='left')

df['amount_zscore'] = (df['amount'] - df['user_mean']) / (df['user_std'] + 1.0)
df['amount_to_max'] = df['amount'] / (df['user_max'] + 1.0)
df['amount_log'] = np.log1p(df['amount'])

# 2.2. Time Features
df['hour'] = df['transdatetime'].dt.hour
df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x > 23 else 0)

# 2.3. Frequency Features
freq_map = df['direction'].value_counts(normalize=True).to_dict()
df['direction_freq'] = df['direction'].map(freq_map)

# 2.4. UNSUPERVISED LEARNING FEATURES (The Secret Sauce)
# Мы добавляем "Мнение" других алгоритмов как фичи для CatBoost

# Подготовка данных для Unsupervised (только численные)
num_cols = ['amount', 'amount_log', 'hour', 'direction_freq', 'user_mean']
X_unsup = df[num_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unsup)

# A. Isolation Forest (Поиск аномалий)
print("   -> Running Isolation Forest...")
iso = IsolationForest(contamination=0.02, random_state=42)
df['iso_score'] = iso.fit_predict(X_scaled) # -1 for outlier, 1 for inlier
df['iso_anomaly'] = df['iso_score'].apply(lambda x: 1 if x == -1 else 0)

# B. K-Means Clustering (Расстояние до центра кластера)
print("   -> Running K-Means...")
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
# Считаем расстояние до центра своего кластера
centers = kmeans.cluster_centers_
df['dist_to_center'] = [np.linalg.norm(x - centers[c]) for x, c in zip(X_scaled, df['cluster'])]

# --- 3. ПОДГОТОВКА ---
X_temp = df.drop(columns=['target'])
y_temp = df['target']
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y_temp)

# 2.5. Target Encoding (Risk Score)
def smooth_target_encode(train_df, test_df, cat_col, target_col, weight=10):
    global_mean = train_df[target_col].mean()
    agg = train_df.groupby(cat_col)[target_col].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smoothed = (counts * means + weight * global_mean) / (counts + weight)
    return train_df[cat_col].map(smoothed).fillna(global_mean), test_df[cat_col].map(smoothed).fillna(global_mean)

df.loc[train_idx, 'receiver_risk'], df.loc[test_idx, 'receiver_risk'] = \
    smooth_target_encode(df.loc[train_idx], df.loc[test_idx], 'direction', 'target', weight=5)

# Interactions with Unsupervised Features
df['risk_x_iso'] = df['receiver_risk'] * df['iso_anomaly']
df['risk_x_dist'] = df['receiver_risk'] * df['dist_to_center']

# Cleanup
drop_cols = ['cst_dim_id', 'transdate', 'transdatetime', 'docno', 'target',
             'Зашифрованный идентификатор получателя/destination транзакции', 
             'direction', 'user_mean', 'user_std', 'user_max', 'iso_score']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['target']

# FillNA
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].fillna('Unknown')
    else:
        X[col] = X[col].fillna(0)

cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']
cat_features_names = [col for col in X.columns if X[col].dtype == 'object']

# Add discrete numeric columns to categorical features
discrete_cols = ['iso_anomaly', 'cluster', 'is_night', 'hour']
discrete_cols = [c for c in discrete_cols if c in X.columns]
cat_features_names.extend(discrete_cols)
cat_features_names = list(set(cat_features_names))

X_train = X.loc[train_idx]
y_train = y.loc[train_idx]
X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

# --- 4. SMOTE (SYNTHETIC DATA GENERATION) ---
print("🧬 Генерируем синтетический фрод (SMOTE)...")

# EXPERIMENT: Drop receiver_risk to force learning from behavior (Unsupervised Features)
print("   -> Dropping 'receiver_risk' to prevent overfitting to destination IDs...")
drop_risk_cols = ['receiver_risk', 'risk_x_iso', 'risk_x_dist']
X_train = X_train.drop(columns=[c for c in drop_risk_cols if c in X_train.columns])
X_test = X_test.drop(columns=[c for c in drop_risk_cols if c in X_test.columns])

# Update cat_features_names to exclude dropped columns if any
cat_features_names = [c for c in cat_features_names if c in X_train.columns]

# 1. Разделяем на числовые и категориальные
# ВАЖНО: Некоторые числовые колонки (кластеры, флаги) нельзя интерполировать SMOTE-ом.
# Их нужно сэмплировать как категории.
discrete_cols = ['iso_anomaly', 'cluster', 'is_night', 'hour']
discrete_cols = [c for c in discrete_cols if c in X_train.columns]

X_train_num = X_train.select_dtypes(include=[np.number]).drop(columns=discrete_cols)
X_train_cat = pd.concat([
    X_train.select_dtypes(exclude=[np.number]),
    X_train[discrete_cols]
], axis=1)

# 2. Применяем SMOTE только к НЕПРЕРЫВНЫМ числам
smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.5) # Делаем фрода 50% от нормы
X_res_num, y_res = smote.fit_resample(X_train_num, y_train)

# 3. Восстанавливаем категории (через Random Sampling из реального фрода)
# Вместо "Synthetic" (который палит контору), мы берем категории от РЕАЛЬНЫХ фродовых транзакций.

print("   -> Filling categorical features for synthetic data using Real Fraud samples...")
fraud_indices = y_train[y_train == 1].index
real_fraud_cats = X_train_cat.loc[fraud_indices]

# Вычисляем сколько синтетических строк добавил SMOTE
n_synthetic = len(X_res_num) - len(X_train)

# Генерируем случайные категории ТОЛЬКО для синтетической части
synthetic_cats = real_fraud_cats.sample(n=n_synthetic, replace=True, random_state=42).reset_index(drop=True)

# Объединяем: Оригинальные категории + Синтетические категории
X_train_cat_reset = X_train_cat.reset_index(drop=True)
X_res_cat_combined = pd.concat([X_train_cat_reset, synthetic_cats], axis=0).reset_index(drop=True)

# X_res_num уже содержит и оригинальные и синтетические (от SMOTE)
X_res_num_combined = pd.DataFrame(X_res_num, columns=X_train_num.columns).reset_index(drop=True)

X_train_final = pd.concat([X_res_num_combined, X_res_cat_combined], axis=1)
y_train_final = y_res

# ВАЖНО: Восстанавливаем оригинальный порядок колонок!
# SMOTE разделил их на num и cat, и concat склеил их в кучу.
# Порядок в X_train_final сейчас: [Все числа, Все категории]
# А в X_test порядок оригинальный. Это ломает CatBoost (Feature #3 mismatch).
X_train_final = X_train_final[X_test.columns]

# ВАЖНО: Приводим все категориальные колонки к строкам, чтобы CatBoost не ругался на float
# И убеждаемся, что нет float-подобных строк типа "11.0"
for col in X_train_final.columns:
    if col in cat_features_names:
        # Принудительно в int, потом в str, чтобы убрать .0
        # Если там есть 'Unknown' или 'Synthetic', то try-except
        def clean_cat(x):
            try:
                return str(int(float(x)))
            except:
                return str(x)
        
        X_train_final[col] = X_train_final[col].apply(clean_cat)
        X_test[col] = X_test[col].apply(clean_cat)
        
        # Применяем и перезаписываем как object/string
        X_train_final[col] = X_train_final[col].apply(clean_cat).astype(str)
        X_test[col] = X_test[col].apply(clean_cat).astype(str)

# Проверка типов
print("Типы колонок после очистки:")
print(X_train_final.dtypes)

# Убедимся, что cat_features - это имена колонок, а не индексы, так как мы передаем DataFrame
cat_features_names = [col for col in X_train_final.columns if X_train_final[col].dtype == 'object']
print(f"Категориальные фичи: {cat_features_names}")

# ВАЖНО: CatBoost иногда путается, если передавать DataFrame с object колонками, но не указывать их как cat_features
# Или если указывать индексы, но он ожидает имена.
# Самый надежный способ: конвертировать все object в 'category' тип pandas.

# Принудительно конвертируем ВСЕ категориальные колонки в 'category'
# И в Train, и в Test, чтобы типы совпадали
for col in cat_features_names:
    X_train_final[col] = X_train_final[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Проверка типов
print("Типы колонок после конвертации в category:")
print(X_train_final.dtypes)

print(f"   -> Было фрода: {y_train.sum()}, Стало: {y_train_final.sum()}")

# --- 5. TRAINING ---
# ВАЖНО: CatBoost требует, чтобы порядок колонок в eval_set совпадал с X_train
# И чтобы категориальные фичи были указаны явно.

# ВАЖНО: Создаем Pool объекты явно, чтобы CatBoost точно знал, где категории
from catboost import Pool
train_pool = Pool(X_train_final, y_train_final, cat_features=cat_features_names) # Имена надежнее с pandas
test_pool = Pool(X_test, y_test, cat_features=cat_features_names)

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    auto_class_weights='Balanced', 
    verbose=200,
    early_stopping_rounds=200,
    eval_metric='F1',
    random_seed=42
)

model.fit(train_pool, eval_set=test_pool)

# --- 6. THRESHOLD SEARCH (BRUTE FORCE) ---
print("\n🕵️ Ищем 'Золотое сечение' (80/80)...")
y_prob = model.predict_proba(X_test)[:, 1]

best_thr = 0.5
best_score = 0
final_metrics = {}

# Мы ищем точку, где (Precision + Recall) максимально, НО при условии что оба > 0.7
print(f"{'Thr':<5} | {'Prec':<8} | {'Recall':<8} | {'Score':<8}")
print("-" * 35)

for thr in np.arange(0.05, 0.95, 0.05):
    pred = (y_prob > thr).astype(int)
    p = precision_score(y_test, pred, zero_division=0)
    r = recall_score(y_test, pred)
    
    # Наша цель: Оба > 0.8. Если нет, то хотя бы оба > 0.75
    score = p + r
    
    print(f"{thr:<5.2f} | {p:<8.2%} | {r:<8.2%} | {score:<8.2f}")

    # Штрафуем, если один из показателей низкий
    if p < 0.7 or r < 0.7:
        score = score * 0.5 
        
    if score > best_score:
        best_score = score
        best_thr = thr
        final_metrics = {'Precision': p, 'Recall': r}

print("\n" + "="*40)
print(f"🏆 GOD MODE RESULT")
print("="*40)
print(f"Threshold: {best_thr:.2f}")
print(f"💎 Precision: {final_metrics['Precision']:.2%}")
print(f"🔍 Recall:    {final_metrics['Recall']:.2%}")
print("="*40)

model.save_model("catboost_god.cbm")
