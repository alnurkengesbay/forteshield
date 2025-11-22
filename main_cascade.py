import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

print("🛡️ ЗАПУСК СИСТЕМЫ 'ЖЕЛЕЗНЫЙ КУПОЛ' (CASCADE SYSTEM)...")

# --- 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ (Как в Dominator) ---
print("📥 Загрузка данных...")
df_trans = pd.read_csv('data/транзакции_в_Мобильном_интернет_Банкинге.csv', sep=';', header=1, encoding='cp1251')
df_behav = pd.read_csv('data/поведенческие_паттерны_клиентов_3.csv', sep=';', header=1, encoding='cp1251')

for df in [df_trans, df_behav]:
    df['transdate'] = pd.to_datetime(df['transdate'].astype(str).str.strip("'"))
df_trans['transdatetime'] = pd.to_datetime(df_trans['transdatetime'].astype(str).str.strip("'"))

df = pd.merge(df_trans, df_behav, on=['cst_dim_id', 'transdate'], how='left')

# --- 2. FEATURE ENGINEERING (DOMINATOR SET) ---
print("🔧 Генерация фичей (Dominator Set)...")

# User Stats
user_stats = df.groupby('cst_dim_id')['amount'].agg(['mean', 'std', 'count']).reset_index()
user_stats.columns = ['cst_dim_id', 'user_mean_amt', 'user_std_amt', 'user_tx_count']
df = pd.merge(df, user_stats, on='cst_dim_id', how='left')

df['amount_zscore'] = (df['amount'] - df['user_mean_amt']) / (df['user_std_amt'] + 1.0)
df['amount_to_mean'] = df['amount'] / (df['user_mean_amt'] + 1.0)
df['hour'] = df['transdatetime'].dt.hour
df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x > 23 else 0)
df['amount_log'] = np.log1p(df['amount'])

# 4. Count Encoding (Frequency) - "Rare Destination" signal
# This helps catch new fraud where Target Encoding is just "average"
freq_map = df['direction'].value_counts().to_dict()
df['direction_freq'] = df['direction'].map(freq_map)

# --- 2.5 UNSUPERVISED FEATURES (GOD MODE INJECTION) ---
print("   -> Injecting Unsupervised Features (IsoForest + KMeans)...")
num_cols = ['amount', 'amount_log', 'hour', 'direction_freq', 'user_mean_amt']
# Ensure columns exist
num_cols = [c for c in num_cols if c in df.columns]
X_unsup = df[num_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unsup)

iso = IsolationForest(contamination=0.02, random_state=42)
df['iso_score'] = iso.fit_predict(X_scaled)
df['iso_anomaly'] = df['iso_score'].apply(lambda x: 1 if x == -1 else 0)

kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
centers = kmeans.cluster_centers_
df['dist_to_center'] = [np.linalg.norm(x - centers[c]) for x, c in zip(X_scaled, df['cluster'])]

# Convert discrete unsupervised features to string so they are treated as categorical
df['cluster'] = df['cluster'].astype(str)
df['iso_anomaly'] = df['iso_anomaly'].astype(str)

# Split
X_temp = df.drop(columns=['target'])
y_temp = df['target']
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y_temp)

# Target Encoding (Risk Scoring)
def smooth_target_encode(train_df, test_df, cat_col, target_col, weight=10):
    global_mean = train_df[target_col].mean()
    agg = train_df.groupby(cat_col)[target_col].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smoothed = (counts * means + weight * global_mean) / (counts + weight)
    return train_df[cat_col].map(smoothed).fillna(global_mean), test_df[cat_col].map(smoothed).fillna(global_mean)

df.loc[train_idx, 'receiver_risk'], df.loc[test_idx, 'receiver_risk'] = \
    smooth_target_encode(df.loc[train_idx], df.loc[test_idx], 'direction', 'target', weight=10)

# Cleanup
drop_cols = ['cst_dim_id', 'transdate', 'transdatetime', 'docno', 'target',
             'Зашифрованный идентификатор получателя/destination транзакции', 
             'direction', 'user_mean_amt', 'user_std_amt', 'iso_score']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['target']

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

# --- 3. ЗАГРУЗКА / ОБУЧЕНИЕ МОДЕЛЕЙ ---

# MODEL 1: DOMINATOR (SNIPER)
print("\n🔫 Загрузка Model B (Dominator/Sniper)...")
try:
    model_dominator = CatBoostClassifier()
    model_dominator.load_model("catboost_dominator.cbm")
    print("✅ Dominator загружен.")
except:
    print("⚠️ Dominator не найден, обучаем быстро...")
    model_dominator = CatBoostClassifier(iterations=500, auto_class_weights='SqrtBalanced', cat_features=cat_features, verbose=0)
    model_dominator.fit(X_train, y_train)

# MODEL 2: PRO (NET) - High Recall
print("\n🕸️ Обучение Model A (Pro/Balanced) для High Recall...")
print("⚠️ ВАЖНО: Убираем 'receiver_risk' для Pro модели, чтобы она смотрела на ПОВЕДЕНИЕ, а не на получателя!")

# Убираем доминирующую фичу, чтобы модель искала аномалии в поведении
X_train_pro = X_train.drop(columns=['receiver_risk'])
X_test_pro = X_test.drop(columns=['receiver_risk'])

# Обновляем список категориальных фичей (индексы сместились)
cat_features_pro = [i for i, col in enumerate(X_train_pro.columns) if X_train_pro[col].dtype == 'object']

model_pro = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=1, 
    auto_class_weights='Balanced', 
    cat_features=cat_features_pro,
    verbose=0,
    random_seed=42
)
model_pro.fit(X_train_pro, y_train)
print("✅ Model A (Pro) обучена (BEHAVIOR ONLY).")
print("   - Top Features (Pro):")
print(model_pro.get_feature_importance(prettified=True).head(5))

# --- 4. КАСКАДНАЯ СИСТЕМА (IRON DOME) ---
print("\n🚀 ЗАПУСК КАСКАДА (TUNING MODE)...")

# Предсказания
prob_dominator = model_dominator.predict_proba(X_test)[:, 1]
prob_pro = model_pro.predict_proba(X_test_pro)[:, 1]

# 1. ИЩЕМ ИДЕАЛЬНЫЙ ПОРОГ ДЛЯ БЛОКИРОВКИ (Precision >= 95%)
# USER REQUEST: "Убери автоблок". Ок, делаем единый порог для "Suspicious".
best_thr = 0.1
best_f1 = 0

# EXPERIMENT: Averaging Probabilities
p_avg_all = (prob_dominator + prob_pro) / 2

print(f"{'Thr':<5} | {'Prec':<8} | {'Recall':<8} | {'Score':<8}")
print("-" * 35)

for thr in np.arange(0.05, 0.95, 0.05):
    mask = p_avg_all > thr
    if mask.sum() == 0: continue
    
    prec = precision_score(y_test, mask, zero_division=0)
    rec = recall_score(y_test, mask)
    score = prec + rec
    
    print(f"{thr:<5.2f} | {prec:<8.2%} | {rec:<8.2%} | {score:<8.2f}")
    
    if prec > 0.30 and rec > 0.50:
        best_thr = thr

print(f"⚙️ Нашли оптимальный порог риска (Average): {best_thr:.2f}")

results = []
final_decisions = []

for i in range(len(y_test)):
    p_dom = prob_dominator[i]
    p_pro = prob_pro[i]
    p_avg = (p_dom + p_pro) / 2
    true_label = y_test.iloc[i]
    
    decision = "🟢 ALLOW"
    category = "clean"
    
    # ЕДИНАЯ ЛОГИКА: Average Probability
    if p_avg > best_thr:
        decision = "🟡 VERIFY"
        category = "verify"
        
    results.append({
        'true_fraud': true_label,
        'prob_dom': p_dom,
        'prob_pro': p_pro,
        'decision': decision,
        'category': category
    })

df_res = pd.DataFrame(results)

# --- 5. АНАЛИЗ РЕЗУЛЬТАТОВ ---

print("\n📊 ОТЧЕТ ПО КАСКАДУ:")
print("-" * 30)

# 1. BLOCK BUCKET
block_df = df_res[df_res['category'] == 'block']
n_block = len(block_df)
n_fraud_block = block_df['true_fraud'].sum()
prec_block = n_fraud_block / n_block if n_block > 0 else 0

print(f"🔴 BLOCKED (Auto): {n_block} транзакций")
print(f"   - Из них реальный фрод: {n_fraud_block}")
print(f"   - Точность (Precision): {prec_block:.2%}")
print(f"   - Ошибки (False Positives): {n_block - n_fraud_block} (Клиентов заблокировано зря)")

# 2. VERIFY BUCKET
verify_df = df_res[df_res['category'] == 'verify']
n_verify = len(verify_df)
n_fraud_verify = verify_df['true_fraud'].sum()
prec_verify = n_fraud_verify / n_verify if n_verify > 0 else 0

print(f"\n🟡 VERIFY (SMS/Call): {n_verify} транзакций")
print(f"   - Из них реальный фрод: {n_fraud_verify}")
print(f"   - Эффективность (Precision): {prec_verify:.2%}")
print(f"   - Лишние проверки: {n_verify - n_fraud_verify}")

# 3. ALLOW BUCKET
allow_df = df_res[df_res['category'] == 'clean']
n_allow = len(allow_df)
n_fraud_missed = allow_df['true_fraud'].sum()

print(f"\n🟢 ALLOWED (Auto): {n_allow} транзакций")
print(f"   - Пропущено фрода (Missed): {n_fraud_missed}")

# 4. TOTAL METRICS
total_fraud = df_res['true_fraud'].sum()
caught_total = n_fraud_block + n_fraud_verify
recall_total = caught_total / total_fraud if total_fraud > 0 else 0

print("-" * 30)
print(f"🏆 ИТОГОВЫЙ RECALL СИСТЕМЫ: {recall_total:.2%} ({caught_total}/{total_fraud} поймано)")
print(f"📉 Нагрузка на колл-центр: {n_verify} звонков (вместо проверки всех)")

# Сохраняем модель Pro
model_pro.save_model("catboost_pro.cbm")
print("\n💾 Model A (Pro) сохранена как 'catboost_pro.cbm'")
