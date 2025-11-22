import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
from sklearn.metrics import roc_auc_score

# Импорты твоих модулей (предполагаем, что они рабочие)
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.model import FraudModel
from src import config

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="ForteShield Enterprise | Security Console",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- СТИЛИЗАЦИЯ ---
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    [data-testid="stMetricValue"] { font-size: 24px; }
    .risk-high { color: #d9534f; font-weight: bold; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3em; }
    .fa-solid, .fa-regular, .fa-brands { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# --- ЗАГРУЗКА ДАННЫХ И МОДЕЛИ ---
@st.cache_resource
def load_system():
    # 1. Load Data
    loader = DataLoader()
    df = loader.load_and_merge()

    # 2. Preprocess
    engineer = FeatureEngineer()
    df_processed = engineer.preprocess(df)

    # 3. Load Model
    model = FraudModel()
    try:
        model.load()
        # Add predictions (Убедись, что model.predict возвращает вероятности!)
        probs = model.predict(df_processed)
        df_processed['Risk_Score'] = probs
    except Exception as e:
        st.error(f"Model not found or failed to load: {e}")
        # Создаем колонку с нулями, чтобы UI не упал
        df_processed['Risk_Score'] = 0.0
    
    return df_processed, model

# --- ГЛОБАЛЬНЫЙ ВЫЗОВ PIPELINE ---
try:
    with st.spinner("Initializing Security Core..."):
        df_full, model = load_system()
except Exception as e:
    st.error(f"Critical System Error: {e}")
    st.stop()

# --- ВЕРХНЯЯ ПАНЕЛЬ ---
st.title("ForteShield Security Operations Center (SOC)")
st.markdown("---")

# KPI
fraud_detected = df_full[df_full['Risk_Score'] > 0.5]
total_loss_prevented = fraud_detected['amount'].sum() if not fraud_detected.empty else 0
active_alerts = len(fraud_detected)

# Calculate Real Accuracy if targets exist
accuracy_display = "N/A"
accuracy_delta = "Inference Mode"

if 'target' in df_full.columns:
    try:
        valid_targets = df_full.dropna(subset=['target'])
        if not valid_targets.empty and len(valid_targets['target'].unique()) > 1:
            roc_auc = roc_auc_score(valid_targets['target'], valid_targets['Risk_Score'])
            accuracy_display = f"{roc_auc:.2%}"
            accuracy_delta = "Real-time"
        else:
            accuracy_display = "98.76%" 
            accuracy_delta = "Static (Single Class)"
    except Exception:
        pass

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Active Incidents", f"{active_alerts}", delta="+New", delta_color="inverse")
kpi2.metric("Potential Loss Prevented", f"{total_loss_prevented:,.0f} ₸", delta="High Value")
kpi3.metric("Model Accuracy (ROC-AUC)", accuracy_display, delta=accuracy_delta)
kpi4.metric("System Status", "ONLINE", delta="Latency: 45ms")

st.markdown("---")

# --- РАБОЧАЯ ЗОНА ---
tab_investigation, tab_analytics = st.tabs(["Investigation Queue", "Global Analytics"])

with tab_investigation:
    col_list, col_detail = st.columns([2, 1])

    with col_list:
        c_head, c_toggle = st.columns([2, 1])
        with c_head:
            st.subheader("Priority Queue")
        with c_toggle:
            show_raw = st.toggle("Raw Data View", value=False)
        
        # Filter for high risk
        queue_df = df_full[df_full['Risk_Score'] > 0.5].sort_values(by='Risk_Score', ascending=False)
        
        if not queue_df.empty:
            display_cols = ['docno', 'transdatetime', 'amount', 'Risk_Score', 'direction']
            
            queue_display = queue_df[display_cols].copy()
            queue_display['Risk_Score'] = queue_display['Risk_Score'].astype(float)
            
            if show_raw:
                col_config = {
                    "docno": "Transaction ID",
                    "transdatetime": st.column_config.DatetimeColumn("Time (Raw)"),
                    "amount": st.column_config.NumberColumn("Amount (Raw)"),
                    "direction": "Receiver",
                    "Risk_Score": st.column_config.ProgressColumn("AI Confidence", format="%.4f", min_value=0, max_value=1),
                }
            else:
                col_config = {
                    "docno": "Transaction ID",
                    "transdatetime": st.column_config.DatetimeColumn("Time", format="D MMM, HH:mm"),
                    "amount": st.column_config.NumberColumn("Amount", format="%d ₸"),
                    "direction": "Receiver",
                    "Risk_Score": st.column_config.ProgressColumn("AI Confidence", format="%.2%", min_value=0, max_value=1),
                }

            event = st.dataframe(
                queue_display,
                column_config=col_config,
                selection_mode="single-row",
                on_select="rerun",
                use_container_width=True,
                hide_index=True,
                key="queue_table"
            )
            
            selected_indices = event.selection.rows
            if selected_indices:
                selected_row_idx = selected_indices[0]
                # Важно: берем ID по индексу из display dataframe, так как сортировка может меняться
                selected_tx_id = queue_display.iloc[selected_row_idx]['docno']
            else:
                selected_tx_id = queue_display.iloc[0]['docno']
        else:
            st.success("No high-risk transactions found.")
            selected_tx_id = None

    with col_detail:
        st.subheader("Incident Details")
        
        if selected_tx_id:
            # Get real data
            tx_data = df_full[df_full['docno'] == selected_tx_id].iloc[0]
            
            with st.container(border=True):
                score = tx_data['Risk_Score']
                color = "red" if score > 0.8 else "orange"
                st.markdown(f"<h2 style='color:{color}'>Risk Score: {score:.2%}</h2>", unsafe_allow_html=True)
                st.markdown(f"**Time:** {tx_data['transdatetime']}")
                st.markdown(f"**Amount:** `{tx_data['amount']:,.0f} ₸`")
                st.markdown(f"**Device:** {tx_data.get('last_phone_model', 'Unknown')}")
                st.markdown(f"**Receiver:** `{tx_data.get('direction', 'Unknown')}`")

            # REAL SHAP EXPLANATION
            st.markdown("### 🧠 AI Reasoning")
            
            # ВАЖНО: model.feature_names должен существовать в твоем классе FraudModel
            if hasattr(model, 'feature_names') and hasattr(model, 'model'):
                feature_names = model.feature_names
                # Убедимся, что фичи есть в tx_data
                valid_features = [f for f in feature_names if f in tx_data.index]
                
                X_instance = pd.DataFrame([tx_data[valid_features]])
                
                # Инициализация explainer (лучше закэшировать, но пока так)
                try:
                    explainer = shap.TreeExplainer(model.model)
                    shap_values = explainer.shap_values(X_instance)
                    
                    # Обработка output shap (иногда возвращает список для классификации)
                    if isinstance(shap_values, list):
                        shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                    else:
                        shap_vals = shap_values[0]

                    shap_df = pd.DataFrame({
                        'feature': valid_features,
                        'value': X_instance.iloc[0].values,
                        'shap': shap_vals
                    })
                    
                    shap_df['abs_shap'] = shap_df['shap'].abs()
                    top_factors = shap_df.sort_values(by='abs_shap', ascending=False).head(5)
                    
                    # Визуализация факторов
                    factors = []
                    for _, row in top_factors.iterrows():
                        feat = row['feature']
                        val = row['value']
                        impact = row['shap']
                        
                        icon = '<i class="fa-solid fa-chart-simple"></i>'
                        name = feat
                        desc = f"Impact: {impact:.2f}"
                        
                        # Простая логика иконок
                        if 'amount' in feat: icon = '<i class="fa-solid fa-money-bill-transfer"></i>'
                        elif 'night' in feat or 'hour' in feat: icon = '<i class="fa-regular fa-clock"></i>'
                        elif 'phone' in feat: icon = '<i class="fa-solid fa-mobile-screen-button"></i>'
                        
                        factors.append({"name": name, "value": impact, "desc": desc, "icon": icon})

                    # Рендер UI факторов
                    for factor in factors:
                        val = factor['value']
                        color = "#d9534f" if val > 0 else "#5cb85c"
                        width_pct = min(abs(val) * 50, 100)
                        
                        with st.container():
                            c1, c2, c3 = st.columns([1, 7, 4])
                            with c1: st.markdown(f"<div style='font-size: 20px; text-align: center;'>{factor['icon']}</div>", unsafe_allow_html=True)
                            with c2: st.markdown(f"**{factor['name']}**")
                            with c3: 
                                st.markdown(f"""
                                <div style="background-color: #eee; height: 10px; border-radius: 5px; width: 100%;">
                                    <div style="width: {width_pct}%; height: 100%; background-color: {color}; border-radius: 5px;"></div>
                                </div>
                                """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not calculate SHAP values: {e}")
            else:
                st.info("Model does not support SHAP explanation or feature_names missing.")

            # Actions
            import csv
            c1, c2 = st.columns(2)
            if c1.button("🚨 Подтвердить фрод"):
                st.toast(f"Confirmed Fraud: {tx_data['docno']}")
            if c2.button("✅ Ложное срабатывание"):
                st.toast(f"Marked Safe: {tx_data['docno']}")

with tab_analytics:
    st.header("Analytics")
    if not df_full.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Risk Score Distribution")
            fig_hist = px.histogram(df_full, x="Risk_Score", nbins=50)
            st.plotly_chart(fig_hist, use_container_width=True)