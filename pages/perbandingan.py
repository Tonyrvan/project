import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analytics | Stunting AI", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .hero-title { font-size: 42px; font-weight: 800; letter-spacing: -1.5px; background: -webkit-linear-gradient(#fff, #999); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2; margin: 0; }
    .hero-subtitle { color: #86868b; font-size: 16px; margin-top: 10px; line-height: 1.5; }
    .status-badge { background: rgba(10, 132, 255, 0.1); color: #0A84FF; padding: 6px 16px; border-radius: 100px; font-size: 13px; font-weight: 600; border: 1px solid rgba(10, 132, 255, 0.2); display: inline-block; margin-bottom: 10px; }
    .best-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(15px); border-radius: 20px; padding: 24px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); }
    
    /* Style Tabel Custom */
    .metric-table { width: 100%; border-collapse: collapse; margin-top: 10px; background: rgba(255,255,255,0.02); border-radius: 10px; overflow: hidden; }
    .metric-table th { background: rgba(255,255,255,0.05); color: #86868b; padding: 12px; text-align: left; font-size: 12px; text-transform: uppercase; }
    .metric-table td { padding: 12px; border-top: 1px solid rgba(255,255,255,0.05); font-size: 15px; }
    .metric-val { font-weight: 700; color: #fff; }
    </style>
    """, unsafe_allow_html=True)

def prepare_input(X_raw, scaler_path):
    try:
        if not os.path.exists(scaler_path): return None
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        X_scaled = scaler.transform(X_raw)
        return X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    except: return None

@st.cache_data(ttl=1)
def load_and_evaluate_all():
    variations = ["bilstm256", "bilstm512", "stacked256", "stacked512"]
    summary, storage, errors = [], {}, []

    for name in variations:
        try:
            model_path, hist_path, data_path, scaler_path = f"model_{name}.h5", f"history_{name}.pkl", f"data_test_{name}.pkl", f"scaler_{name}.pkl"
            if name == "stacked512" and not os.path.exists(data_path):
                if os.path.exists("data_test_stacked512 .pkl"): data_path = "data_test_stacked512 .pkl"

            if not all(os.path.exists(p) for p in [model_path, hist_path, data_path, scaler_path]):
                continue

            model = load_model(model_path)
            with open(hist_path, "rb") as f: hist = pickle.load(f)
            with open(data_path, "rb") as f: X_test_raw, y_test = pickle.load(f)

            X_ready = prepare_input(X_test_raw, scaler_path)
            if X_ready is None: continue

            y_pred = (model.predict(X_ready, verbose=0) > 0.5).astype(int).flatten()

            res = {
                "Model Variant": name.upper(),
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "Type": "BiLSTM" if "bilstm" in name else "Stacked"
            }
            summary.append(res)
            storage[name] = {"history": hist, "metrics": res, "y_test": y_test, "y_pred": y_pred}
        except: continue
    return pd.DataFrame(summary), storage

df_global, data_storage = load_and_evaluate_all()

# --- HERO & TOP PERFORMANCE ---
col_t, col_b = st.columns([1.8, 1.2])
with col_t:
    st.markdown('<div class="status-badge">Ilmu Komputer</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Deep Learning Performance</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Perbandingan model arsitektur BiLSTM dan Stacked LSTM.</p>', unsafe_allow_html=True)

with col_b:
    if not df_global.empty:
        best_model = df_global.loc[df_global["F1-Score"].idxmax()]
        st.markdown(f"""
            <div class="best-card">
                <p style="color: #86868b; margin: 0; font-size: 11px; font-weight: 600;">🏆 TOP PERFORMER</p>
                <h2 style="color: #fff; margin: 5px 0; font-size: 24px;">{best_model['Model Variant']}</h2>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <div><p style="color: #86868b; margin:0; font-size:10px;">F1-SCORE</p><p style="color:#0A84FF; font-weight:800; margin:0; font-size:18px;">{best_model['F1-Score']:.2%}</p></div>
                    <div><p style="color: #86868b; margin:0; font-size:10px;">ACCURACY</p><p style="color:#fff; font-weight:800; margin:0; font-size:18px;">{best_model['Accuracy']:.2%}</p></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

st.divider()

# --- MAIN DASHBOARD ---
if df_global.empty:
    st.error("Model tidak ditemukan.")
    st.stop()

st.subheader("📊 Comparative Detail (Batch Size)")
tab256, tab512 = st.tabs(["Batch Size 256", "Batch Size 512"])

def _get_val_acc(hist):
    return hist.get("val_accuracy", hist.get("val_acc", [0]))

def draw_metric_table(title, metrics):
    st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <p style="color: #86868b; font-weight: 600; margin-bottom: 5px;">{title}</p>
            <table class="metric-table">
                <tr><th>Metric</th><th>Score</th></tr>
                <tr><td>Accuracy</td><td class="metric-val">{metrics['Accuracy']:.4f}</td></tr>
                <tr><td>F1-Score</td><td class="metric-val">{metrics['F1-Score']:.4f}</td></tr>
                <tr><td>Recall</td><td class="metric-val">{metrics['Recall']:.4f}</td></tr>
            </table>
        </div>
    """, unsafe_allow_html=True)

def render_tab_content(batch_label):
    k_s, k_b = f"stacked{batch_label}", f"bilstm{batch_label}"
    has_s, has_b = k_s in data_storage, k_b in data_storage
    
    # 1. Tabel Metrik 
    t1, t2 = st.columns(2)
    with t1:
        if has_s: draw_metric_table("STACKED LSTM", data_storage[k_s]['metrics'])
    with t2:
        if has_b: draw_metric_table("BiLSTM", data_storage[k_b]['metrics'])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Grafik Loss & Accuracy
    g1, g2 = st.columns(2)
    with g1:
        st.write("**Validation Loss Curve**")
        fig = go.Figure()
        if has_s: fig.add_trace(go.Scatter(y=data_storage[k_s]["history"].get('val_loss', []), name="Stacked", line=dict(color="#FF9500", width=3)))
        if has_b: fig.add_trace(go.Scatter(y=data_storage[k_b]["history"].get('val_loss', []), name="BiLSTM", line=dict(color="#007AFF", width=3)))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        st.write("**Validation Accuracy Curve**")
        fig = go.Figure()
        if has_s: fig.add_trace(go.Scatter(y=_get_val_acc(data_storage[k_s]["history"]), name="Stacked", line=dict(color="#FF9500", width=3)))
        if has_b: fig.add_trace(go.Scatter(y=_get_val_acc(data_storage[k_b]["history"]), name="BiLSTM", line=dict(color="#007AFF", width=3)))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with tab256: render_tab_content("256")
with tab512: render_tab_content("512")