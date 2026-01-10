import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Perbandingan Model AI", page_icon="⚖️", layout="wide")

st.title("⚖️ Perbandingan Performa: Stacked LSTM vs BiLSTM")
st.markdown("Evaluasi mendalam performa kedua model menggunakan data uji (Test Data).")

# ==========================================
# 2. FUNGSI LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    files = {
        'model_s': 'model_stacked.h5',
        'model_b': 'model_bilstm.h5',
        'hist_s': 'history_stacked.pkl',
        'hist_b': 'history_bilstm.pkl',
        'test_data': 'data_test.pkl'
    }
    
    loaded = {}
    
    try:
        # Load Models
        loaded['model_s'] = load_model(files['model_s'])
        loaded['model_b'] = load_model(files['model_b'])
        
        # Load Histories
        with open(files['hist_s'], 'rb') as f: loaded['hist_s'] = pickle.load(f)
        with open(files['hist_b'], 'rb') as f: loaded['hist_b'] = pickle.load(f)
            
        # Load Test Data
        with open(files['test_data'], 'rb') as f:
            loaded['X_test'], loaded['y_test'] = pickle.load(f)
            
        return loaded

    except FileNotFoundError:
        return None

# Load Data
data = load_resources()

if data is None:
    st.error("⚠️ File Model/History tidak ditemukan!")
    st.warning("Harap jalankan `train_stacked.py` dan `train_bilstm.py` terlebih dahulu.")
    st.stop()

# Ambil variabel
model_s, model_b = data['model_s'], data['model_b']
hist_s, hist_b = data['hist_s'], data['hist_b']
X_test, y_test = data['X_test'], data['y_test']

y_pred_s = (model_s.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
y_pred_b = (model_b.predict(X_test, verbose=0) > 0.5).astype(int).flatten()

# Fungsi hitung skor biar rapi
def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }

skor_s = get_metrics(y_test, y_pred_s)
skor_b = get_metrics(y_test, y_pred_b)

# ==========================================
# 4. TAMPILAN DASHBOARD
# ==========================================

# --- A. TABEL SKOR ---
st.subheader("🏆 Scorecard Performa")

# Buat DataFrame untuk tabel perbandingan
df_skor = pd.DataFrame({
    "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Stacked LSTM": [skor_s['Accuracy'], skor_s['Precision'], skor_s['Recall'], skor_s['F1-Score']],
    "BiLSTM (Usulan)": [skor_b['Accuracy'], skor_b['Precision'], skor_b['Recall'], skor_b['F1-Score']]
})

# Format angka jadi persen
df_skor["Stacked LSTM"] = df_skor["Stacked LSTM"].apply(lambda x: f"{x:.2%}")
df_skor["BiLSTM (Usulan)"] = df_skor["BiLSTM (Usulan)"].apply(lambda x: f"{x:.2%}")

# Tampilkan Tabel
c_table, c_winner = st.columns([2, 1])
with c_table:
    st.table(df_skor.set_index("Metrik"))

with c_winner:
    # Tentukan pemenang berdasarkan F1-Score
    diff = skor_b['F1-Score'] - skor_s['F1-Score']
    if diff > 0:
        st.success(f"### 🎉 BiLSTM Unggul!\nSelisih F1-Score: **+{diff:.2%}**")
        st.write("Model BiLSTM lebih baik dalam menyeimbangkan Precision dan Recall.")
    elif diff < 0:
        st.warning(f"### ⚠️ Stacked Unggul\nSelisih F1-Score: **{abs(diff):.2%}**")
    else:
        st.info("### ⚖️ Hasil Seimbang\nKedua model memiliki performa identik.")

st.divider()

# --- B. GRAFIK KURVA (LOSS & AKURASI) ---
col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.subheader("📉 Grafik Loss (Error)")
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=hist_s['val_loss'], mode='lines', name='Stacked', line=dict(color='orange', dash='dash')))
    fig_loss.add_trace(go.Scatter(y=hist_b['val_loss'], mode='lines', name='BiLSTM', line=dict(color='blue')))
    fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Loss", hovermode="x unified")
    st.plotly_chart(fig_loss, use_container_width=True)

with col_graph2:
    st.subheader("📈 Grafik Akurasi Training")
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(y=hist_s['val_accuracy'], mode='lines', name='Stacked', line=dict(color='orange', dash='dash')))
    fig_acc.add_trace(go.Scatter(y=hist_b['val_accuracy'], mode='lines', name='BiLSTM', line=dict(color='blue')))
    fig_acc.update_layout(xaxis_title="Epoch", yaxis_title="Akurasi", hovermode="x unified")
    st.plotly_chart(fig_acc, use_container_width=True)

# --- C. CONFUSION MATRIX ---
st.divider()
st.subheader("🧩 Detail Confusion Matrix")
cm_col1, cm_col2 = st.columns(2)

with cm_col1:
    st.write("**Stacked LSTM**")
    cm_s = confusion_matrix(y_test, y_pred_s)
    fig_cm_s = go.Figure(data=go.Heatmap(
        z=cm_s, x=['Aman', 'Berisiko'], y=['Aman', 'Berisiko'],
        text=cm_s, texttemplate="%{text}", colorscale='Oranges'
    ))
    fig_cm_s.update_layout(height=300, margin=dict(t=0, b=0))
    st.plotly_chart(fig_cm_s, use_container_width=True)

with cm_col2:
    st.write("**BiLSTM**")
    cm_b = confusion_matrix(y_test, y_pred_b)
    fig_cm_b = go.Figure(data=go.Heatmap(
        z=cm_b, x=['Aman', 'Berisiko'], y=['Aman', 'Berisiko'],
        text=cm_b, texttemplate="%{text}", colorscale='Blues'
    ))
    fig_cm_b.update_layout(height=300, margin=dict(t=0, b=0))
    st.plotly_chart(fig_cm_b, use_container_width=True)