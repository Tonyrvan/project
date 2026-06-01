import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Dashboard | Stunting", 
    page_icon="🏠", 
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Section */
    .hero-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -1px;
        background: -webkit-linear-gradient(#fff, #999);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    /* Info Box Metode AI */
    .method-box {
        background: rgba(10, 132, 255, 0.05);
        border: 1px solid rgba(10, 132, 255, 0.2);
        border-radius: 15px;
        padding: 15px;
        margin-top: 15px;
    }

    /* Metric Card - Compact */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px 15px !important;
        border-radius: 15px !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 24px !important;
    }

    /* Dataframe Styling */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_raw_dataset():
    try:

        df = pd.read_excel("penelitian_bersih.xlsx", engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")
        return None

df_tampil = load_raw_dataset()

col_header, col_logo = st.columns([4, 1])

with col_header:
    st.markdown("""
        <div style="margin-bottom: 10px;">
            <span style="background: rgba(10, 132, 255, 0.1); color: #0A84FF; padding: 4px 12px; border-radius: 100px; font-size: 12px; font-weight: 600; border: 1px solid rgba(10, 132, 255, 0.2);">
                Ilmu Komputer
            </span>
            <h1 class="hero-title">Selamat Datang di Aplikasi Klasifikasi dan Visualisasi Keluarga Berisiko Stunting</h1>
            <p style="color: #86868b; margin-top: 5px; font-size: 15px;">
                Aplikasi ini merupakan implementasi dari penelitian skripsi
“Klasifikasi dan Visualisasi Keluarga Berisiko Stunting dengan Komparasi Model Stacked LSTM dan BiLSTM.”
                Aplikasi ini menampilkan hasil klasifikasi dan visualisasi data keluarga berisiko stunting
            </p>
        </div>
    """, unsafe_allow_html=True)

with col_logo:
    st.markdown('<div style="text-align: right; margin-top: 10px;"><img src="https://img.icons8.com/fluency/96/artificial-intelligence.png" width="60"/></div>', unsafe_allow_html=True)


col_desc, col_ai = st.columns([2, 1])
with col_desc:
    st.write("Dashboard ini menampilkan data awal penelitian yang digunakan untuk melatih model klasifikasi risiko stunting. Data mencakup informasi geografis, kondisi kesehatan, dan faktor lingkungan.")

with col_ai:
    st.markdown("""
        <div class="method-box">
            <p style="color: #0A84FF; font-size: 13px; font-weight: 700; margin-bottom: 5px;">🧠 METODE:</p>
            <li style="font-size: 13px; color: #f5f5f7;"><b>BiLSTM</b> (Bidirectional LSTM)</li>
            <li style="font-size: 13px; color: #f5f5f7;"><b>Stacked LSTM</b> (Deep Layers)</li>
        </div>
    """, unsafe_allow_html=True)

st.divider()

if df_tampil is not None:

    st.markdown("### 📊 Statistik Dataset")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1: st.metric("Total Data", f"{df_tampil.shape[0]}")
    with c2: st.metric("Jumlah Fitur", f"{df_tampil.shape[1]} Kolom")
    
    target_col = 'risiko_stunting' 
    if target_col in df_tampil.columns:
        risiko_count = df_tampil[df_tampil[target_col].isin([1, 'V', 'v'])].shape[0]
        aman_count = df_tampil[df_tampil[target_col].isin([0, 'X', 'x'])].shape[0]
        with c3: st.metric("🔴 Berisiko", risiko_count)
        with c4: st.metric("🟢 Aman", aman_count)
    else:
        with c3: st.metric("Status Data", "Raw/Mentah")
        with c4: st.metric("Tipe File", "Excel (.xlsx)")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🔍 Eksplorasi Data Interaktif")
    st.dataframe(df_tampil, use_container_width=True, height=500)

    with st.expander("ℹ️ Keterangan Dataset"):
        st.markdown("""
        * **Data Mentah**: Tabel di atas menampilkan data asli sebelum dilakukan normalisasi atau transformasi data.
        * **Fitur**: Mencakup data kecamatan, kelurahan, status balita (baduta/balita), status PUS, dan kondisi sarana sanitasi (sumber air, jamban).
        * **Tujuan**: Data ini digunakan sebagai input bagi model **BiLSTM** dan **Stacked LSTM** untuk mengukur tingkat risiko stunting secara otomatis.
        """)
else:
    st.error("⚠️ File 'penelitian_bersih.xlsx' tidak ditemukan di direktori aplikasi.")