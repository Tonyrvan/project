import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Sistem Deteksi Stunting",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_dataset():
    try:
        # Load Data Excel
        df = pd.read_excel("penelitian_bersih.xlsx", engine='openpyxl')
        df = df.replace({'X': 0, 'V': 1, 'x': 0, 'v': 1})
        return df
    except Exception as e:
        return None

df_tampil = load_dataset()

st.title("🎓 Sistem Deteksi Dini Stunting")
st.markdown("### Penerapan Deep Learning untuk Analisis Risiko Stunting")

col_intro, col_metode = st.columns([2, 1])

with col_intro:
    st.write("""
    Selamat datang di dashboard hasil penelitian skripsi. Sistem ini dirancang untuk memprediksi dan memetakan 
    risiko stunting pada balita berdasarkan data spasial dan faktor lingkungan di wilayah penelitian.
    
    **Fitur Aplikasi:**
    * **Visualisasi:** Melihat peta sebaran risiko stunting.
    * **Perbandingan:** Mengevaluasi kinerja model BiLSTM vs Stacked LSTM.
    """)

with col_metode:
    st.info("""
    **Metode AI:**
    - 🧠 **BiLSTM** (Bidirectional LSTM)
    - 📚 **Stacked LSTM** (Layer Bertumpuk)
    """)

st.divider()

# --- Tinjauan Dataset ---
if df_tampil is not None:
    st.subheader("📂 Tinjauan Dataset")
    st.write("Berikut adalah sampel data yang digunakan dalam penelitian ini:")

    # Info Statistik Data
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data", f"{df_tampil.shape[0]} Baris")
    c2.metric("Jumlah Fitur", f"{df_tampil.shape[1]} Kolom")
    
    # Cek kolom risiko untuk statistik
    if 'risiko_stunting' in df_tampil.columns:
        jml_risiko = len(df_tampil[df_tampil['risiko_stunting']==1])
        jml_aman = len(df_tampil[df_tampil['risiko_stunting']==0])
        c3.metric("🔴 Berisiko", jml_risiko)
        c4.metric("🟢 Aman", jml_aman)

    # Tampilkan Tabel Data (Interactive)
    st.dataframe(df_tampil, use_container_width=True, height=400)

    # Penjelasan Kolom
    with st.expander("ℹ️ Keterangan Label Data"):
        st.write("""
        - **risiko_stunting**: Target prediksi (1 = Berisiko, 0 = Aman).
        - **lat / lon**: Koordinat geografis lokasi.
        - **Tahun**: Tahun pengambilan data.
        - **Fitur Lainnya**: Variabel lingkungan yang mempengaruhi stunting.
        """)
else:
    st.error("⚠️ File 'penelitian_bersih.xlsx' tidak ditemukan di folder ini.")