import streamlit as st
import pandas as pd
import numpy as np
import time

def prediksi_stunting(data_input):
    total_risiko = data_input.sum(axis=1).values[0]
    if total_risiko >= 3:
        return 1 # Berisiko
    else:
        return 0 # Tidak Berisiko

st.markdown("""
    <div style='background-color:#2c3e50; padding:20px; border-radius:10px; margin-bottom:25px;'>
        <h2 style='text-align: center; color: white;'>Klasifikasi Keluarga Rentan Stunting</h2>
    </div>
""", unsafe_allow_html=True)


# ==========================================
# 1. LOAD DATA WILAYAH DARI EXCEL OTOMATIS
# ==========================================
@st.cache_data
def get_data_wilayah():
    df = pd.read_excel("penelitian_bersih.xlsx") 
    return df[['namakecamatan', 'namakelurahan']].dropna()

df_wilayah = get_data_wilayah()

# Ambil semua nama kecamatan yang unik
daftar_kecamatan = ["-- Pilih Kecamatan --"] + sorted(df_wilayah['namakecamatan'].unique().tolist())


# ==========================================
# 2. INPUT WILAYAH (Dinamis dari Excel)
# ==========================================
st.markdown("#### 📍 Data Wilayah")
col_wilayah1, col_wilayah2 = st.columns(2)

with col_wilayah1:
    kecamatan = st.selectbox("Kecamatan", daftar_kecamatan)

with col_wilayah2:
    if kecamatan == "-- Pilih Kecamatan --":
        daftar_kelurahan = ["-- Pilih Kelurahan --"]
    else:
        df_filtered = df_wilayah[df_wilayah['namakecamatan'] == kecamatan]
        daftar_kelurahan = ["-- Pilih Kelurahan --"] + sorted(df_filtered['namakelurahan'].unique().tolist())
        
    kelurahan = st.selectbox("Kelurahan / Desa", daftar_kelurahan)
    
st.markdown("---")


# ==========================================
# 3. FORM INPUT FAKTOR RISIKO
# ==========================================
with st.form("form_klasifikasi"):
    st.markdown("#### 📋 Faktor Risiko")
    col1, col2 = st.columns(2)
        
    with col1:
        baduta = st.radio("Apakah memiliki anak Baduta (0–24 bulan)?", ["Tidak", "Ya"])
        balita = st.radio("Apakah memiliki anak Balita (0–59 bulan)?", ["Tidak", "Ya"])
        pus = st.radio("Apakah termasuk Pasangan Usia Subur (PUS)?", ["Tidak", "Ya"])
        pus_hamil = st.radio("Apakah ada yang sedang hamil?", ["Tidak", "Ya"])
        terlalu_muda = st.radio("Apakah ibu hamil terlalu muda (< 20 tahun)?", ["Tidak", "Ya"])
        terlalu_tua = st.radio("Apakah ibu hamil terlalu tua (> 35 tahun)?", ["Tidak", "Ya"])
            
    with col2:
        jarak_dekat = st.radio("Apakah jarak kelahiran < 2 tahun?", ["Tidak", "Ya"])
        banyak_anak = st.radio("Apakah jumlah anak > 4?", ["Tidak", "Ya"])
        tanpa_kb = st.radio("Apakah tidak menggunakan KB modern?", ["Tidak", "Ya"])
        jamban_tidak_layak = st.radio("Apakah jamban tidak memenuhi standar?", ["Tidak", "Ya"])
            
        sumber_air = st.selectbox("Sumber Air Utama Keluarga", 
                                ["Air kemasan/isi ulang", "Ledeng/PAM", "Sumur bor/pompa", "Sumur terlindung", "Lainnya (Tidak Layak)"])
        kesejahteraan = st.selectbox("Peringkat Kesejahteraan Keluarga", 
                                ["Peringkat Kesejahteraan >4", "Peringkat 1 (Sangat Miskin)", "Peringkat 2 (Miskin)", "Peringkat 3 (Rentan)"])

    submitted = st.form_submit_button("Analisis Risiko Stunting", use_container_width=True)


# ==========================================
# 4. PROSES HASIL
# ==========================================
if submitted:
    if kecamatan == "-- Pilih Kecamatan --" or kelurahan == "-- Pilih Kelurahan --":
        st.warning("⚠️ Mohon pilih Kecamatan dan Kelurahan terlebih dahulu di bagian atas!")
    else:
        with st.spinner('Sedang menganalisis data keluarga...'):
            time.sleep(1)
                
            def convert_yes_no(val): return 1 if val == "Ya" else 0
            air_layak_tidak = 1 if sumber_air == "Lainnya (Tidak Layak)" else 0
            
            data_untuk_model = {
                "baduta": [convert_yes_no(baduta)],
                "balita": [convert_yes_no(balita)],
                "pus": [convert_yes_no(pus)],
                "pus_hamil": [convert_yes_no(pus_hamil)],
                "terlalu_muda": [convert_yes_no(terlalu_muda)],
                "terlalu_tua": [convert_yes_no(terlalu_tua)],
                "jarak_dekat": [convert_yes_no(jarak_dekat)],
                "banyak_anak": [convert_yes_no(banyak_anak)],
                "tanpa_kb": [convert_yes_no(tanpa_kb)],
                "jamban_layak_tidak": [convert_yes_no(jamban_tidak_layak)],
                "sumber_air_layak_tidak": [air_layak_tidak]
            }
            
            df_input = pd.DataFrame(data_untuk_model)
            hasil = prediksi_stunting(df_input)
                
            st.markdown("---")
            st.markdown(f"### Hasil Analisis Wilayah: Kec. {kecamatan}, Kel. {kelurahan}")
            
            if hasil == 0:
                st.success("#### ✅ Tidak Berisiko\nKeluarga ini teridentifikasi **tidak berisiko** stunting.")
            else:
                st.error("#### ⚠️ Berisiko\nKeluarga ini teridentifikasi **berisiko** stunting. Diperlukan intervensi dan pendampingan.")
                
            with st.expander("Lihat Ringkasan Data yang Dimasukkan", expanded=True):
                # --- BAGIAN YANG DIPERBAIKI ---
                # Kita jadikan DataFrame dulu, baru kita insert kolomnya
                df_tampil = pd.DataFrame(data_untuk_model)
                df_tampil.insert(0, "Kelurahan", [kelurahan])
                df_tampil.insert(0, "Kecamatan", [kecamatan])
                # ------------------------------
                
                st.dataframe(df_tampil, use_container_width=True)