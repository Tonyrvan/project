import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import FastMarkerCluster, MarkerCluster

st.set_page_config(
    page_title="Peta Cepat Stunting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Visualisasi Sebaran Cepat")
st.markdown("Peta sebaran lokasi stunting yang dioptimalkan untuk performa.")

@st.cache_data
def load_data():
    try:
        nama_file = "penelitian_bersih.xlsx"
        # Gunakan usecols untuk hanya mengambil kolom penting biar memori ringan
        cols_needed = ['namakecamatan', 'namakelurahan', 'lat', 'lon', 'risiko_stunting', 'Tahun']
        
        df = pd.read_excel(nama_file, engine='openpyxl', usecols=lambda x: x in cols_needed)
        
        # Preprocessing
        df = df.replace({'X': 0, 'V': 1, 'x': 0, 'v': 1})
        df = df.dropna(subset=['lat', 'lon', 'risiko_stunting', 'Tahun'])
        df['Tahun'] = df['Tahun'].astype(int)
        
        return df
    except Exception as e:
        return None

df_raw = load_data()

if df_raw is None:
    st.error("Gagal memuat data. Pastikan file Excel benar.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Filter Peta")
    
    list_tahun = sorted(df_raw['Tahun'].unique())
    pilih_tahun = st.selectbox("Pilih Tahun:", list_tahun)
    
    pilih_status = st.radio("Status Risiko:", ["Semua", "Hanya Berisiko (Merah)", "Hanya Aman (Hijau)"])
    
    list_kecamatan = ["Semua Kecamatan"] + sorted(df_raw['namakecamatan'].unique().tolist())
    pilih_kecamatan = st.selectbox("Pilih Kecamatan:", list_kecamatan)

    st.divider()
    st.info("💡 Tips: Gunakan filter Kecamatan jika peta masih terasa lambat.")


df_filtered = df_raw[df_raw['Tahun'] == pilih_tahun]

if pilih_status == "Hanya Berisiko (Merah)":
    df_filtered = df_filtered[df_filtered['risiko_stunting'] == 1]
elif pilih_status == "Hanya Aman (Hijau)":
    df_filtered = df_filtered[df_filtered['risiko_stunting'] == 0]

if pilih_kecamatan != "Semua Kecamatan":
    df_filtered = df_filtered[df_filtered['namakecamatan'] == pilih_kecamatan]

c1, c2, c3 = st.columns(3)
c1.metric("Total Data Tampil", len(df_filtered))
c2.metric("🔴 Berisiko", len(df_filtered[df_filtered['risiko_stunting']==1]))
c3.metric("🟢 Aman", len(df_filtered[df_filtered['risiko_stunting']==0]))

if not df_filtered.empty:

    avg_lat = df_filtered['lat'].mean()
    avg_lon = df_filtered['lon'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, prefer_canvas=True)
    
    data_merah = df_filtered[df_filtered['risiko_stunting'] == 1]
    data_hijau = df_filtered[df_filtered['risiko_stunting'] == 0]

    if len(df_filtered) > 2000:
        st.caption("ℹ️ Mode Ringan Aktif (Menampilkan Titik Lingkaran karena data banyak)")
        
        # Gambar Titik Merah
        for _, row in data_merah.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=3,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup=f"Berisiko: {row.get('namakelurahan', '-')}"
            ).add_to(m)
            
        # Gambar Titik Hijau
        for _, row in data_hijau.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=3,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.5,
                popup=f"Aman: {row.get('namakelurahan', '-')}"
            ).add_to(m)
            
    # --- OPSI 2: JIKA DATA SEDIKIT (< 2000), PAKAI MARKER CLUSTER BIASA ---
    else:
        marker_cluster = MarkerCluster().add_to(m)
        
        for _, row in df_filtered.iterrows():
            warna = 'red' if row['risiko_stunting'] == 1 else 'green'
            status = "BERISIKO" if row['risiko_stunting'] == 1 else "AMAN"
            
            folium.Marker(
                location=[row['lat'], row['lon']],
                icon=folium.Icon(color=warna, icon="info-sign"),
                popup=f"{status}\n{row.get('namakelurahan', '-')}"
            ).add_to(marker_cluster)

    st_folium(m, height=500, use_container_width=True)

else:
    st.warning("Data tidak ditemukan untuk filter ini.")