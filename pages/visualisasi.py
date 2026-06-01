import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px

st.set_page_config(
    page_title="Map Intelligence Stunting",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #000000;
        color: #ffffff;
    }

    [data-testid="stSidebar"] {
        background-color: #0e1117 !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }

    .hero-title {
        font-size: 40px;
        font-weight: 800;
        letter-spacing: -1.5px;
        background: -webkit-linear-gradient(#fff, #999);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }

    div[data-testid="stMetric"] {
        background: #1c1c1e !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 20px !important;
        border-radius: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_excel("penelitian_bersih.xlsx")
        df.columns = df.columns.str.lower()
        mapping = {'1': 'Berisiko', '0': 'Aman', 'V': 'Berisiko', 'X': 'Aman'}
        if 'risiko_stunting' in df.columns:
            df['risiko_stunting'] = df['risiko_stunting'].astype(str).str.strip().replace(mapping)
        return df
    except:
        return None

df_raw = load_data()

with st.sidebar:
    st.markdown("### ⚙️ Panel Kontrol")
    
    if df_raw is not None:
        kec_list = ['Semua'] + sorted(df_raw['namakecamatan'].dropna().unique().tolist())
        sel_kec = st.selectbox("Wilayah Kecamatan", kec_list)

        # FILTER TAHUN
        if 'tahun' in df_raw.columns:
            tahun_list = ['Semua'] + sorted(df_raw['tahun'].dropna().astype(str).unique().tolist())
            sel_tahun = st.selectbox("Pilih Tahun", tahun_list)
        else:
            sel_tahun = 'Semua'
            st.warning("Kolom 'tahun' tidak ditemukan di data.")

if df_raw is not None:
    df_f = df_raw.copy()

    if sel_kec != 'Semua':
        df_f = df_f[df_f['namakecamatan'] == sel_kec]

    if sel_tahun != 'Semua':
        df_f = df_f[df_f['tahun'].astype(str) == sel_tahun]

    st.markdown('<h1 class="hero-title">Map Intelligence Stunting</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:#86868b; font-size:18px; margin-bottom:30px;">Visualisasi spasial data kerentanan keluarga Kota Bogor'
        + (f' Tahun {sel_tahun}.' if sel_tahun != 'Semua' else '.')
        + '</p>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Observasi", f"{len(df_f):,}")
    m2.metric("Kecamatan", df_f['namakecamatan'].nunique() if not df_f.empty else 0)
    m3.metric("Kelurahan", df_f['namakelurahan'].nunique() if not df_f.empty else 0)
    risiko_pct = (len(df_f[df_f['risiko_stunting'] == 'Berisiko']) / len(df_f)) * 100 if len(df_f) > 0 else 0
    m4.metric("% Kerentanan", f"{risiko_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    if not df_f.empty:
        col_map, col_pie = st.columns([2.2, 0.8])

        with col_map:
            st.markdown("#### 🗺️ Sebaran Geospasial")

            m = folium.Map(
                location=[df_f['lat'].mean(), df_f['lon'].mean()],
                zoom_start=13,
                tiles='OpenStreetMap'
            )

            map_data = df_f.groupby('namakelurahan').agg({
                'lat': 'mean',
                'lon': 'mean',
                'risiko_stunting': lambda x: x.mode()[0],
                'namakecamatan': 'first'
            }).reset_index()

            for _, row in map_data.iterrows():
                pin_color = 'red' if row['risiko_stunting'] == 'Berisiko' else 'green'

                detail_kel = df_f[df_f['namakelurahan'] == row['namakelurahan']]
                jml_aman = len(detail_kel[detail_kel['risiko_stunting'] == 'Aman'])
                jml_berisiko = len(detail_kel[detail_kel['risiko_stunting'] == 'Berisiko'])
                total_data = jml_aman + jml_berisiko

                popup_html = f"""
                <div style="font-family: 'Inter', sans-serif; width: 220px; color: #1e293b; line-height: 1.6;">
                    <div style="margin-bottom: 8px;">
                        <span style="font-size: 13px;">📍 <b style="color: #0A84FF;">Kelurahan:</b> {row['namakelurahan']}</span><br>
                        <span style="font-size: 13px;">🏡 <b style="color: #0A84FF;">Kecamatan:</b> {row['namakecamatan']}</span><br>
                        <span style="font-size: 13px;">📊 <b style="color: #ff4b4b;">Status Dominan:</b> {row['risiko_stunting']}</span>
                    </div>
                    <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0;">
                        <span style="font-size: 12px; font-weight: bold; color: #64748b;">📈 Distribusi Data:</span><br>
                        <span style="font-size: 12px;">✅ Tidak Berisiko: <b>{jml_aman}</b></span><br>
                        <span style="font-size: 12px;">⚠️ Berisiko: <b>{jml_berisiko}</b></span><br>
                        <hr style="border: 0.5px solid #cbd5e1; margin: 4px 0;">
                        <span style="font-size: 12px;">📊 <b style="color: #7c3aed;">Total: {total_data}</b></span>
                    </div>
                </div>
                """

                folium.Marker(
                    location=[row['lat'], row['lon']],
                    icon=folium.Icon(color=pin_color, icon='info-sign'),
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(m)

            st_folium(m, height=450, use_container_width=True, key="main_map")

        with col_pie:
            st.markdown("#### 🍩 Rasio Risiko")
            risk_dist = df_f['risiko_stunting'].value_counts()
            fig_pie = px.pie(
                risk_dist,
                values=risk_dist.values,
                names=risk_dist.index,
                hole=0.6,
                color_discrete_sequence=['#10b981', '#ef4444']
            )
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()
        st.markdown("#### 📊 Analisis Detail Wilayah")
        kec_data = df_f.groupby(['namakecamatan', 'risiko_stunting']).size().unstack(fill_value=0).reset_index()

        for col in ['Aman', 'Berisiko']:
            if col not in kec_data.columns:
                kec_data[col] = 0

        fig_bar = px.bar(
            kec_data,
            y='namakecamatan',
            x=['Aman', 'Berisiko'],
            orientation='h',
            barmode='group',
            color_discrete_map={'Aman': '#10b981', 'Berisiko': '#ef4444'}
        )
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.warning("Data tidak tersedia untuk filter yang dipilih.")

else:
    st.error("File 'penelitian_bersih.xlsx' tidak ditemukan.")