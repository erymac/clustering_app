from fungsi import (
    validate_columns_and_data, heatmap_corr, penyesuaian, pie_kontribusi, validate_columns_and_data, plot_tren_panen,
    avg_features, add_provinsi_to_df, preprocess_data, show_prod_dan_lp, columns_to_drop, data_selection
    )
import streamlit as st
import csv
import io
import re
import os
import pandas as pd
from utils import show_navbar, hide_sidebar, show_footer

st.set_page_config(layout="wide", page_title="About")
hide_sidebar()

with open( "app\style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" crossorigin="anonymous">', unsafe_allow_html=True)

# Get page from URL query params
query_params = st.query_params
page = query_params.get("page", "home")  # Default to 'Home' if none

show_navbar()

st.markdown("")

st.markdown('''<h1 class="custom-header" style="font-size:47px; align: center; color: black; margin-bottom: 26px; font-family: Inter;">
            Analisis Data
            </h1>''',
            unsafe_allow_html=True)

st.write('''Halaman ini digunakan untuk menganalisis dataset hasil panen kacang hijau. Analisis mencakup korelasi antar fitur, 
         tren panen dari tahun ke tahun, serta visualisasi kontribusi produksi dan luas panen berdasarkan provinsi.''')

uploaded_file = st.file_uploader("Unggah dataset berbentuk excel (.csv / .xlsx).")
if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
    if file_ext == '.csv':
        file_contents = uploaded_file.getvalue()
        dialect = csv.Sniffer().sniff(file_contents.decode())
        if dialect.delimiter == ',':
            dataframe_mentah = pd.read_csv(io.StringIO(file_contents.decode()))
        elif dialect.delimiter == ';':
            dataframe_mentah = pd.read_csv(io.StringIO(file_contents.decode()), sep=';')
        # dataframe_mentah = pd.read_csv(uploaded_file, sep=';')
    elif file_ext == '.xlsx':
        dataframe_mentah = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        dataframe_mentah = None
        st.error("Jenis file tidak didukung. Harap unggah file Excel (.csv / .xlsx).")

    st.dataframe(dataframe_mentah, hide_index=True, height=300)

if uploaded_file is not None:
    try:
        validate_columns_and_data(dataframe_mentah)
        st.success(":green[:material/done:] Data berhasil divalidasi dan diproses.")
        df_copy = preprocess_data(dataframe_mentah)
        df_array = df_copy.drop(['Lokasi'], axis=1)
        df_array = df_array.values

        with st.spinner("Memproses data..."):
            df_copy = avg_features(df_copy)
            dataframe_mentah = avg_features(dataframe_mentah)
            cols = st.columns(2, gap="small", vertical_alignment="top")
            with cols[0]:
                heatmap_corr(df_copy[['Luas Panen', 'Produksi', 'Produktivitas']])
            with cols[1]:
                st.write("##### Data Rata-rata Fitur per Lokasi")
                nama_lokasi_awal = dataframe_mentah['Lokasi'].to_list()
                # dataframe_mentah = columns_to_drop (dataframe_mentah)
                # dataframe_mentah = data_selection(dataframe_mentah)
                # dataframe_mentah = dataframe_mentah.reset_index() # Reset index 'Lokasi'
                dataframe_mentah = penyesuaian(dataframe_mentah)
                dataframe_mentah = add_provinsi_to_df(dataframe_mentah)
                # drop_lokasi = ['Pesisir Barat', 'Mempawah', 'Kab. Banjar', 'Buton Tengah', 'Pegunungan Arfak']
                # # nama_lokasi_awal = [item for item in drop_lokasi if item in nama_lokasi_awal]
                # [x for x in sents if not x.startswith('@$\t') and not x.startswith('#')]
                # # dataframe_mentah['Lokasi'] = nama_lokasi_awal
                # st.write(len(nama_lokasi_awal))
                st.dataframe(dataframe_mentah[['Provinsi', 'Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas']], hide_index=True)

            cols = st.columns(2, gap="small", vertical_alignment="top")
            with cols[0]:
                year_pattern = r' (\d{4})$'
                metric_cols = [col for col in df_copy.columns 
                            if re.search(year_pattern, col)]
                
                tahun = sorted([int(re.search(year_pattern, col).group(1)) for col in metric_cols])
                df_copy = penyesuaian(df_copy)
                df_copy = add_provinsi_to_df(df_copy)
                pie_kontribusi(df_copy, min(tahun), max(tahun))
            with cols[1]:
                bar_chart = show_prod_dan_lp(df_copy.drop(['Luas Panen', 'Produksi', 'Produktivitas'], axis=1))
                st.plotly_chart(bar_chart)

            # st.dataframe(df_copy.drop(['Luas Panen', 'Produksi', 'Produktivitas'], axis=1))
            # df_copy = df_copy.drop(['Luas Panen', 'Produksi', 'Produktivitas'], axis=1)
            plot_tren_panen(df_copy)

    except ValueError as e:
        st.error(f"Terjadi kesalahan: {e}")

show_footer()