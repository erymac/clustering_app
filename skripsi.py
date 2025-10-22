from fungsi import (
    proses_clustering, validate_columns_and_data, preprocess_data, columns_to_drop, data_selection, BKMeans, AHC,
    normalize, proses_clustering_perbandingan
    )
from utils import show_navbar, hide_sidebar, show_footer
import re
import csv
import os
import io
import streamlit as st
import pandas as pd

st.set_page_config(
    layout="wide",
    page_title="Clustering Data Kacang Hijau",
    page_icon="app/images/kacang_hijau_icon.png"
)
hide_sidebar()

with open("app/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" crossorigin="anonymous">', unsafe_allow_html=True)

# Get page from URL query params
query_params = st.query_params
page = query_params.get("page", "home")  

show_navbar()

#Home Page
st.markdown("")

st.markdown('<h1 class="custom-header" style="font-size:47px; align: center; color: black; margin-bottom: 36px; font-family: Inter;">Selamat Datang di Situs Clustering Data Kacang Hijau</h1>',
            unsafe_allow_html=True)

data_path = 'https://bdsp2.pertanian.go.id/bdsp/id/lokasi'
data_sample = 'data sample/Kacang Hijau.csv'

col = st.columns(4, gap="large", vertical_alignment="top")
with col[1]:
    st.link_button(
        "Sumber Data",
        "https://bdsp2.pertanian.go.id/bdsp/id/lokasi",
        use_container_width=True
    )
with col[2]:
    st.download_button(
        label="Contoh Dataset",
        data=open(f'{data_sample}'),
        file_name='Sampel Kacang Hijau 2010-2024.csv',
        use_container_width=True
    )

if 'instruction_shown' not in st.session_state:
    st.session_state.instruction_shown = False

@st.dialog("Cara Kerja", width="large")
def instruction():
    if not st.session_state.instruction_shown:
        st.write("""
        **Sumber data** dapat diunduh dari [BDSP](https://bdsp2.pertanian.go.id/bdsp/id/lokasi) atau menggunakan contoh dataset yang diunduh melalui tombol "Contoh Dataset".
        
        Berikut cara menggunakan situs clustering data kacang hijau :
        1. **Unggah Data**: Klik "Browse files" dan mengunggah dataset berbentuk excel (.csv / .xlsx).
        2. **Pilih Rentang Tahun**: Pilih rentang tahun yang ingin digunakan untuk proses clustering.
        2. **Pilih Algoritma dan Parameter**: Pilih algoritma dan jumlah cluster yang ingin diterapkan pada dataset Anda.
        3. **Mulai Clustering**: Dataset yang diunggah dapat diproses setelah pengguna memencet tombol "Mulai Clustering".
        4. **Lihat Hasil**: Setelah proses clustering selesai, hasil pengelompokan akan ditampilkan beserta metrik evaluasi.

        Jenis linkage Agglomerative Hierarchical Clustering :
        - Ward adalah metode yang meminimalkan variansi total dalam cluster.
        - Complete adalah metode yang meminimalkan jarak maksimum antara titik dalam cluster.
        - Average adalah metode yang meminimalkan jarak rata-rata antara titik dalam cluster.
        - Single adalah metode yang meminimalkan jarak minimum antara titik dalam cluster.

        """)
        st.session_state.instruction_shown = True

if not st.session_state.instruction_shown:
    instruction()

uploaded_file = st.file_uploader("Unggah file dataset dalam excel (.csv / .xlsx)")
dataframe_mentah = None
if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[-1]
    if file_ext == '.csv':
        file_contents = uploaded_file.getvalue()
        dialect = csv.Sniffer().sniff(file_contents.decode())
        if dialect.delimiter == ',':
            dataframe_mentah = pd.read_csv(io.StringIO(file_contents.decode()))
        elif dialect.delimiter == ';':
            dataframe_mentah = pd.read_csv(io.StringIO(file_contents.decode()), sep=';')
    elif file_ext == '.xlsx':
        dataframe_mentah = pd.read_excel(uploaded_file, engine='openpyxl')
    elif file_ext == '.xls':
        dataframe_mentah = pd.read_excel(uploaded_file, engine='xlrd')
    else:
        dataframe_mentah = None
        st.error("Jenis file tidak didukung. Harap unggah file Excel (.csv / .xls / .xlsx).")

if dataframe_mentah is not None:
    year_pattern = r' (\d{4})$'
    metric_cols = [col for col in dataframe_mentah.columns 
                    if re.search(year_pattern, col)]
    tahun = sorted([int(re.search(year_pattern, col).group(1)) for col in metric_cols])

    cluster_value = st.slider(
        "Pilih rentang tahun data", 
        min_value = min(tahun), max_value=max(tahun), step=1, value=(min(tahun), max(tahun)),
        disabled = uploaded_file is None
    )

    dataframe_mentah = dataframe_mentah[[col for col in dataframe_mentah.columns if not re.search(year_pattern, col) or (re.search(year_pattern, col) and int(re.search(year_pattern, col).group(1)) >= cluster_value[0] and int(re.search(year_pattern, col).group(1)) <= cluster_value[1])]]
    st.dataframe(dataframe_mentah, hide_index=True, height=250)

cols = st.columns(2, gap="large", vertical_alignment="top")
with cols[0]:
    metode = st.multiselect(
        "Pilih metode clustering",
        options=["Bisecting K-Means", "Agglomerative Hierarchical Clustering"],
        default=["Agglomerative Hierarchical Clustering"],
        help="Pilih satu atau dua metode clustering untuk dibandingkan."
    )
    enabled = metode != ""
    if "Agglomerative Hierarchical Clustering" in metode:
        linkage = st.selectbox(
            "Pilih jenis linkage untuk Agglomerative Hierarchical Clustering",
            options=["ward", "complete", "average", "single"],
            index=0, width="stretch",
            help="Linkage menentukan cara pengukuran jarak antar cluster.",
        )
    
with cols[1]:
    cluster_option = st.radio(
        "Pilih jumlah cluster atau dengan rentang",
        options=["Jumlah cluster", "Rentang cluster"],
        horizontal=True,
        help="Pilih jumlah cluster data atau beberapa rentang jumlah cluster untuk dievaluasi."
    )

    if cluster_option == "Jumlah cluster": # Pilih angka jumlah cluster
        cluster_value = st.slider(
            "Pilih berapa banyak data ingin dikelompokkan", 
            min_value=2, max_value=5, step=1, value=3,
            help="Pilih jumlah cluster data yang diinginkan."
        )
        n_cluster = range(cluster_value, cluster_value+1)
    else:
        cluster_range = st.slider( # pilih rentang jumlah cluster
            "Rentang cluster", 
            min_value=2, max_value=5, step=1, value=(2, 5),
            help="Pilih rentang jumlah cluster data yang diinginkan."
        )
        min_cluster, max_cluster = cluster_range
        n_cluster = range(min_cluster, max_cluster + 1)

# TOMBOL MULAI CLUSTERING
st.markdown("<br>", unsafe_allow_html=True)
cols = st.columns(5, gap="large", vertical_alignment="top")
with cols[2]:
    start = st.button("Mulai Clustering")
# PROSES CLUSTERING
st.markdown("<br>", unsafe_allow_html=True)
if start and uploaded_file and len(metode) > 0:
    try:
        validate_columns_and_data(dataframe_mentah)
        # st.success(":green[:material/done:] Data berhasil divalidasi dan diproses.")
        st.badge("Data berhasil divalidasi dan diproses.", icon=":material/check:", color="green")
        df_copy = preprocess_data(dataframe_mentah)
        nama_lokasi_awal = df_copy['Lokasi'].to_list()
        df_copy = normalize(df_copy)

        df_array = df_copy.drop(['Lokasi'], axis=1)
        df_array = df_array.values

        df_temp = columns_to_drop(dataframe_mentah) # temp untuk menampilkan tabel dataframe asli
        df_temp = data_selection (df_temp)
        df_temp = df_temp.reset_index()
        
        with st.spinner("Clustering data..."):
            if "Bisecting K-Means" in metode and len(metode) < 2: # BISECTING K-MEANS ONLY
                metode = "Bisecting K-Means"
                df_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, _, bestcluster_bkmeans, labels_bkmeans = BKMeans(df_array, n_cluster)

                proses_clustering(df_copy, metode, labels_bkmeans, bestcluster_bkmeans, cluster_option,
                                df_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, df_temp)

            if "Agglomerative Hierarchical Clustering" in metode and len(metode) < 2: # AGGLOMERATIVE HIERARCHICAL CLUSTERING ONLY
                metode = "Agglomerative Clustering"
                linkage = linkage.lower()
                df_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, _, bestcluster_ahc, labels_ahc = AHC(df_array, n_cluster, linkage)

                proses_clustering(df_copy, metode, labels_ahc, bestcluster_ahc, cluster_option,
                                df_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, df_temp)

            # KETIKA MEMILIH 2 METODE AKAN DIBANDINGKAN
            elif len(metode) == 2:
                proses_clustering_perbandingan(linkage, df_copy, df_temp, df_array, n_cluster, cluster_option)

    except ValueError as e:
        st.error(f"Terjadi kesalahan: {e}")

else:
    if start and uploaded_file is None:
        st.error("⚠️ Harap unggah file dataset terlebih dahulu.")
        # st.badge("⚠️ Harap unggah file dataset terlebih dahulu.", color="orange")
    elif start and len(metode) == 0:
        st.error("⚠️ Harap setidaknya pilih salah satu metode clustering.")
    
show_footer()