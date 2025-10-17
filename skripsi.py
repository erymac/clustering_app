from fungsi import (
    proses_clustering, show_map_explanation, validate_columns_and_data, compare_cluster, cluster_and_category_result,
    show_n_cluster, show_map, avg_features, evaluate, preprocess_data, columns_to_drop, data_selection, BKMeans, AHC,
    linechart_evaluation, compare, visualize_silhouette, normalize
    )
from utils import show_navbar, hide_sidebar, show_footer
import csv
import os
import io
import streamlit as st
import pandas as pd

st.set_page_config(
    layout="wide",
    page_title="Clustering Data Kacang Hijau",
    page_icon=":white[:material/home:]"
)
hide_sidebar()

with open( "app\style.css" ) as css:
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
data_sample = 'data sample\Kacang Hijau.csv'

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
        1. **Unggah Data**: Mulai dengan klik "Browse files" dan mengunggah dataset berbentuk excel (.csv / .xlsx).
        2. **Pilih Algoritma dan Parameter**: Pilih algoritma dan jumlah cluster yang ingin diterapkan pada dataset Anda.
        3. **Mulai Clustering**: Dataset yang diunggah akan langsung diproses dan mengeluarkan hasil clustering.

        Sumber data dapat diunduh dari [BDSP](https://bdsp2.pertanian.go.id/bdsp/id/lokasi) atau menggunakan contoh dataset yang diunduh melalui tombol "Contoh Dataset".
        """)
        st.session_state.instruction_shown = True

if not st.session_state.instruction_shown:
    instruction()

uploaded_file = st.file_uploader("Unggah file dataset dalam excel (.csv / .xlsx)")
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

    st.dataframe(dataframe_mentah, hide_index=True, height=300)

cols = st.columns(2, gap="large", vertical_alignment="top")
with cols[0]:
    metode = st.multiselect(
        "Pilih metode clustering",
        options=["Bisecting K-Means", "Agglomerative Hierarchical Clustering"],
        default=["Agglomerative Hierarchical Clustering"]  # Default selection can be K-Means
    )
    enabled = metode != ""
    if "Agglomerative Hierarchical Clustering" in metode:
        linkage = st.selectbox(
            "Pilih jenis linkage untuk Agglomerative Hierarchical Clustering",
            options=["ward", "complete", "average", "single"],
            index=0, width="stretch"
        )
    
with cols[1]:
    cluster_option = st.radio(
        "Pilih jumlah cluster atau dengan rentang",
        options=["Jumlah cluster", "Rentang cluster"],
        horizontal=True)

    if cluster_option == "Jumlah cluster": # Pilih angka jumlah cluster
        cluster_value = st.slider(
            "Pilih banyak cluster", 
            min_value=2, max_value=5, step=1, value=3
        )
        n_cluster = range(cluster_value, cluster_value+1)
    else:
        cluster_range = st.slider( # pilih rentang jumlah cluster
            "Rentang cluster", 
            min_value=2, max_value=5, step=1, value=(2, 5)
        )
        min_cluster, max_cluster = cluster_range
        n_cluster = range(min_cluster, max_cluster + 1)

# PROSES CLUSTERING
st.markdown("<br>", unsafe_allow_html=True)
if uploaded_file is not None:
    try:
        validate_columns_and_data(dataframe_mentah)
        st.success(":green[:material/done:] Data berhasil divalidasi dan diproses.")
        df_copy = preprocess_data(dataframe_mentah)
        df_copy = normalize(df_copy)
        st.dataframe(df_copy, hide_index=True, height=300)
        df_array = df_copy.drop(['Lokasi'], axis=1)
        df_array = df_array.values

        df_temp = columns_to_drop(dataframe_mentah)
        df_temp = data_selection (df_temp)
        df_temp = df_temp.reset_index()
        
        with st.spinner("Clustering data..."):
            if "Bisecting K-Means" in metode and len(metode) < 2: # BISECTING K-MEANS ONLY
                metode = "Bisecting K-Means"
                df_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, _, bestcluster_bkmeans, labels_bkmeans = BKMeans(df_array, n_cluster)

                proses_clustering(df_copy, metode, labels_bkmeans, bestcluster_bkmeans, cluster_option,
                                df_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans)

            if "Agglomerative Hierarchical Clustering" in metode and len(metode) < 2: # AGGLOMERATIVE HIERARCHICAL CLUSTERING ONLY
                metode = "Agglomerative Clustering"
                linkage = linkage.lower()
                df_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, _, bestcluster_ahc, labels_ahc = AHC(df_array, n_cluster, linkage)

                proses_clustering(df_copy, metode, labels_ahc, bestcluster_ahc, cluster_option,
                                df_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc)

            # KETIKA MEMILIH 2 METODE AKAN DIBANDINGKAN
            elif len(metode) == 2:
                linkage = linkage.lower()
                df_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, avg_silhouette_bkmeans, bestcluster_bkmeans, labels_bkmeans = BKMeans(df_array, n_cluster)
                df_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, avg_silhouette_ahc, bestcluster_ahc, labels_ahc = AHC(df_array, n_cluster, linkage)
                df_bkmeans['Metode'] = 'Bisecting K-Means'
                df_ahc['Metode'] = 'AHC'

                # Menyimpan nama metode
                metode1 = 'Bisecting K-Means'
                metode2 = 'Agglomerative Clustering'

                # JIKA MEMILIH RENTANG CLUSTER
                result = compare(silhouette_bkmeans, silhouette_ahc, dbi_bkmeans, dbi_ahc, avg_silhouette_bkmeans, avg_silhouette_ahc, bestcluster_bkmeans, bestcluster_ahc)

                # PERBANDINGAN ALGORITMA CLUSTERING
                st.subheader("Evaluasi Model Clustering", divider=True, anchor="evaluasi_model")
                subcol = st.columns([13,13], border=True, gap="medium")
                with subcol[0]:
                    if result == 'BKMeans':
                        st.write("#### Bisecting K-Means :green[:material/done:]")
                    else:
                        st.write("#### Bisecting K-Means")
                    st.write("")
                    evaluate(bestcluster_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, cluster_option)

                with subcol[1]:
                    if result == 'AHC':
                        st.write("#### Agglomerative Hierarchical Clustering :green[:material/done:]")
                    else:
                        st.write("#### Agglomerative Hierarchical Clustering")
                    st.write("")
                    evaluate(bestcluster_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, cluster_option)

                df_copy = cluster_and_category_result(df_copy, labels_bkmeans, bestcluster_bkmeans, 'Kategori (Bisecting K-Means)', 'Cluster BKM')
                df_copy = cluster_and_category_result(df_copy, labels_ahc, bestcluster_ahc, 'Kategori (Agglomerative Clustering)', 'Cluster AHC')
                df_copy = avg_features(df_copy)

                if cluster_option == "Rentang cluster":
                    # SILHOUETTE DAN DBI LINE CHART
                    fig_silhouette, fig_dbi = linechart_evaluation (df_bkmeans, df_ahc)
                    subcol = st.columns([13,13], border=True, gap="medium")
                    with subcol[0]:
                        st.plotly_chart(fig_silhouette, use_container_width=True)
                    with subcol[1]:
                        st.plotly_chart(fig_dbi, use_container_width=True)

                subcol = st.columns([13,13], gap="medium", vertical_alignment='top')
                with subcol[0]:
                    visualize_silhouette(df_array, df_copy['Cluster BKM'], bestcluster_bkmeans, silhouette_bkmeans, metode1)
                
                with subcol[1]:
                    visualize_silhouette(df_array, df_copy['Cluster AHC'], bestcluster_ahc, silhouette_ahc, metode2)

                # PERBANDINGAN FITUR SETIAP CLUSTER
                st.subheader("Perbandingan Fitur Tiap Cluster", divider=True, anchor="perbandingan_cluster")
                subsubcol = st.columns(2, gap="medium")
                with subsubcol[0]:
                    st.write("##### Bisecting K-Means")
                    compare_cluster(df_copy, 'Cluster BKM')
                with subsubcol[1]:
                    st.write("##### Agglomerative Hierarchical Clustering")
                    compare_cluster(df_copy, 'Cluster AHC')

                # HASIL CLUSTERING ALGORITMA BISECTING K-MEANS DAN AGGLOMERATIVE HIERARCHICAL CLUSTERING
                st.subheader("Hasil Clustering", divider=True, anchor="hasil_clustering")
                st.write("##### Tabel Kategori Hasil Clustering")
                df_temp = cluster_and_category_result(df_temp, labels_bkmeans, bestcluster_bkmeans, 'Kategori (Bisecting K-Means)', 'Cluster BKM')
                df_temp = cluster_and_category_result(df_temp, labels_ahc, bestcluster_ahc, 'Kategori (Agglomerative Clustering)', 'Cluster AHC')
                df_temp = avg_features(df_temp)
                df_temp = df_temp.drop(columns=df_temp.filter(regex='20', axis=1).columns)
                st.dataframe(df_temp[['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas', 'Cluster BKM',
                                        'Kategori (Bisecting K-Means)', 'Cluster AHC', 'Kategori (Agglomerative Clustering)']], hide_index=True)

                # PEMETAAN HASIL CLUSTERING
                subsubcol = st.columns(2, gap="medium")
                st.write("##### Pemetaan Cluster Berdasarkan Tingkat Produksi")
                with subsubcol[0]:
                    st.write(f"##### {metode1}")
                with subsubcol[1]:
                    st.write(f"##### {metode2}")
                subsubcol = st.columns(2, gap="medium")
                with subsubcol[0]:
                    show_map(df_copy, labels_bkmeans, bestcluster_bkmeans)
                with subsubcol[1]:
                    show_map(df_copy, labels_ahc, bestcluster_ahc)
                show_map_explanation()

                # PIECHART JUMLAH ANGGOTA CLUSTER
                subsubcol = st.columns(2, gap="medium")
                with subsubcol[0]:
                    show_n_cluster(df_temp, 'Cluster BKM', metode1)
                with subsubcol[1]:
                    show_n_cluster(df_temp, 'Cluster AHC', metode2)

    except ValueError as e:
        st.error(f"Terjadi kesalahan: {e}")
    

# else:
#     df = pd.read_csv('data sample\Kacang Hijau.csv', sep=';')
#     clustering_sample(df)

show_footer()