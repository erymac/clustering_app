from fungsi import validate_columns_and_data, plot_panen_trends, plot_data_cluster, compare_cluster, cluster_and_category_result, show_n_cluster, show_map, avg_features, evaluate, preprocess_data, columns_to_drop, data_selection, BKMeans, AHC, fig_evaluate, linechart_evaluation, compare, visualize_data, visualize_silhouette, penyesuaian, map_folium, sort_cluster, generate_cluster_category, merge_gdf, clustering_results_dataframe, show_prod_dan_lp, clustering_sample
from utils import show_navbar, hide_sidebar, show_footer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
import folium
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import date
from io import StringIO
import os
import plotly.express as px
import plotly.graph_objects as go
import re

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Clustering Data Kacang Hijau")
hide_sidebar()

with open( "app\style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" crossorigin="anonymous">', unsafe_allow_html=True)

# Get page from URL query params
query_params = st.query_params
page = query_params.get("page", "home")  # Default to 'Home' if none

# st.markdown("""
# <nav class='navbar fixed-top navbar-dark' style='background-color: #984216;'>
#     <div class='container'>
#         <span class='nav-title'>Clustering Data Kacang Hijau</span>
#         <div class='nav-text navbar-nav'>
#             <ul class='nav justify-content-end '>
#                 <li class='nav-item'>
#                     <a class='nav-link' href='/'>Home</a>
#                 </li>
#                 <li class='nav-item'>
#                     <a class='nav-link' href='/analyze'>Analisis Data</a>
#                 </li>
#                 <li class='nav-item'>
#                     <a class='nav-link' href='/about'>Tentang</a>
#                 </li>
#                 <li class='nav-item'>
#                     <a class='nav-link' href='/profile'>Profil</a>
#                 </li>
#             </ul>
#         </div>
#     </div>
# </nav>
# """, unsafe_allow_html=True)

# st.markdown("""
# <nav class='navbar fixed-top navbar-dark' style='background-color: #183a1d;'>
#     <div class='container'>
#         <span class='nav-title' style='color: #f0e0b1;'>Clustering Data Kacang Hijau</span>
#         <div class='nav-text navbar-nav'>
#             <ul class='nav justify-content-end'>
#                 <li class='nav-item'>
#                     <a class='nav-link' href='/'>Home</a>
#                 </li>
#                 <li class='nav-item'>
#                     <a class='nav-link' href='/analyze'>Analisis Data</a>
#                 </li>
#                 <li class='nav-item'>
#                     <a class='nav-link' href='/about'>Tentang</a>
#                 </li>
#                 <li class='nav-item'>
#                     <a class='nav-link' href='/profile'>Profil</a>
#                 </li>
#             </ul>
#         </div>
#     </div>
# </nav>
# """, unsafe_allow_html=True)

# # Add a minimal hover effect with light underline and subtle pale green
# st.markdown("""
#     <style>
#         .navbar-dark .navbar-nav .nav-link {
#             color: #f0e0b1;  /* Off-white text color */
#             transition: color 0.3s ease-in-out, border-bottom 0.3s ease-in-out;
#         }
        
#         .navbar-dark .navbar-nav .nav-link:hover {
#             color: #ff6f61;  /* Bright orange on hover */
#         }
#     </style>
# """, unsafe_allow_html=True)

show_navbar()

# home ()

#Home Page
st.markdown("")

st.markdown('<h1 class="custom-header" style="font-size:48px; align: center; color: black; margin-bottom: 36px; font-family: Inter;">Selamat Datang di Situs Clustering Data Kacang Hijau</h1>',
            unsafe_allow_html=True)

data_path = 'https://bdsp2.pertanian.go.id/bdsp/id/lokasi'
data_sample = 'data sample\Kacang Hijau.csv'

@st.dialog("Contoh struktur dataset")
def data_example():
    st.download_button(
        label="Contoh Sampel",
        data=open(f'{data_sample}'),
        file_name='Sampel Kacang Hijau 2010-2024.csv',
        use_container_width=True
    )
    st.dataframe(open(f'{data_sample}'))

col = st.columns([4,3,1,3,4])
with col[1]:
    st.link_button(
        "Sumber Data",
        "https://bdsp2.pertanian.go.id/bdsp/id/lokasi",
        use_container_width=True
    )
with col[3]:
    st.download_button(
        label="Contoh Dataset",
        data=open(f'{data_sample}'),
        file_name='Sampel Kacang Hijau 2010-2024.csv',
        use_container_width=True
    )
# with col[3]:
#     if st.button("Contoh Dataset"):
#         data_example()


uploaded_file = st.file_uploader("Unggah data dalam bentuk excel")
if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[-1]
    if file_ext == '.csv':
        dataframe_mentah = pd.read_csv(uploaded_file, sep=';')
    elif file_ext == '.xlsx':
        dataframe_mentah = pd.read_excel(uploaded_file, engine='openpyxl')
    elif file_ext == '.xls':
        dataframe_mentah = pd.read_excel(uploaded_file, engine='xlrd')
    else:
        dataframe_mentah = None
        st.error("Jenis file tidak didukung. Harap unggah file Excel (.csv / .xls / .xlsx).")

    st.dataframe(dataframe_mentah, hide_index=True, height=300)

# def check_input_data(df):
#     # Ensure column names are in string format
#     df.columns = df.columns.map(str)

#     # List of required columns
#     required_list = ['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas']

#     # Check which required columns are missing
#     required_columns = [col for col in required_list if col not in df.columns]

#     # Handle missing columns
#     if required_columns:
#         raise ValueError(f"Terdapat kolom yang kurang lengkap, yaitu {required_columns}")
#     else:
#         # Now you can proceed with your drop or other operations
#         # For example, dropping some columns
#         try:
#             df.drop(columns=required_columns, inplace=True)
#         except KeyError as e:
#             st.error(f"Error while dropping columns: {e}")
    
#     numeric_cols = df.drop(columns=required_list).select_dtypes(include=['number']).columns
#     non_numeric_cols = set(df.columns) - set(numeric_cols) - set(required_list)
    
#     if non_numeric_cols: # jika terdapat kolom non numerik
#         df[list(non_numeric_cols)] = df[list(non_numeric_cols)].apply(pd.to_numeric, errors='coerce')
#         if df.isnull().any().any():
#             error_cols = df.columns[df.isnull().any()].tolist()
#             raise ValueError(f" Nilai non numerik ditemukan di kolom: {error_cols}")
    
#     return df

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

# colmulai = st.columns([3,1,3], gap=None)
# with colmulai[1]:
#     st.markdown("<div style='width: 1px; height: 13px; align-items: center; display: flex;'></div>", unsafe_allow_html=True)
#     mulai = st.button("Mulai Cluster")

st.markdown("<br>", unsafe_allow_html=True)

# MEMULAI PROSES CLUSTERING
# if mulai:
#     if uploaded_file is None:
#         st.error("File tidak ditemukan / belum diunggah.")
#     elif dataframe_mentah is None:
#         st.error("Unggah file data dalam bentuk Excel terlebih dahulu.")
#     elif not metode:
#         st.error("Pilih jenis metode clustering.")
#     elif uploaded_file is not None:
#         dataframe = preprocess_data(dataframe_mentah)
#         df_copy = dataframe.copy()
#         df_array = df_copy.drop(['Lokasi'], axis=1)
#         df_array = df_array.values
        
#         with st.spinner("Clustering data..."):
#             if "Bisecting K-Means" in metode and len(metode) < 2: # BISECTING K-MEANS ONLY
#                 metode = "Bisecting K-Means"
#                 df_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, _, bestcluster_bkmeans, labels_bkmeans = BKMeans(df_array, n_cluster)

#                 # ANALISIS ALGORITMA CLUSTERING
#                 subcol = st.columns([13,13], gap="medium", vertical_alignment='top')
#                 with subcol[0]:
#                     st.write(f"#### {metode}")
#                     st.write("")
#                     evaluate(bestcluster_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, cluster_option)
#                 with subcol[1]:
#                     df_copy['Cluster'] = labels_bkmeans
#                     df_copy = sort_cluster(df_copy)
                    
#                     cluster_category = generate_cluster_category(bestcluster_bkmeans) # beri kategori string
#                     label_mapping = {i: label for i, label in enumerate(cluster_category)}
#                     df_copy['Kategori'] = df_copy['Cluster'].map(label_mapping)

#                     df_copy = avg_features(df_copy) # rata-rata fitur per lokasi

#                     if cluster_option == "Rentang cluster":
#                         st.write("##### Hasil Silhouette dan Davies-Bouldin Index")
#                         fig_evaluate(df_bkmeans, metode)
#                     else:
#                         st.dataframe(df_copy[['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas', 'Cluster', 'Kategori']], hide_index=True, height=400)

#                 # ANALISIS ALGORITMA CLUSTERING
#                 # st.subheader("Hasil Clustering", divider=True, anchor="hasil_clustering")
#                 st.write("#### Pemetaan Tingkat Produksi Kacang Hijau")
                
#                 if cluster_option == "Rentang cluster":
#                     st.dataframe(df_copy[['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas', 'Cluster', 'Kategori']], hide_index=True)
#                     st.write("##### Pemetaan Tingkat Produksi Kacang Hijau")
#                     show_map(df_copy, labels_bkmeans, bestcluster_bkmeans, zoom=5, height=500)
#                     with st.expander("Lihat penjelasan"):
#                         st.write('''
#                             Peta ini menggambarkan wilayah di Indonesia yang sudah dibagi berdasarkan jumlah cluster yang Anda input.
#                             Label tinggi / rendah pada peta adalah berdasarkan tingkat produksi.
#                         ''')
#                 else:
#                     show_map(df_copy, labels_bkmeans, bestcluster_bkmeans, zoom=5, height=500)
#                     with st.expander("Lihat penjelasan"):
#                         st.write('''
#                             Peta ini menggambarkan wilayah di Indonesia yang sudah dibagi berdasarkan jumlah cluster yang Anda input.
#                             Label tinggi / rendah pada peta adalah berdasarkan tingkat produksi.
#                         ''')

#                 # DETAIL HASIL CLUSTER
#                 subcol = st.columns(2, gap="medium")
#                 with subcol[0]:
#                     plot_data_cluster(df_copy, df_copy['Cluster'], metode)
#                 with subcol[1]:
#                     show_n_cluster(df_copy, df_copy["Cluster"], metode)

#                 st.write("#### Perbandingan Fitur Setiap Cluster")
#                 compare_cluster(df_copy, 'Cluster', height=400, direction='horizontal')

#                 # ANALISIS DATA
#                 dataframe_baru = clustering_results_dataframe(df_copy, bestcluster_bkmeans, labels_bkmeans)

#                 cols = st.columns(2, gap="small", vertical_alignment="top")
#                 with cols[0]:
#                     pie_df = dataframe_baru.groupby('Provinsi')['Produksi'].sum().reset_index()
#                     prod_pct = []
#                     unique_provinsi = np.unique(pie_df['Provinsi'])
#                     for i in range(len(unique_provinsi)):
#                         temp = (pie_df.loc[i, 'Produksi'] / pie_df['Produksi'].sum()) * 100
#                         prod_pct.append(round(temp, 2))

#                     pie_df['prod_pct'] = prod_pct
#                     pie_df = pie_df[pie_df['prod_pct'] > 2].reset_index()

#                     labels = pie_df['Provinsi']
#                     values = pie_df['Produksi']

#                     fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', textfont_size=10)])
#                     # fig.update_layout(title_text="Kontribusi Produksi Kacang Hijau 2010 - 2024")
#                     fig.update_layout(
#                         title_text="Kontribusi Produksi Kacang Hijau 2010 - 2024",
#                         title_font_size=22
#                     )
#                     st.plotly_chart(fig)

#                 with cols[1]:
#                     bar_chart = show_prod_dan_lp(df_copy.drop(['Luas Panen', 'Produksi', 'Produktivitas'], axis=1))
#                     st.plotly_chart(bar_chart)

#             if "Agglomerative Hierarchical Clustering" in metode and len(metode) < 2: # AGGLOMERATIVE HIERARCHICAL CLUSTERING ONLY
#                 if linkage is not None:
#                     metode = "Agglomerative Hierarchical Clustering"
#                     linkage = linkage.lower()
#                     df_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, _, bestcluster_ahc, labels_ahc = AHC(df_array, n_cluster, linkage)

#                     # ANALISIS ALGORITMA CLUSTERING
#                     # st.subheader("Evaluasi Model Clustering", divider=True, anchor="evaluasi_model")
#                     subcol = st.columns([13,13], gap="medium")
#                     with subcol[0]:
#                         st.write(f"#### {metode}")
#                         st.write("")
#                         evaluate(bestcluster_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, cluster_option)
#                     with subcol[1]:
#                         df_copy['Cluster'] = labels_ahc
#                         df_copy = sort_cluster(df_copy)
                        
#                         cluster_category = generate_cluster_category(bestcluster_ahc)
#                         label_mapping = {i: label for i, label in enumerate(cluster_category)}
#                         df_copy['Kategori'] = df_copy['Cluster'].map(label_mapping)

#                         if cluster_option == "Rentang cluster":
#                             st.write("#### Hasil Silhouette dan Davies-Bouldin Index")
#                             fig_evaluate(df_ahc, metode)
#                         else:
#                             st.dataframe(df_copy[['Lokasi', 'Cluster', 'Kategori']], hide_index=True, height=400)

#                     # ANALISIS ALGORITMA CLUSTERING
#                     # st.subheader("Hasil Clustering", divider=True, anchor="hasil_clustering")
#                     st.write("#### Pemetaan Tingkat Produksi Kacang Hijau")
                    
#                     if cluster_option == "Rentang cluster":
#                         subcol = st.columns([5,2], gap="medium")
#                         with subcol[0]:
#                             show_map(df_copy, labels_ahc, bestcluster_ahc, zoom=5, height=500)
#                             with st.expander("Lihat penjelasan"):
#                                 st.write('''
#                                     Peta ini menggambarkan wilayah di Indonesia yang sudah dibagi berdasarkan jumlah cluster yang Anda input.
#                                     Label tinggi / rendah pada peta adalah berdasarkan tingkat produksi.
#                                 ''')
#                         with subcol[1]:
#                             st.dataframe(df_copy[['Lokasi', 'Cluster', 'Kategori']], hide_index=True, height=500)
#                     else:
#                         show_map(df_copy, labels_ahc, bestcluster_ahc, zoom=5, height=500)
#                         with st.expander("Lihat penjelasan"):
#                             st.write('''
#                                 Peta ini menggambarkan wilayah di Indonesia yang sudah dibagi berdasarkan jumlah cluster yang Anda input.
#                                 Label tinggi / rendah pada peta adalah berdasarkan tingkat produksi.
#                             ''')

#                     # DETAIL HASIL CLUSTER
#                     subcol = st.columns(2, gap="medium")
#                     with subcol[0]:
#                         plot_data_cluster(df_copy, df_copy['Cluster'], metode)
#                     with subcol[1]:
#                         show_n_cluster(df_copy, df_copy["Cluster"], metode)
                        
#                     # PERBANDINGAN FITUR SETIAP CLUSTER
#                     st.write("#### Perbandingan Fitur Setiap Cluster")
#                     compare_cluster(df_copy, 'Cluster', height=400, direction='horizontal')

#                     # ANALISIS DATA
#                     dataframe_baru = clustering_results_dataframe(df_copy, bestcluster_ahc, labels_ahc)

#                     cols = st.columns(2, gap="small", vertical_alignment="top")
#                     with cols[0]:
#                         pie_df = dataframe_baru.groupby('Provinsi')['Produksi'].sum().reset_index()
#                         prod_pct = []
#                         unique_provinsi = np.unique(pie_df['Provinsi'])
#                         for i in range(len(unique_provinsi)):
#                             temp = (pie_df.loc[i, 'Produksi'] / pie_df['Produksi'].sum()) * 100
#                             prod_pct.append(round(temp, 2))

#                         pie_df['prod_pct'] = prod_pct
#                         pie_df = pie_df[pie_df['prod_pct'] > 2].reset_index()

#                         labels = pie_df['Provinsi']
#                         values = pie_df['Produksi']

#                         fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', textfont_size=10)])
#                         fig.update_layout(
#                             title_text="Kontribusi Produksi Kacang Hijau 2010 - 2024",
#                             title_font_size=22
#                         )
#                         st.plotly_chart(fig)

#                     with cols[1]:
#                         bar_chart = show_prod_dan_lp(df_copy.drop(['Luas Panen', 'Produksi', 'Produktivitas'], axis=1))
#                         st.plotly_chart(bar_chart)

#                 # JIKA LINKAGE BELUM TERPILIH KELUAR WARNING
#                 else:
#                     st.warning("Pilih jenis linkage metode AHC terlebih dahulu.")
            
#             # Ketika memilih kedua metode, akan dibandingkan
#             elif "Bisecting K-Means" in metode and "Agglomerative Hierarchical Clustering" in metode:
#                 if linkage is not None:
#                     linkage = linkage.lower()
#                     df_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, avg_silhouette_bkmeans, bestcluster_bkmeans, labels_bkmeans = BKMeans(df_array, n_cluster)
#                     df_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, avg_silhouette_ahc, bestcluster_ahc, labels_ahc = AHC(df_array, n_cluster, linkage)
#                     df_bkmeans['Metode'] = 'Bisecting K-Means'
#                     df_ahc['Metode'] = 'AHC'

#                     # Menyimpan nama metode
#                     metode1 = 'Bisecting K-Means'
#                     metode2 = 'Agglomerative Hierarchical Clustering'

#                     # JIKA MEMILIH RENTANG CLUSTER
#                     if cluster_option == "Rentang cluster":
#                         result = compare(silhouette_bkmeans, silhouette_ahc, dbi_bkmeans, dbi_ahc, avg_silhouette_bkmeans, avg_silhouette_ahc, bestcluster_bkmeans, bestcluster_ahc)

#                         # PERBANDINGAN ALGORITMA CLUSTERING
#                         st.subheader("Evaluasi Model Clustering", divider=True, anchor="evaluasi_model")
#                         subcol = st.columns([13,13], border=True, gap="medium")
#                         with subcol[0]:
#                             if result == 'BKMeans':
#                                 st.write("#### Bisecting K-Means ✅")
#                             else:
#                                 st.write("#### Bisecting K-Means")
#                             st.write("")
#                             evaluate(bestcluster_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, cluster_option)

#                         with subcol[1]:
#                             if result == 'AHC':
#                                 st.write("#### Agglomerative Hierarchical Clustering ✅")
#                             else:
#                                 st.write("#### Agglomerative Hierarchical Clustering")
#                             st.write("")
#                             evaluate(bestcluster_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, cluster_option)

#                         # SILHOUETTE DAN DBI LINE CHART
#                         fig_silhouette, fig_dbi = linechart_evaluation (df_bkmeans, df_ahc)
#                         subcol = st.columns([13,13], border=True, gap="medium")
#                         with subcol[0]:
#                             st.plotly_chart(fig_silhouette, use_container_width=True)
#                         with subcol[1]:
#                             st.plotly_chart(fig_dbi, use_container_width=True)

#                         # HASIL CLUSTERING ALGORITMA BISECTING K-MEANS DAN AGGLOMERATIVE HIERARCHICAL CLUSTERING
#                         st.subheader("Hasil Clustering", divider=True, anchor="hasil_clustering")
#                         st.write("##### Tabel Kategori Hasil Clustering")
#                         df_temp = columns_to_drop(dataframe_mentah)
#                         df_temp = data_selection (df_temp)
#                         df_temp = df_temp.reset_index()
#                         df_temp = cluster_and_category_result(df_temp, labels_bkmeans, bestcluster_bkmeans, 'Kategori (Bisecting K-Means)', 'Cluster BKM')
#                         df_temp = cluster_and_category_result(df_temp, labels_ahc, bestcluster_ahc, 'Kategori (Agglomerative Hierarchical Clustering)', 'Cluster AHC')
#                         df_temp = avg_features(df_temp)
#                         df_temp = df_temp.drop(columns=df_temp.filter(regex='20', axis=1).columns)
#                         # mapping = {0:1, 1:2, 2:3}
#                         # df_temp['Cluster BKM'] = [mapping[i] for i in df_temp['Cluster BKM']]
#                         # df_temp['Cluster AHC'] = [mapping[i] for i in df_temp['Cluster AHC']]
#                         st.dataframe(df_temp[['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas', 'Cluster BKM',
#                                               'Kategori (Bisecting K-Means)', 'Cluster AHC', 'Kategori (Agglomerative Hierarchical Clustering)']], hide_index=True)

#                         # PEMETAAN HASIL CLUSTERING
#                         subsubcol = st.columns(2, gap="medium")
#                         st.write("##### Pemetaan Cluster Berdasarkan Tingkat Produksi")
#                         with subsubcol[0]:
#                             st.write(f"##### {metode1}")
#                         with subsubcol[1]:
#                             st.write(f"##### {metode2}")
#                         subsubcol = st.columns(2, gap="medium")
#                         with subsubcol[0]:
#                             show_map(df_copy, labels_bkmeans, bestcluster_bkmeans)
#                         with subsubcol[1]:
#                             show_map(df_copy, labels_ahc, bestcluster_ahc)
#                         with st.expander("Lihat penjelasan"):
#                             st.write('''
#                                 Peta ini menggambarkan wilayah di Indonesia yang sudah dibagi berdasarkan jumlah cluster yang Anda input.
#                                 Label tinggi / rendah pada peta adalah berdasarkan tingkat produksi.
#                             ''')

#                         # PIECHART JUMLAH ANGGOTA CLUSTER
#                         subsubcol = st.columns(2, gap="medium")
#                         with subsubcol[0]:
#                             show_n_cluster(df_temp, 'Cluster BKM', metode1)
#                         with subsubcol[1]:
#                             show_n_cluster(df_temp, 'Cluster AHC', metode2)

#                         st.subheader("Perbandingan Fitur Tiap Cluster", divider=True, anchor="perbandingan_cluster")
#                         df_copy = cluster_and_category_result(df_copy, labels_bkmeans, bestcluster_bkmeans, 'Kategori (Bisecting K-Means)', 'Cluster BKM')
#                         df_copy = cluster_and_category_result(df_copy, labels_ahc, bestcluster_ahc, 'Kategori (Agglomerative Hierarchical Clustering)', 'Cluster AHC')
#                         df_copy = avg_features(df_copy)

#                         subsubcol = st.columns(2, gap="medium")
#                         with subsubcol[0]:
#                             st.write("##### Bisecting K-Means")
#                             # compare_cluster(df_copy, 'Cluster BKM', 'Luas Panen (Hektar)', 'Luas Panen 2')
#                             # compare_cluster(df_copy, 'Cluster BKM', 'Produksi (Ton)', 'Produksi 2')
#                             # compare_cluster(df_copy, 'Cluster BKM', 'Produktivitas (Kuintal/Ha)', 'Produktivitas 2')
#                             compare_cluster(df_copy, 'Cluster BKM')
#                         with subsubcol[1]:
#                             st.write("##### Agglomerative Hierarchical Clustering")
#                             # compare_cluster(df_copy, 'Cluster AHC', 'Luas Panen (Hektar)', 'Luas Panen 2')
#                             # compare_cluster(df_copy, 'Cluster AHC', 'Produksi (Ton)', 'Produksi 2')
#                             # compare_cluster(df_copy, 'Cluster AHC', 'Produktivitas (Kuintal/Ha)', 'Produktivitas 2')
#                             compare_cluster(df_copy, 'Cluster AHC')
                        
#                     else: # JIKA MEMILIH JUMLAH CLUSTER TERTENTU
#                         result = compare(silhouette_bkmeans, silhouette_ahc, dbi_bkmeans, dbi_ahc, avg_silhouette_bkmeans, avg_silhouette_ahc, bestcluster_bkmeans, bestcluster_ahc)

#                         st.subheader("Evaluasi Model Clustering", divider=True, anchor="evaluasi_model")
#                         subcol = st.columns([13,13], border=True, gap="medium")
#                         with subcol[0]:
#                             if result == 'BKMeans':
#                                 st.write("#### Bisecting K-Means ✅")
#                             else:
#                                 st.write("#### Bisecting K-Means")
#                             st.write("")
#                             evaluate(bestcluster_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, cluster_option)

#                         with subcol[1]:
#                             if result == 'AHC':
#                                 st.write("#### Agglomerative Hierarchical Clustering ✅")
#                             else:
#                                 st.write("#### Agglomerative Hierarchical Clustering")
#                             st.write("")
#                             evaluate(bestcluster_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, cluster_option)
                        
#                         # HASIL CLUSTERING ALGORITMA BISECTING K-MEANS DAN AGGLOMERATIVE HIERARCHICAL CLUSTERING
#                         st.subheader("Hasil Clustering", divider=True, anchor="hasil_clustering")
#                         st.write("##### Tabel Kategori Hasil Clustering")
#                         df_temp = columns_to_drop(dataframe_mentah)
#                         df_temp = data_selection (df_temp)
#                         df_temp = df_temp.reset_index()
#                         df_temp = cluster_and_category_result(df_temp, labels_bkmeans, bestcluster_bkmeans, 'Kategori (Bisecting K-Means)', 'Cluster BKM')
#                         df_temp = cluster_and_category_result(df_temp, labels_ahc, bestcluster_ahc, 'Kategori (Agglomerative Hierarchical Clustering)', 'Cluster AHC')
#                         df_temp = avg_features(df_temp)
#                         df_temp = df_temp.drop(columns=df_temp.filter(regex='20', axis=1).columns)
#                         # mapping = {0:1, 1:2, 2:3}
#                         # df_temp['Cluster BKM'] = [mapping[i] for i in df_temp['Cluster BKM']]
#                         # df_temp['Cluster AHC'] = [mapping[i] for i in df_temp['Cluster AHC']]
#                         st.dataframe(df_temp[['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas', 'Cluster BKM',
#                                               'Kategori (Bisecting K-Means)', 'Cluster AHC', 'Kategori (Agglomerative Hierarchical Clustering)']], hide_index=True)

#                         # PEMETAAN HASIL CLUSTERING
#                         subsubcol = st.columns(2, gap="medium")
#                         with subsubcol[0]:
#                             st.write("##### Pemetaan dengan Bisecting K-Means")
#                         with subsubcol[1]:
#                             st.write("##### Pemetaan dengan Agglomerative Hierarchical Clustering")
#                         subsubcol = st.columns(2, gap="medium")
#                         with subsubcol[0]:
#                             show_map(df_copy, labels_bkmeans, bestcluster_bkmeans)
#                         with subsubcol[1]:
#                             show_map(df_copy, labels_ahc, bestcluster_ahc)
#                         with st.expander("Lihat penjelasan"):
#                             st.write('''
#                                 Peta ini menggambarkan wilayah di Indonesia yang sudah dibagi berdasarkan jumlah cluster yang Anda input.
#                                 Label tinggi / rendah pada peta adalah berdasarkan tingkat produksi.
#                             ''')

#                         # PIECHART JUMLAH ANGGOTA CLUSTER
#                         subsubcol = st.columns(2, gap="medium")
#                         with subsubcol[0]:
#                             show_n_cluster(df_temp, 'Cluster BKM', metode1)
#                         with subsubcol[1]:
#                             show_n_cluster(df_temp, 'Cluster AHC', metode2)

#                         st.subheader("Perbandingan Fitur Tiap Cluster", divider=True, anchor="perbandingan_cluster")
#                         df_copy = cluster_and_category_result(df_copy, labels_bkmeans, bestcluster_bkmeans, 'Kategori (Bisecting K-Means)', 'Cluster BKM')
#                         df_copy = cluster_and_category_result(df_copy, labels_ahc, bestcluster_ahc, 'Kategori (Agglomerative Hierarchical Clustering)', 'Cluster AHC')
#                         df_copy = avg_features(df_copy)

#                         subsubcol = st.columns(2, gap="medium")
#                         with subsubcol[0]:
#                             st.write("##### Bisecting K-Means")
#                             compare_cluster(df_copy, 'Cluster BKM')
#                         with subsubcol[1]:
#                             st.write("##### Agglomerative Hierarchical Clustering")
#                             compare_cluster(df_copy, 'Cluster AHC')
#                 else:
#                     st.warning("Jenis linkage metode AHC belum dipilih.")
#     else:
#         st.error("File tidak ditemukan / belum diunggah.")
if uploaded_file is not None:
    try:
        validate_columns_and_data(dataframe_mentah, ['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas'])
        dataframe = preprocess_data(dataframe_mentah)
        st.success("Data berhasil divalidasi dan diproses.")
        dataframe = preprocess_data(dataframe_mentah)
        df_copy = dataframe.copy()
        df_array = df_copy.drop(['Lokasi'], axis=1)
        df_array = df_array.values
        
        with st.spinner("Clustering data..."):
            if "Bisecting K-Means" in metode and len(metode) < 2: # BISECTING K-MEANS ONLY
                metode = "Bisecting K-Means"
                df_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, _, bestcluster_bkmeans, labels_bkmeans = BKMeans(df_array, n_cluster)

                # ANALISIS ALGORITMA CLUSTERING
                subcol = st.columns([13,13], gap="medium", vertical_alignment='top')
                with subcol[0]:
                    st.write(f"#### {metode}")
                    st.write("")
                    evaluate(bestcluster_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, cluster_option)
                with subcol[1]:
                    df_copy['Cluster'] = labels_bkmeans
                    df_copy = sort_cluster(df_copy)
                    
                    cluster_category = generate_cluster_category(bestcluster_bkmeans) # beri kategori string
                    label_mapping = {i: label for i, label in enumerate(cluster_category)}
                    df_copy['Kategori'] = df_copy['Cluster'].map(label_mapping)

                    df_copy = avg_features(df_copy) # rata-rata fitur per lokasi

                    if cluster_option == "Rentang cluster":
                        st.write("##### Hasil Silhouette dan Davies-Bouldin Index")
                        fig_evaluate(df_bkmeans, metode)
                    else:
                        st.dataframe(df_copy[['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas', 'Cluster', 'Kategori']], hide_index=True, height=400)

                # ANALISIS ALGORITMA CLUSTERING
                # st.subheader("Hasil Clustering", divider=True, anchor="hasil_clustering")
                st.write("#### Pemetaan Tingkat Produksi Kacang Hijau")
                
                if cluster_option == "Rentang cluster":
                    st.dataframe(df_copy[['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas', 'Cluster', 'Kategori']], hide_index=True)
                    st.write("##### Pemetaan Tingkat Produksi Kacang Hijau")
                    show_map(df_copy, labels_bkmeans, bestcluster_bkmeans, zoom=5, height=500)
                    with st.expander("Lihat penjelasan"):
                        st.write('''
                            Peta ini menggambarkan wilayah di Indonesia yang sudah dibagi berdasarkan jumlah cluster yang Anda input.
                            Label tinggi / rendah pada peta adalah berdasarkan tingkat produksi.
                        ''')
                else:
                    show_map(df_copy, labels_bkmeans, bestcluster_bkmeans, zoom=5, height=500)
                    with st.expander("Lihat penjelasan"):
                        st.write('''
                            Peta ini menggambarkan wilayah di Indonesia yang sudah dibagi berdasarkan jumlah cluster yang Anda input.
                            Label tinggi / rendah pada peta adalah berdasarkan tingkat produksi.
                        ''')

                # DETAIL HASIL CLUSTER
                subcol = st.columns(2, gap="medium")
                with subcol[0]:
                    plot_data_cluster(df_copy, df_copy['Cluster'], metode)
                with subcol[1]:
                    show_n_cluster(df_copy, df_copy["Cluster"], metode)

                st.write("#### Perbandingan Fitur Setiap Cluster")
                compare_cluster(df_copy, 'Cluster', height=400, direction='horizontal')

                # ANALISIS DATA
                dataframe_baru = clustering_results_dataframe(df_copy, bestcluster_bkmeans, labels_bkmeans)

                cols = st.columns(2, gap="small", vertical_alignment="top")
                with cols[0]:
                    pie_df = dataframe_baru.groupby('Provinsi')['Produksi'].sum().reset_index()
                    prod_pct = []
                    unique_provinsi = np.unique(pie_df['Provinsi'])
                    for i in range(len(unique_provinsi)):
                        temp = (pie_df.loc[i, 'Produksi'] / pie_df['Produksi'].sum()) * 100
                        prod_pct.append(round(temp, 2))

                    pie_df['prod_pct'] = prod_pct
                    pie_df = pie_df[pie_df['prod_pct'] > 2].reset_index()

                    labels = pie_df['Provinsi']
                    values = pie_df['Produksi']

                    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', textfont_size=10)])
                    # fig.update_layout(title_text="Kontribusi Produksi Kacang Hijau 2010 - 2024")
                    fig.update_layout(
                        title_text="Kontribusi Produksi Kacang Hijau",
                        title_font_size=20
                    )
                    st.plotly_chart(fig)

                with cols[1]:
                    bar_chart = show_prod_dan_lp(df_copy.drop(['Luas Panen', 'Produksi', 'Produktivitas'], axis=1))
                    st.plotly_chart(bar_chart)

            if "Agglomerative Hierarchical Clustering" in metode and len(metode) < 2: # AGGLOMERATIVE HIERARCHICAL CLUSTERING ONLY
                if linkage is not None:
                    metode = "Agglomerative Hierarchical Clustering"
                    linkage = linkage.lower()
                    df_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, _, bestcluster_ahc, labels_ahc = AHC(df_array, n_cluster, linkage)

                    # ANALISIS ALGORITMA CLUSTERING
                    # st.subheader("Evaluasi Model Clustering", divider=True, anchor="evaluasi_model")
                    subcol = st.columns([13,13], gap="medium")
                    with subcol[0]:
                        st.write(f"#### {metode}")
                        st.write("")
                        evaluate(bestcluster_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, cluster_option)
                    with subcol[1]:
                        df_copy['Cluster'] = labels_ahc
                        df_copy = sort_cluster(df_copy)
                        
                        cluster_category = generate_cluster_category(bestcluster_ahc)
                        label_mapping = {i: label for i, label in enumerate(cluster_category)}
                        df_copy['Kategori'] = df_copy['Cluster'].map(label_mapping)

                        if cluster_option == "Rentang cluster":
                            st.write("#### Hasil Silhouette dan Davies-Bouldin Index")
                            fig_evaluate(df_ahc, metode)
                        else:
                            st.dataframe(df_copy[['Lokasi', 'Cluster', 'Kategori']], hide_index=True, height=400)

                    # ANALISIS ALGORITMA CLUSTERING
                    # st.subheader("Hasil Clustering", divider=True, anchor="hasil_clustering")
                    st.write("#### Pemetaan Tingkat Produksi Kacang Hijau")
                    
                    if cluster_option == "Rentang cluster":
                        subcol = st.columns([5,2], gap="medium")
                        with subcol[0]:
                            show_map(df_copy, labels_ahc, bestcluster_ahc, zoom=5, height=500)
                            with st.expander("Lihat penjelasan"):
                                st.write('''
                                    Peta ini menggambarkan wilayah di Indonesia yang sudah dibagi berdasarkan jumlah cluster yang Anda input.
                                    Label tinggi / rendah pada peta adalah berdasarkan tingkat produksi.
                                ''')
                        with subcol[1]:
                            st.dataframe(df_copy[['Lokasi', 'Cluster', 'Kategori']], hide_index=True, height=500)
                    else:
                        show_map(df_copy, labels_ahc, bestcluster_ahc, zoom=5, height=500)
                        with st.expander("Lihat penjelasan"):
                            st.write('''
                                Peta ini menggambarkan wilayah di Indonesia yang sudah dibagi berdasarkan jumlah cluster yang Anda input.
                                Label tinggi / rendah pada peta adalah berdasarkan tingkat produksi.
                            ''')

                    # DETAIL HASIL CLUSTER
                    subcol = st.columns(2, gap="medium")
                    with subcol[0]:
                        plot_data_cluster(df_copy, df_copy['Cluster'], metode)
                    with subcol[1]:
                        show_n_cluster(df_copy, df_copy["Cluster"], metode)
                        
                    # PERBANDINGAN FITUR SETIAP CLUSTER
                    st.write("##### Perbandingan Fitur Setiap Cluster")
                    compare_cluster(df_copy, 'Cluster', height=400, direction='horizontal')

                    # ANALISIS DATA
                    dataframe_baru = clustering_results_dataframe(df_copy, bestcluster_ahc, labels_ahc)

                    cols = st.columns(2, gap="small", vertical_alignment="top")
                    with cols[0]:
                        pie_df = dataframe_baru.groupby('Provinsi')['Produksi'].sum().reset_index()
                        prod_pct = []
                        unique_provinsi = np.unique(pie_df['Provinsi'])
                        for i in range(len(unique_provinsi)):
                            temp = (pie_df.loc[i, 'Produksi'] / pie_df['Produksi'].sum()) * 100
                            prod_pct.append(round(temp, 2))

                        pie_df['prod_pct'] = prod_pct
                        pie_df = pie_df[pie_df['prod_pct'] > 2].reset_index()

                        labels = pie_df['Provinsi']
                        values = pie_df['Produksi']

                        fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', textfont_size=10)])
                        fig.update_layout(
                            title_text="Kontribusi Produksi Kacang Hijau",
                            title_font_size=20
                        )
                        st.plotly_chart(fig)

                    with cols[1]:
                        bar_chart = show_prod_dan_lp(df_copy.drop(['Luas Panen', 'Produksi', 'Produktivitas'], axis=1))
                        st.plotly_chart(bar_chart)

                # JIKA LINKAGE BELUM TERPILIH KELUAR WARNING
                else:
                    st.warning("Pilih jenis linkage metode AHC terlebih dahulu.")
            
            # Ketika memilih kedua metode, akan dibandingkan
            elif "Bisecting K-Means" in metode and "Agglomerative Hierarchical Clustering" in metode:
                if linkage is not None:
                    linkage = linkage.lower()
                    df_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, avg_silhouette_bkmeans, bestcluster_bkmeans, labels_bkmeans = BKMeans(df_array, n_cluster)
                    df_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, avg_silhouette_ahc, bestcluster_ahc, labels_ahc = AHC(df_array, n_cluster, linkage)
                    df_bkmeans['Metode'] = 'Bisecting K-Means'
                    df_ahc['Metode'] = 'AHC'

                    # Menyimpan nama metode
                    metode1 = 'Bisecting K-Means'
                    metode2 = 'Agglomerative Hierarchical Clustering'

                    # JIKA MEMILIH RENTANG CLUSTER
                    if cluster_option == "Rentang cluster":
                        result = compare(silhouette_bkmeans, silhouette_ahc, dbi_bkmeans, dbi_ahc, avg_silhouette_bkmeans, avg_silhouette_ahc, bestcluster_bkmeans, bestcluster_ahc)

                        # PERBANDINGAN ALGORITMA CLUSTERING
                        st.subheader("Evaluasi Model Clustering", divider=True, anchor="evaluasi_model")
                        subcol = st.columns([13,13], border=True, gap="medium")
                        with subcol[0]:
                            if result == 'BKMeans':
                                st.write("#### Bisecting K-Means ✅")
                            else:
                                st.write("#### Bisecting K-Means")
                            st.write("")
                            evaluate(bestcluster_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, cluster_option)

                        with subcol[1]:
                            if result == 'AHC':
                                st.write("#### Agglomerative Hierarchical Clustering ✅")
                            else:
                                st.write("#### Agglomerative Hierarchical Clustering")
                            st.write("")
                            evaluate(bestcluster_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, cluster_option)

                        # SILHOUETTE DAN DBI LINE CHART
                        fig_silhouette, fig_dbi = linechart_evaluation (df_bkmeans, df_ahc)
                        subcol = st.columns([13,13], border=True, gap="medium")
                        with subcol[0]:
                            st.plotly_chart(fig_silhouette, use_container_width=True)
                        with subcol[1]:
                            st.plotly_chart(fig_dbi, use_container_width=True)

                        # HASIL CLUSTERING ALGORITMA BISECTING K-MEANS DAN AGGLOMERATIVE HIERARCHICAL CLUSTERING
                        st.subheader("Hasil Clustering", divider=True, anchor="hasil_clustering")
                        st.write("##### Tabel Kategori Hasil Clustering")
                        df_temp = columns_to_drop(dataframe_mentah)
                        df_temp = data_selection (df_temp)
                        df_temp = df_temp.reset_index()
                        df_temp = cluster_and_category_result(df_temp, labels_bkmeans, bestcluster_bkmeans, 'Kategori (Bisecting K-Means)', 'Cluster BKM')
                        df_temp = cluster_and_category_result(df_temp, labels_ahc, bestcluster_ahc, 'Kategori (Agglomerative Hierarchical Clustering)', 'Cluster AHC')
                        df_temp = avg_features(df_temp)
                        df_temp = df_temp.drop(columns=df_temp.filter(regex='20', axis=1).columns)
                        # mapping = {0:1, 1:2, 2:3}
                        # df_temp['Cluster BKM'] = [mapping[i] for i in df_temp['Cluster BKM']]
                        # df_temp['Cluster AHC'] = [mapping[i] for i in df_temp['Cluster AHC']]
                        st.dataframe(df_temp[['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas', 'Cluster BKM',
                                                'Kategori (Bisecting K-Means)', 'Cluster AHC', 'Kategori (Agglomerative Hierarchical Clustering)']], hide_index=True)

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
                        with st.expander("Lihat penjelasan"):
                            st.write('''
                                Peta ini menggambarkan wilayah di Indonesia yang sudah dibagi berdasarkan jumlah cluster yang Anda input.
                                Label tinggi / rendah pada peta adalah berdasarkan tingkat produksi.
                            ''')

                        # PIECHART JUMLAH ANGGOTA CLUSTER
                        subsubcol = st.columns(2, gap="medium")
                        with subsubcol[0]:
                            show_n_cluster(df_temp, 'Cluster BKM', metode1)
                        with subsubcol[1]:
                            show_n_cluster(df_temp, 'Cluster AHC', metode2)

                        st.subheader("Perbandingan Fitur Tiap Cluster", divider=True, anchor="perbandingan_cluster")
                        df_copy = cluster_and_category_result(df_copy, labels_bkmeans, bestcluster_bkmeans, 'Kategori (Bisecting K-Means)', 'Cluster BKM')
                        df_copy = cluster_and_category_result(df_copy, labels_ahc, bestcluster_ahc, 'Kategori (Agglomerative Hierarchical Clustering)', 'Cluster AHC')
                        df_copy = avg_features(df_copy)

                        subsubcol = st.columns(2, gap="medium")
                        with subsubcol[0]:
                            st.write("##### Bisecting K-Means")
                            # compare_cluster(df_copy, 'Cluster BKM', 'Luas Panen (Hektar)', 'Luas Panen 2')
                            # compare_cluster(df_copy, 'Cluster BKM', 'Produksi (Ton)', 'Produksi 2')
                            # compare_cluster(df_copy, 'Cluster BKM', 'Produktivitas (Kuintal/Ha)', 'Produktivitas 2')
                            compare_cluster(df_copy, 'Cluster BKM')
                        with subsubcol[1]:
                            st.write("##### Agglomerative Hierarchical Clustering")
                            # compare_cluster(df_copy, 'Cluster AHC', 'Luas Panen (Hektar)', 'Luas Panen 2')
                            # compare_cluster(df_copy, 'Cluster AHC', 'Produksi (Ton)', 'Produksi 2')
                            # compare_cluster(df_copy, 'Cluster AHC', 'Produktivitas (Kuintal/Ha)', 'Produktivitas 2')
                            compare_cluster(df_copy, 'Cluster AHC')
                        
                    else: # JIKA MEMILIH JUMLAH CLUSTER TERTENTU
                        result = compare(silhouette_bkmeans, silhouette_ahc, dbi_bkmeans, dbi_ahc, avg_silhouette_bkmeans, avg_silhouette_ahc, bestcluster_bkmeans, bestcluster_ahc)

                        st.subheader("Evaluasi Model Clustering", divider=True, anchor="evaluasi_model")
                        subcol = st.columns([13,13], border=True, gap="medium")
                        with subcol[0]:
                            if result == 'BKMeans':
                                st.write("#### Bisecting K-Means ✅")
                            else:
                                st.write("#### Bisecting K-Means")
                            st.write("")
                            evaluate(bestcluster_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, cluster_option)

                        with subcol[1]:
                            if result == 'AHC':
                                st.write("#### Agglomerative Hierarchical Clustering ✅")
                            else:
                                st.write("#### Agglomerative Hierarchical Clustering")
                            st.write("")
                            evaluate(bestcluster_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, cluster_option)
                        
                        # HASIL CLUSTERING ALGORITMA BISECTING K-MEANS DAN AGGLOMERATIVE HIERARCHICAL CLUSTERING
                        st.subheader("Hasil Clustering", divider=True, anchor="hasil_clustering")
                        st.write("##### Tabel Kategori Hasil Clustering")
                        df_temp = columns_to_drop(dataframe_mentah)
                        df_temp = data_selection (df_temp)
                        df_temp = df_temp.reset_index()
                        df_temp = cluster_and_category_result(df_temp, labels_bkmeans, bestcluster_bkmeans, 'Kategori (Bisecting K-Means)', 'Cluster BKM')
                        df_temp = cluster_and_category_result(df_temp, labels_ahc, bestcluster_ahc, 'Kategori (Agglomerative Hierarchical Clustering)', 'Cluster AHC')
                        df_temp = avg_features(df_temp)
                        df_temp = df_temp.drop(columns=df_temp.filter(regex='20', axis=1).columns)
                        # mapping = {0:1, 1:2, 2:3}
                        # df_temp['Cluster BKM'] = [mapping[i] for i in df_temp['Cluster BKM']]
                        # df_temp['Cluster AHC'] = [mapping[i] for i in df_temp['Cluster AHC']]
                        st.dataframe(df_temp[['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas', 'Cluster BKM',
                                                'Kategori (Bisecting K-Means)', 'Cluster AHC', 'Kategori (Agglomerative Hierarchical Clustering)']], hide_index=True)

                        # PEMETAAN HASIL CLUSTERING
                        subsubcol = st.columns(2, gap="medium")
                        with subsubcol[0]:
                            st.write("##### Pemetaan dengan Bisecting K-Means")
                        with subsubcol[1]:
                            st.write("##### Pemetaan dengan Agglomerative Hierarchical Clustering")
                        subsubcol = st.columns(2, gap="medium")
                        with subsubcol[0]:
                            show_map(df_copy, labels_bkmeans, bestcluster_bkmeans)
                        with subsubcol[1]:
                            show_map(df_copy, labels_ahc, bestcluster_ahc)
                        with st.expander("Lihat penjelasan"):
                            st.write('''
                                Peta ini menggambarkan wilayah di Indonesia yang sudah dibagi berdasarkan jumlah cluster yang Anda input.
                                Label tinggi / rendah pada peta adalah berdasarkan tingkat produksi.
                            ''')

                        # PIECHART JUMLAH ANGGOTA CLUSTER
                        subsubcol = st.columns(2, gap="medium")
                        with subsubcol[0]:
                            show_n_cluster(df_temp, 'Cluster BKM', metode1)
                        with subsubcol[1]:
                            show_n_cluster(df_temp, 'Cluster AHC', metode2)

                        st.subheader("Perbandingan Fitur Tiap Cluster", divider=True, anchor="perbandingan_cluster")
                        df_copy = cluster_and_category_result(df_copy, labels_bkmeans, bestcluster_bkmeans, 'Kategori (Bisecting K-Means)', 'Cluster BKM')
                        df_copy = cluster_and_category_result(df_copy, labels_ahc, bestcluster_ahc, 'Kategori (Agglomerative Hierarchical Clustering)', 'Cluster AHC')
                        df_copy = avg_features(df_copy)

                        subsubcol = st.columns(2, gap="medium")
                        with subsubcol[0]:
                            st.write("##### Bisecting K-Means")
                            compare_cluster(df_copy, 'Cluster BKM')
                        with subsubcol[1]:
                            st.write("##### Agglomerative Hierarchical Clustering")
                            compare_cluster(df_copy, 'Cluster AHC')
    except ValueError as e:
        st.error(f"Terjadi kesalahan: {e}")
    

# else:
#     df = pd.read_csv('data sample\Kacang Hijau.csv', sep=';')
#     clustering_sample(df)

show_footer()