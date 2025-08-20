from sklearn.cluster import BisectingKMeans, AgglomerativeClustering
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.datasets import load_nfl
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from streamlit_folium import st_folium, folium_static
from branca.element import Element
import json
import pyogrio
import folium
from folium.map import Marker
from folium.features import DivIcon
from branca.element import MacroElement
from jinja2 import Template
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from datetime import date
from io import StringIO
import os
import plotly.express as px
import plotly.graph_objects as go
import re
import geopandas as gpd

import streamlit as st
import pandas as pd
import numpy as np

def preprocess_data(data):
    df = data.replace(0, np.nan)

    column_null = df.isna().sum()/len(df) * 100
    columns_to_drop = column_null[column_null == 100].index
    df.drop(columns_to_drop, axis=1, inplace=True)

    # Drop data with NaNs > 30% and < 100%
    df_clean = df.set_index('Nama Kota') # Exclude 'Nama Kota' from the calculation
    row_null_pct = df_clean.transpose().isna().sum()/len(df_clean.transpose()) * 100 # Calculate % of missing values per row
    rows_to_drop = row_null_pct[(row_null_pct > 35) & (row_null_pct < 100)].index # Identify rows where NaNs > 30% and < 100%
    df_clean = df_clean.drop(index=rows_to_drop) # Drop those rows

    #Handle data with 100% missing values
    df_clean = df_clean.interpolate(method='linear', axis=1, limit_direction='both') # fills missing values horizontally across columns
    df_clean = df_clean.fillna(df_clean.mean(axis=0)) # fills missing values horizontally across columns using the average value of each column across all cities

    df_clean = df_clean.reset_index() # Reset index to restore 'Nama Kota' as a column
    df = df_clean.copy()

    df.columns = df.columns.astype(str) # change columns type
    return df

def normalize(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values)

    data_scaled = pd.DataFrame(scaled_data, columns=data.columns, index=data.index) # Mengubah hasil scaled_data (array) kembali menjadi DataFrame
    data.update(data_scaled) # Memperbarui DataFrame asli
    return data

def BKMeans(data, n_cluster): # Fungsi metode Bisecting K-Means
    silhouette_temp = 0
    silhouette_bkmeans = []
    dbi_bkmeans = []
    waktu_bkmeans = []
    cluster_bkmeans = []
    waktu_avg = 0
    db_index_temp = float('inf')
    silhouette_avg_avg = 0
    avg_silhouette_total = 0
    dbi_avg = float('inf')
    for i in n_cluster:
        silhouette_total = 0
        dbi_total = 0
        waktu_total = 0
        cluster_bkmeans.append(i)
        for j in range (5):
            bkm_clusterer = BisectingKMeans(n_clusters=i, random_state=np.random.randint(1,5000), n_init = 1, bisecting_strategy='largest_cluster')

            start = time.time()
            cluster_labels = bkm_clusterer.fit_predict(data)
            end = time.time()
            waktu = end - start
            waktu_total = waktu_total + waktu

            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_total = silhouette_total + silhouette_avg

            db_index = davies_bouldin_score(data, cluster_labels)
            dbi_total = dbi_total + db_index

        silhouette_avg_avg = silhouette_total/5
        dbi_avg = dbi_total/5
        waktu_avg = waktu_total/5

        silhouette_bkmeans.append(round(silhouette_avg_avg, 3))
        dbi_bkmeans.append(round(dbi_avg, 3))
        waktu_bkmeans.append(round(waktu_avg, 5))

        if silhouette_avg_avg > silhouette_temp:
            silhouette_temp = silhouette_avg_avg
            waktu_temp = waktu
            num_cluster = i
            labels = cluster_labels
            clusterer = bkm_clusterer
            db_index_temp = dbi_avg

        # if dbi_avg < db_index_temp:
        #     db_index_temp = dbi_avg
        #     db_waktu_temp = waktu
        #     db_num_cluster = i

        avg_silhouette_total = avg_silhouette_total + silhouette_avg_avg
    
    avg_silhouette_total = round(avg_silhouette_total/5, 3)

    df_bkmeans = pd.DataFrame(columns=['Cluster', 'Silhouette Score', 'DBI Score'])
    df_bkmeans['Cluster'] = cluster_bkmeans
    df_bkmeans['Silhouette Score'] = silhouette_bkmeans
    df_bkmeans['DBI Score'] = dbi_bkmeans
    df_bkmeans.set_index(['Cluster'], inplace=True)

    dfwaktu_bkmeans = pd.DataFrame(columns=['Cluster', 'Waktu Komputasi (detik)'])
    dfwaktu_bkmeans['Cluster'] = cluster_bkmeans
    dfwaktu_bkmeans['Waktu Komputasi (detik)'] = waktu_bkmeans
    dfwaktu_bkmeans.set_index(['Cluster'], inplace=True)
    print ('\nBisecting K-Means dengan', num_cluster, 'Klaster memiliki rata - rata skor terbaik dengan: \nSilhouette score =', silhouette_temp, '\nDavies-Bouldin Index =', db_index_temp, f'\nWaktu komputasi = {waktu_temp:.6f} detik')
    # Mengembalikan hasil kinerja metode, waktu komputasi, silhouette dan dbi cluster terbaik, rerata silhouette cluster terbaik, cluster terbaik, label
    return df_bkmeans, dfwaktu_bkmeans, round(silhouette_temp, 3), round(db_index_temp, 3), avg_silhouette_total, num_cluster, labels

def AHC(data, n_cluster, linkage): # Fungsi metode AHC mengembalikan skor silhouette, DBI, dan waktu komputasi/pelatihan
    silhouette_ahc = []
    dbi_ahc = []
    waktu_ahc = []
    cluster_ahc = []
    silhouette_temp = 0
    db_index_temp = float('inf')
    silhouette_avg_avg = 0
    waktu_avg = 0
    avg_silhouette_total = 0
    dbi_avg = float('inf')
    for i in n_cluster:
        silhouette_total = 0
        dbi_total = 0
        waktu_total = 0
        cluster_ahc.append(i)
        for j in range (5):
            ahc_clusterer = AgglomerativeClustering(n_clusters=i, metric='euclidean', linkage=linkage)

            start = time.time()
            cluster_labels = ahc_clusterer.fit_predict(data)
            end = time.time()
            waktu = end - start
            waktu_total = waktu_total + waktu

            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_total = silhouette_total + silhouette_avg

            db_index = davies_bouldin_score(data, cluster_labels)
            dbi_total = dbi_total + db_index

        silhouette_avg_avg = silhouette_total/5
        dbi_avg = dbi_total/5
        waktu_avg = waktu_total/5

        silhouette_ahc.append(round(silhouette_avg_avg, 3))
        dbi_ahc.append(round(dbi_avg, 3))
        waktu_ahc.append(round(waktu_avg, 5))

        if silhouette_avg_avg > silhouette_temp:
            silhouette_temp = silhouette_avg_avg
            waktu_temp = waktu
            num_cluster = i
            labels = cluster_labels
            clusterer = ahc_clusterer
            db_index_temp = dbi_avg

    #   if dbi_avg < db_index_temp:
        #     db_index_temp = dbi_avg
        #     db_waktu_temp = waktu
        #     db_num_cluster = i

        avg_silhouette_total = avg_silhouette_total + silhouette_avg_avg
    
    avg_silhouette_total = round(avg_silhouette_total/5, 3)

    df_ahc = pd.DataFrame(columns=['Cluster', 'Silhouette Score', 'DBI Score'])
    df_ahc['Cluster'] = cluster_ahc
    df_ahc['Silhouette Score'] = silhouette_ahc
    df_ahc['DBI Score'] = dbi_ahc

    dfwaktu_ahc = pd.DataFrame(columns=['Cluster', 'Waktu Komputasi (detik)'])
    dfwaktu_ahc['Cluster'] = cluster_ahc
    dfwaktu_ahc['Waktu Komputasi (detik)'] = waktu_ahc

    df_ahc.set_index(['Cluster'],inplace=True)
    dfwaktu_ahc.set_index(['Cluster'],inplace=True)
    print ('\nHierarchical Clustering dengan', num_cluster, 'Klaster memiliki rata - rata skor terbaik dengan: \nSilhouette score =', silhouette_temp, '\nDavies-Bouldin Index =', db_index_temp, f'\nWaktu komputasi = {waktu_temp:.6f} detik')
    return df_ahc, dfwaktu_ahc, round(silhouette_temp, 3), round(db_index_temp, 3), avg_silhouette_total, num_cluster, labels

def create_figure(cluster, df, dfwaktu, metode):
    if cluster == "2 - 10":
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"#### Skor Silhouette dan DBI {metode}")
            fig_bkmeans = px.line(df,
                            markers=True,
                            color_discrete_map={
                                'Silhouette Score': '#BD4B46',
                                'DBI Score': '#8D957E'})
            st.plotly_chart(fig_bkmeans, use_container_width=True)
        with col2:
            st.markdown("#### Waktu Komputasi")
            st.write(dfwaktu)
    # else:
    #     col1, col2, col3 = st.columns(3)
    #     with col1:
    #         df['Waktu Komputasi (detik)'] = dfwaktu
    #         st.write(df)

def analisis(metode, bestcluster, silhouette, dbi, cluster):
    if cluster == "2 - 10":
        st.info(f"""
                ##### Hasil analisis :  
                                    Metode {metode} dengan {bestcluster} klaster memiliki kinerja terbaik dengan silhouette sebesar {silhouette} dan DBI sebesar {dbi}.
                """)
    else:
        st.info(f"""
                ##### Hasil analisis :  
                                    Metode {metode} dengan {bestcluster} klaster menghasilkan silhouette sebesar {silhouette} dan DBI sebesar {dbi}.
                """)
    # st.write(f"Metode {metode} {bestcluster} klaster memiliki kinerja terbaik dengan silhouette {silhouette} dan DBI {dbi}.")

def compare(silhouette_bkmeans, silhouette_ahc, dbi_bkmeans, dbi_ahc, avg_silhouette_bkmeans, avg_silhouette_ahc, bestcluster_bkmeans, bestcluster_ahc):
    unggul_silhouette_bkmeans = silhouette_bkmeans > silhouette_ahc
    unggul_dbi_bkmeans = dbi_bkmeans < dbi_ahc
    unggul_silhouette_ahc = silhouette_ahc > silhouette_bkmeans
    unggul_dbi_ahc = dbi_ahc < dbi_bkmeans
    sama_silhouette = silhouette_bkmeans == silhouette_ahc
    if unggul_silhouette_bkmeans:
        st.info(f"""
                ##### Hasil perbandingan :  
                    Metode Bisecting K-Means dengan {bestcluster_bkmeans} klaster lebih unggul dibandingkan Agglomerative Hierarchical Clustering, yaitu sebesar {silhouette_bkmeans} dan Davies-Bouldin Index sebesar {dbi_bkmeans}.
                """)
        # st.write(f"Metode BKMeans dengan {bestcluster_bkmeans} klaster lebih unggul, yaitu sebesar {silhouette_bkmeans} dan DBI sebesar {dbi_bkmeans}.")
        return "BKMeans"
    elif unggul_silhouette_ahc:
        st.info(f"""
                ##### Hasil perbandingan :  
                    Metode Agglomerative Hierarchical Clustering dengan {bestcluster_ahc} klaster lebih unggul dibandingkan Bisecting K-Means, dengan silhouette sebesar {silhouette_ahc} dan Davies-Bouldin Index sebesar {dbi_ahc}.
                """)
        # st.write(f"Metode AHC dengan {bestcluster_ahc} klaster lebih unggul, dengan silhouette sebesar {silhouette_ahc} dan DBI sebesar {dbi_ahc}.")
        return "AHC"
    elif sama_silhouette:
        if unggul_dbi_bkmeans:
            st.info(f"""
                    ##### Hasil perbandingan :  
                        Hasil silhouette terbaik dengan Agglomerative Hierarchical Clustering {bestcluster_ahc} klaster dan Bisecting K-Means {bestcluster_bkmeans} klaster sama, namun Davies-Bouldin Index yang dihasilkan Bisecting K-Means lebih unggul, yaitu sebesar {dbi_bkmeans}.
                    """)
            # st.write(f"Hasil silhouette terbaik dengan AHC {bestcluster_ahc} klaster dan Bisecting K-Means {bestcluster_bkmeans} klaster sama, namun DBI BKMeans lebih unggul, yaitu sebesar {dbi_bkmeans}.")
            return "BKMeans"
        elif unggul_dbi_ahc:
            st.info(f"""
                    ##### Hasil perbandingan :  
                        Hasil silhouette terbaik Agglomerative Hierarchical Clustering {bestcluster_ahc} klaster dan Bisecting K-Means {bestcluster_bkmeans} klaster sama, namun DBI yang dihasilkan Agglomerative Hierarchical Clustering lebih unggul, yaitu sebesar {dbi_ahc}.
                    """)
            # st.write(f"Hasil silhouette terbaik AHC {bestcluster_ahc} klaster dan Bisecting K-Means {bestcluster_bkmeans} klaster sama, namun DBI AHC lebih unggul, yaitu sebesar {dbi_ahc}.")
            return "AHC"
        elif avg_silhouette_bkmeans > avg_silhouette_ahc:
            st.info(f"""
                    ##### Hasil perbandingan :  
                        Silhouette dan Davies-Bouldin Index Bisecting K-Means dan Agglomerative Hierarchical Clustering sama-sama bernilai {silhouette_ahc} dan {dbi_ahc} namun rata-rata silhouette dan Davies-Bouldin Index seluruh cluster Bisecting K-Means lebih unggul. Maka clustering data menggunakan metode Bisecting K-Means.
                    """)
            # st.write(f"Silhouette dan DBI Bisecting K-Means dan AHC sama-sama bernilai {silhouette_ahc} dan {dbi_ahc} namun rata-rata silhouette dan DBI seluruh cluster BKMeans lebih unggul maka visualisasi menggunakan metode BKMeans.")
            return "BKMeans"
        elif avg_silhouette_ahc > avg_silhouette_bkmeans:
            st.info(f"""
                    ##### Hasil perbandingan :  
                        Silhouette dan Davies-Bouldin Index Bisecting K-Means dan Agglomerative Hierarchical Clustering sama-sama bernilai {silhouette_ahc} dan {dbi_ahc} namun rata-rata silhouette dan Davies-Bouldin Index seluruh cluster Agglomerative Hierarchical Clustering lebih unggul. Maka clustering data menggunakan metode Agglomerative Hierarchical Clustering.
                    """)
            # st.write(f"Silhouette dan DBI Bisecting K-Means dan AHC sama-sama bernilai {silhouette_ahc} dan {dbi_ahc} namun rata-rata silhouette dan DBI seluruh cluster AHC lebih unggul maka visualisasi menggunakan metode AHC.")
            return "AHC"
        else:
            return "Waktu"
    else:
        st.warning("Belum ditentukan.")
        return "Neither"
    
def visualize_data(data, cluster_labels, metode):
    data['Cluster'] = cluster_labels.astype(str)
    x=data.iloc[:, 0]
    y=data.iloc[:, 1]
    fig = px.scatter(data_frame=data,
                     x=x, y=y,
                     color='Cluster',
                     symbol='Cluster',
                     width=500,
                     height=400
                     )
    fig.update_traces(marker=dict(size=7, line=dict(width=1, color='black')))
    fig.update_layout(
        title=dict(text=f'Ruang Cluster Data dengan {metode}'),
        legend_title_text='Cluster'
    )
    fig.update_xaxes(title='Ruang Fitur 1', showticklabels=False)
    fig.update_yaxes(title='Ruang Fitur 2', showticklabels=False)
    st.plotly_chart(fig, theme=None, use_container_width=False)

def visualize_silhouette(data, cluster_labels, n_clusters, silhouette_avg, metode):
    silhouette_values = silhouette_samples(data, cluster_labels)

    fig = go.Figure()
    y_lower = 10

    for i in range(n_clusters):
        # Mencari silhouette dari cluster i dan diurut
        ith_vals = silhouette_values[cluster_labels == i]
        ith_vals.sort()

        size_cluster_i = len(ith_vals)
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        color_rgba = f'rgba({int(color[3]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 0.7)'

        fig.add_trace(go.Bar(
            x=ith_vals,
            y=np.arange(y_lower, y_upper),
            orientation='h',
            marker=dict(color=color_rgba),
            showlegend=True,
            name=f'Cluster {i} | {size_cluster_i}'
        ))

        # Add cluster label to the middle
        fig.add_annotation(
            x=-0.1,
            y=y_lower + 0.5 * size_cluster_i,
            text=f'Cluster {i}',
            showarrow=False,
            font=dict(size=12)
        )

        y_lower = y_upper + 10  # Space between clusters

    # Add the average silhouette score line
    fig.add_shape(
        type='line',
        x0=silhouette_avg, y0=0,
        x1=silhouette_avg, y1=y_lower,
        line=dict(color='red', dash='dash'),
        name='Rerata silhouette',
        showlegend=True
    )

    fig.add_trace(go.Scatter(
        x=[0.8],
        y=[str(y_lower-0.1)],
        text=[f"Rerata silhouette = {round(silhouette_avg, 3)}"],
        mode="text",
        showlegend=False
    ))

    fig.update_layout(
        title=f'Silhouette Plot Metode {metode} dengan {n_clusters} Cluster',
        xaxis=dict(title='Koefisien Silhouette', range=[-0.2, 1]),
        yaxis=dict(title='Label Cluster', showticklabels=False),
        # height=450,
        # margin=dict(l=50, r=50, t=80, b=50)
    )

    st.plotly_chart(fig, theme=None, use_container_width=True)

def visualize_cluster(data, labels, n_cluster):
    data['Cluster'] = labels # Menggabungkan label ke Klaster ke dataset

    data = data.drop(['Nama Kota'], axis = 1).apply(pd.to_numeric)

    Clusters = {}
    Mean = {}

    for i in range(1, n_cluster+1):
        Clusters[f'Cluster{i}'] = data[data['Cluster'] == i-1]
        Mean[f'mean{i}'] = Clusters[f'Cluster{i}'].describe().loc['mean']

    for i in range(1, n_cluster+1):
        st.write(Mean[f'mean{i}'])

def sort_cluster (data):
    mean = {}
    unique_clusters = np.sort(data['Cluster'].unique())
    for i in range (len(unique_clusters)):
        cluster = data[data['Cluster'] == unique_clusters[i]]
        mean[i] = cluster.describe().loc['mean'].mean()

    # Sort the dictionary by mean value in descending order
    sorted_clusters = sorted(mean.items(), key=lambda item: item[1], reverse=False)

    # Create a mapping from original cluster labels to new labels (0, 1, 2, ... based on sorted mean)
    cluster_mapping = {original_label: new_label for new_label, (original_label, mean_value) in enumerate(sorted_clusters)}

    # Apply the new cluster labels to your DataFrame
    data['Cluster'] = data['Cluster'].map(cluster_mapping)
    return data

def penyesuaian(data):
    data['Nama Kota'] = data['Nama Kota'].str.replace( # Format nama kota
        r'^\s*\d+\s*-\s*(kab\.?)\s*\.?\s*', '',
        flags=re.IGNORECASE,
        regex=True
    )
    data['Nama Kota'] = data['Nama Kota'].str.replace(r'^\d+\s*-\s*', '', regex=True).str.strip()

    # Kamus penyesuaian nama provinsi jika diperlukan
    kamus_penyesuaian = {
        "Daerah Khusus Ibukota Jakarta": "Jakarta Raya", "Kepulauan Bangka Belitung": "Bangka Belitung", "Daerah Istimewa Yogyakarta": "Yogyakarta",
        "Tanjung Jabung Barat": "Tanjung Jabung B", "Tanjung Jabung Timur": "Tanjung Jabung T", "Bireun": "Bireuen", "Aceh Pidie": "Pidie",
        "Kota Banjarbaru": "Banjar Baru", "Batubara": "Batu Bara", "Kota Gunung Sitoli": "Gunungsitoli", "Kota Tanjung Balai": "Kota Tanjungbalai",
        "Labuhan Batu": "Labuhanbatu", "Labuhan Batu Selatan": "Labuhanbatu Selatan", "Labuhan Batu Utara": "Labuhanbatu Utara",
        "Kota Padang Sidimpuan": "Padangsidimpuan", "PakPak Bharat": "Pakpak Barat", "Kota Pematang Siantar": "Pematangsiantar",
        "Kota Tebing Tinggi": "Tebingtinggi", "Kota Bau Bau": "Bau-Bau", "Kota Bukit Tinggi": "Bukittinggi", "Dharmas Raya": "Dharmasraya",
        "Kota Sawah Lunto": "Sawahlunto", "Fak-Fak": "Fakfak", "Kota. Sorong": "Kota Sorong", "Kota Denpasar": "Denpasar",
        "Karang Asem": "Karangasem", "Kota Cimahi": "Cimahi", "Kota Depok": "Depok", "Kota Mataram": "Mataram", "Kota Batu": "Batu",
        "Kota Surabaya": "Surabaya", "Peg. Bintang": "Pegunungan Bintang", "Kab: Lembata": "Lembata", "Kota Salatiga": "Salatiga",
        "Kota Surakarta": "Surakarta", "Pontianak": "Kota Pontianak", "Kota Singkawang": "Singkawang", "Kota Cilegon": "Cilegon",
        "Kota Tangerang Selatan": "Tangerang Selatan", "Kota Bitung": "Bitung", "Kota Manado": "Manado", "Kota Tomohon": "Tomohon",
        "Kota Waringin Barat": "Kotawaringin Barat", "Kota Waringin Timur": "Kotawaringin Timur", "Kota Palangkaraya": "Palangka Raya",
        "Kota Balikpapan": "Balikpapan", "Kota Bontang": "Bontang", "Kutai Kertanegara": "Kutai Kartanegara", "Kota Samarinda": "Samarinda",
        "Kota Lubuk Linggau": "Lubuklinggau", "Musi Banyu Asin": "Musi Banyuasin", "Kota Pagaralam": "Pagar Alam", "Kota Palembang": "Palembang",
        "Kota Prabumulih": "Prabumulih", "Kota Sibolga": "Sibolga", "Mamuju Utara/Kab. Pasangkayu": "Mamuju Utara", "Kota Bengkulu": "Bengkulu",
        "Muko Muko": "Mukomuko", "OKU Selatan": "Ogan Komering Ulu Selatan", "OKU Timur": "Ogan Komering Ulu Timur",
        "Pangkajene dan Kepulauan": "Pangkajene Dan Kepulauan", "Kota Pare-Pare": "Parepare", "Kota Makassar": "Makassar", "Kota Palopo": "Palopo",
        "Kota Pangkal Pinang": "Pangkalpinang", "Kota Tanjung Pinang": "Tanjungpinang", "Kota Batam": "Batam", "Tojo Una-una": "Tojo Una-Una",
        "Kota Palu": "Palu", "Tulang Bawang": "Tulangbawang", "Kota Metro": "Metro", "Kota Bandar Lampung": "Bandar Lampung", "Kota Ambon": "Ambon",
        "Kota Banda Aceh": "Banda Aceh", "Kota Banjarmasin": "Banjarmasin", "Kota Kendari": "Kendari", "Kota Padang": "Padang",
        "Kota Padang Panjang": "Padang Panjang", "Kota Pariaman": "Pariaman", "Kota Payakumbuh": "Payakumbuh", "Kota Dumai": "Dumai",
        "Kota Pekanbaru": "Pekanbaru", "Kota Jakarta Barat": "Jakarta Barat", "Kota Jakarta Pusat": "Jakarta Pusat",
        "Kota Jakarta Selatan": "Jakarta Selatan", "Kota Jakarta Timur": "Jakarta Timur", "Kota Jakarta Utara": "Jakarta Utara",
        "Kota Jambi": "Jambi", "Kota Langsa": "Langsa", "Kota Lhokseumawe": "Lhokseumawe", "Kota Subulussalam": "Subulussalam",
        "Kota Sabang": "Sabang", ": Lembata": "Lembata", "Kota Sungai Penuh": "Sungai Penuh", "Kota Tarakan": "Tarakan", "Kota Ternate": "Ternate",
        "Kota Tidore Kepulauan": "Tidore Kepulauan", "Kota Tual": "Tual"
        }
    
    data['Nama Kota'] = data['Nama Kota'].replace(kamus_penyesuaian) # Terapkan kamus penyesuaian
    return data

def generate_cluster_labels(n_cluster):
    base_labels = ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
    if n_cluster == len(base_labels):
        return base_labels[:n_cluster]
    elif n_cluster == 2:
        return base_labels[1:4:2]
    elif n_cluster == 3:
        return base_labels[1:4]
    elif n_cluster == 4:
        return [base_labels[0], base_labels[1], base_labels[3], base_labels[4]]
    else:
        return base_labels + [f'Cluster {i}' for i in range(len(base_labels), n_cluster)]

def map_folium(data, n_cluster):
    gdf = gpd.read_file('gadm_IDN/gadm41_IDN_2.shp') # Membuat shapefile Indonesia
    data_clustering = data[['Nama Kota', 'Cluster']].rename(columns={'Nama Kota': 'provinsi'}) # Ganti 'Nama Kota' menjadi 'provinsi' untuk konsistensi
    gdf_provinsi = gdf.dissolve(by='NAME_2').reset_index() # Menggabungkan data kabupaten ke provinsi
    gdf_provinsi = gdf_provinsi.merge(data_clustering, left_on='NAME_2', right_on='provinsi', how='left') # Menggabungkan dengan data clustering
    gdf_provinsi['provinsi'] = gdf_provinsi['NAME_1']

    # Define a color map for clusters
    color_map = {
        0: '#3a52b3',
        1: 'red',
        2: "#F2EF4B",
        3: '#FF9E1F',
        4: '#61c956'
    }

    gdf_provinsi['Cluster'] = pd.to_numeric(gdf_provinsi['Cluster'], errors='coerce')
    cluster_labels = generate_cluster_labels(n_cluster)

    gdf_provinsi['color'] = gdf_provinsi['Cluster'].map(color_map)

    label_mapping = {i: label for i, label in enumerate(cluster_labels)}

    unique_clusters = gdf_provinsi[['Cluster', 'color']].drop_duplicates()
    unique_clusters = unique_clusters.sort_values(by='Cluster')
    unique_clusters['Cluster'] = unique_clusters['Cluster'].map(label_mapping).fillna('Tidak ada data')
    unique_clusters['color'] = unique_clusters['color'].fillna('lightgrey')

    # Replace cluster numbers with labels
    gdf_provinsi['Cluster'] = gdf_provinsi['Cluster'].map(label_mapping).fillna('Tidak ada data').astype(str)

    gdf_provinsi['color'] = gdf_provinsi['color'].fillna('lightgrey') # Warna abu-abu untuk kabupaten/kota tanpa data
    print(unique_clusters)

    # Simplify the geometries in gdf_provinsi, keeping the 'NAME_2' column
    gdf_provinsi_simplified = gdf_provinsi[['NAME_2', 'geometry']].simplify(tolerance=0.01, preserve_topology=True)

    # Convert the simplified GeoSeries to a GeoDataFrame (if necessary, though simplify on subset should return gdf)
    if isinstance(gdf_provinsi_simplified, gpd.GeoSeries):
        gdf_provinsi_simplified = gpd.GeoDataFrame(geometry=gdf_provinsi_simplified, crs=gdf_provinsi.crs)
        gdf_provinsi_simplified['NAME_2'] = gdf_provinsi['NAME_2'] # Re-add NAME_2 if simplify dropped it

    # Merge the 'Cluster' and 'provinsi' columns back into the simplified GeoDataFrame
    gdf_provinsi_simplified = gdf_provinsi_simplified.merge(gdf_provinsi[['NAME_2', 'Cluster', 'provinsi', 'color']], on='NAME_2', how='left')

    # Replace NaN in 'Cluster' with 'no data'
    gdf_provinsi_simplified['Cluster'] = gdf_provinsi_simplified['Cluster'].fillna('no data').astype(str)
    gdf_provinsi_simplified.rename(columns={'NAME_2': 'Nama Kota', 'provinsi': 'Provinsi'}, inplace=True)

    m = folium.Map(location=[-2.5, 117], zoom_start=5, tiles='cartodbpositron')

    for _, row in unique_clusters.iterrows():
        cluster_label = row['Cluster']
        cluster_color = row['color']
        
        legend_label = f'<span style="color:{cluster_color};">{cluster_label}</span>'
        fg = folium.FeatureGroup(name=legend_label, show=True)

        # Filter GeoDataFrame for this cluster
        gdf_cluster = gdf_provinsi_simplified[gdf_provinsi_simplified['Cluster'] == cluster_label]

        # Create GeoJson just for this cluster
        gj = folium.GeoJson(
            gdf_cluster,
            style_function=lambda feature, color=cluster_color: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.7,
            },
            tooltip=folium.GeoJsonTooltip(fields=['Cluster', 'Nama Kota', 'Provinsi']),
        )
        gj.add_to(fg)
        fg.add_to(m)

    # Define the bounding box (southwest, northeast)
    bounds = [[-11, 95], [4, 140]]

    # Fit the map to the defined bounds
    m.fit_bounds(bounds)
    m.add_child(folium.LatLngPopup())

    folium.LayerControl(position='bottomleft', collapsed=False).add_to(m)
    folium_static(m, height=400)

def map_out(data):
    gdf = gpd.read_file('gadm_IDN/gadm41_IDN_2.shp') # Membuat shapefile Indonesia
    data_clustering = data[['Nama Kota', 'Cluster']].rename(columns={'Nama Kota': 'provinsi'}) # Ganti 'Nama Kota' menjadi 'provinsi' untuk konsistensi
    gdf_provinsi = gdf.dissolve(by='NAME_2').reset_index() # Menggabungkan data kabupaten ke provinsi 
    gdf_provinsi = gdf_provinsi.merge(data_clustering, left_on='NAME_2', right_on='provinsi', how='left') # Menggabungkan dengan data clustering
    gdf_provinsi['provinsi'] = gdf_provinsi['NAME_1']
    gdf_provinsi_simplified = gdf_provinsi[['NAME_2', 'geometry']].simplify(tolerance=0.01, preserve_topology=True) # Simplify the geometries in gdf_provinsi, keeping the 'NAME_2' column

    # Convert the simplified GeoSeries to a GeoDataFrame (if necessary, though simplify on subset should return gdf)
    if isinstance(gdf_provinsi_simplified, gpd.GeoSeries):
        gdf_provinsi_simplified = gpd.GeoDataFrame(geometry=gdf_provinsi_simplified, crs=gdf_provinsi.crs)
        # Re-add NAME_2 if simplify dropped it
        gdf_provinsi_simplified['NAME_2'] = gdf_provinsi['NAME_2']

    # Merge the 'Klaster' and 'provinsi' columns back into the simplified GeoDataFrame
    gdf_provinsi_simplified = gdf_provinsi_simplified.merge(gdf_provinsi[['NAME_2', 'Cluster', 'provinsi']], on='NAME_2', how='left')

    # Replace NaN in 'Klaster' with 'no data'
    gdf_provinsi_simplified['Cluster'] = gdf_provinsi_simplified['Cluster'].fillna('no data').astype(str)

    # Convert the simplified GeoDataFrame to GeoJSON
    geojson_data_simplified = json.loads(gdf_provinsi_simplified.to_json())

    # Define a color map that includes 'no data' and specific colors for clusters
    color_discrete_map = {
        '0.0': 'red',  # Explicitly assign red to cluster 0.0
        '1.0': '#3a52b3',
        '2.0': '#D4D11E',
        '3.0': '#61c956',
        '4.0': '#86f7ff',
        '5.0': '#377eb8',
        '6.0': '#FF9E1F',
        '7.0': '#905bc2',
        '8.0': '#ffaaeb',
        '9.0': '#9c6b00',
        'no data': '#D3D3D3' # Explicitly assign lightgrey using hex code
    }

    fig = px.choropleth(gdf_provinsi_simplified, geojson=geojson_data_simplified, color="Cluster",
                        locations="NAME_2", featureidkey="properties.NAME_2",
                        projection="orthographic", hover_data=["provinsi", "Cluster"],
                        color_discrete_map=color_discrete_map # Use the defined color map
                    )
    fig.update_geos(fitbounds="locations", visible=False, projection_distance=3)
    # fig.update_layout(margin={"r":0,"t":50,"l":0,"b":50}, width=800, height=600) # Added width and height
    fig.update_layout(autosize=False,
        margin = dict(
                l=0,
                r=0,
                b=0,
                t=0,
                pad=4,
                autoexpand=True
            )
    )
    
    st.plotly_chart(fig, use_container_width=True, width=800, height=400)

def graph_comparison(data):
    nama_kolom = data.columns
    lp = nama_kolom[nama_kolom.str.startswith('Luas Panen')]

    tahun = []
    for j in range(len(lp)):
        res = lp[j].rsplit(' ', 1) # Pisah kata terakhir yang dipisah spasi
        tahun.append(res[1])

    fig = go.Figure()

    for i in range(len(data)):
        city_names = data['Nama Kota'].iloc[i]
        luas_panen_values = data.loc[i, data.columns.str.startswith('Luas Panen')]

    fig.add_trace(go.Scatter(x=tahun, y=luas_panen_values, mode='lines+markers', name=city_names))

    fig.update_layout(
        title='Luas Panen per Kota',
        xaxis_title='Tahun',
        yaxis_title='Luas Panen (Ha)',
        hovermode='x unified'
    )

    fig.show()