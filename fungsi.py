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
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from datetime import date
from io import StringIO
import os
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import re
import geopandas as gpd

import streamlit as st
import pandas as pd
import numpy as np

def validate_columns_and_data(data, required_columns):
    # Cek jika kolom ditemukan (e.g., 'Luas Panen', 'Produksi', etc.)
    missing_columns = []
    for col in required_columns:
        matching_columns = [c for c in data.columns if c.startswith(col)]
        if not matching_columns:
            missing_columns.append(col)
    
    if missing_columns:
        raise ValueError(f"Kolom-kolom berikut tidak ditemukan: {', '.join(missing_columns)}")

    # Check if all matching columns contain numeric data
    for col in required_columns:
        matching_columns = [c for c in data.columns if c.startswith(col)]
        
        for matching_col in matching_columns:
            if matching_col == 'Lokasi':
                continue
            # Check if the data in the column is numeric
            if not pd.to_numeric(data[matching_col], errors='coerce').notna().all():
                raise ValueError(f"Kolom '{matching_col}' mengandung nilai non-numerik. Harap perbaiki data.")
    
    return True  # If all validations pass, return True

def preprocess_data(data):
    # numeric_columns = [col for col in data.columns if col != 'Lokasi']
    # for col in numeric_columns:
    #     # If the column contains non-numeric values, set them as NaN
    #     data[col] = pd.to_numeric(data[col], errors='coerce')
        
    #     # If any NaN values were created (indicating invalid data), raise an error
    #     if data[col].isna().any():
    #         raise ValueError(f"Kolom '{col}' berisi nilai non-numerik. Silakan perbaiki terlebih dahulu sebelum melanjutkan.")
        
    data = data.replace(0, np.nan)

    column_null = data.isna().sum()/len(data) * 100
    columns_to_drop = column_null[column_null == 100].index
    data.drop(columns_to_drop, axis=1, inplace=True)

    # Drop data dgn NaNs > 30% & < 100%
    df_clean = data.set_index('Lokasi')
    row_null_pct = df_clean.transpose().isna().sum()/len(df_clean.transpose()) * 100 # persentase NaNs per baris
    rows_to_drop = row_null_pct[(row_null_pct > 35) & (row_null_pct < 100)].index # NaNs > 30% & < 100%
    df_clean = df_clean.drop(index=rows_to_drop)

    # Handle data with 100% missing values
    df_clean = df_clean.interpolate(method='linear', axis=1, limit_direction='both') # interpolasi linear horizontal tiap kolom
    df_clean = df_clean.fillna(df_clean.mean(axis=0)) # yg 100% NaNs diisi pke mean

    df_clean = df_clean.reset_index() # Reset index 'Lokasi'

    df_clean.columns = df_clean.columns.astype(str) # change columns type
    return df_clean

def columns_to_drop (data):
    data = data.replace(0, np.nan)
    # drop columns with 100% 0 values
    column_null = data.isna().sum()/len(data) * 100
    columns_to_drop = column_null[column_null == 100].index
    data.drop(columns_to_drop, axis=1, inplace=True)
    data.replace(np.nan, 0, inplace=True)
    return data

def data_selection (data):
    columns_to_drop (data)
    data = data.replace(0, np.nan)

    # Drop data dgn NaNs > 30% & < 100%
    df_clean = data.set_index('Lokasi')
    row_null_pct = df_clean.transpose().isna().sum()/len(df_clean.transpose()) * 100 # persentase NaNs per baris
    rows_to_drop = row_null_pct[(row_null_pct > 35) & (row_null_pct < 100)].index # NaNs > 30% & < 100%
    df_clean = df_clean.drop(index=rows_to_drop)
    # df_clean = df_clean.reset_index() # Reset index 'Lokasi'
    df_clean = df_clean.replace(np.nan, 0)

    return df_clean

def handle_null(data):
    data = data.replace(0, np.nan)
    #Handle data with 100% missing values
    data = data.interpolate(method='linear', axis=1, limit_direction='both') # interpolasi linear horizontal tiap kolom
    data = data.fillna(data.mean(axis=0)) # yg 100% NaNs diisi pke mean

    data = data.reset_index() # Reset index 'Lokasi'

    data.columns = data.columns.astype(str) # change columns type
    
    return data

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
            bkm_clusterer = BisectingKMeans(n_clusters=i, random_state=np.random.randint(1,5000), n_init = 1, bisecting_strategy='biggest_inertia')

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
    # print ('\nHierarchical Clustering dengan', num_cluster, 'Klaster memiliki rata - rata skor terbaik dengan: \nSilhouette score =', silhouette_temp, '\nDavies-Bouldin Index =', db_index_temp, f'\nWaktu komputasi = {waktu_temp:.6f} detik')
    return df_ahc, dfwaktu_ahc, round(silhouette_temp, 3), round(db_index_temp, 3), avg_silhouette_total, num_cluster, labels

def fig_evaluate(df, metode):
    fig_bkmeans = px.line(df,
                    markers=True,
                    color_discrete_map={
                        'Silhouette Score': '#BD4B46',
                        'DBI Score': '#8D957E'}, 
                        # title=f"Skor Silhouette dan Davies-Bouldin Index {metode}"
                        )
    st.plotly_chart(fig_bkmeans, use_container_width=True)

def linechart_evaluation (df_bkmeans, df_ahc):
    method_dataframe = pd.concat([df_bkmeans, df_ahc], ignore_index=False)
    method_dataframe = method_dataframe.reset_index() # Reset index

    fig_silhouette = px.line(
        method_dataframe,
        x='Cluster',
        y='Silhouette Score',
        color='Metode',
        markers=True,
        title='Silhouette Score',
        color_discrete_map={
            'Bisecting K-Means': '#BD4B46',
            'AHC': '#8D957E'
        }
    )

    # Plot DBI Score
    fig_dbi = px.line(
        method_dataframe,
        x='Cluster',
        y='DBI Score',
        color='Metode',
        markers=True,
        title='DBI Score',
        color_discrete_map={
            'Bisecting K-Means': '#BD4B46',
            'AHC': '#8D957E'
        }
    )

    return fig_silhouette, fig_dbi

def compare(silhouette_bkmeans, silhouette_ahc, dbi_bkmeans, dbi_ahc, avg_silhouette_bkmeans, avg_silhouette_ahc, bestcluster_bkmeans, bestcluster_ahc):
    conditions = [silhouette_bkmeans > silhouette_ahc, dbi_bkmeans < dbi_ahc, silhouette_ahc > silhouette_bkmeans,
                  dbi_ahc < dbi_bkmeans, silhouette_bkmeans == silhouette_ahc]
    if conditions[0]:
        return "BKMeans"
    elif conditions[2]:
        return "AHC"
    elif conditions[4]:
        if conditions[1]:
            return "BKMeans"
        elif conditions[3]:
            return "AHC"
        elif avg_silhouette_bkmeans > avg_silhouette_ahc:
            return "BKMeans"
        elif avg_silhouette_ahc > avg_silhouette_bkmeans:
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

    data = data.drop(['Lokasi'], axis = 1).apply(pd.to_numeric)

    Clusters = {}
    Mean = {}

    for i in range(1, n_cluster+1):
        Clusters[f'Cluster{i}'] = data[data['Cluster'] == i-1]
        Mean[f'mean{i}'] = Clusters[f'Cluster{i}'].describe().loc['mean']

    for i in range(1, n_cluster+1):
        st.write(Mean[f'mean{i}'])

def sort_cluster (data, nama_kolom='Cluster'): # fungsi sort cluster berdasarkan mean
    mean = {}
    unique_clusters = np.sort(data[f'{nama_kolom}'].unique()) # Hitung banyak cluster
    for i in range (len(unique_clusters)):
        cluster = data[data[f'{nama_kolom}'] == unique_clusters[i]]
        mean[i] = cluster.describe().loc['mean'].mean()

    # sort secara descending
    sorted_clusters = sorted(mean.items(), key=lambda item: item[1], reverse=False)

    # mapping label cluster menjadi label baru (0, 1, 2, ... berdasarkan mean yang terurut)
    cluster_mapping = {original_label: new_label for new_label, (original_label, mean_value) in enumerate(sorted_clusters)}

    # ganti jd label cluster baru 
    data[f'{nama_kolom}'] = data[f'{nama_kolom}'].map(cluster_mapping)
    return data

def penyesuaian(data):
    data['Lokasi'] = data['Lokasi'].str.replace( # Format nama kota
        r'^\s*\d+\s*-\s*(kab\.?)\s*\.?\s*', '',
        flags=re.IGNORECASE,
        regex=True
    )
    data['Lokasi'] = data['Lokasi'].str.replace(r'^\d+\s*-\s*', '', regex=True).str.strip()

    # Menyesuaikan nama
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
        "Kota Tidore Kepulauan": "Tidore Kepulauan", "Kota Tual": "Tual", "Banjar": "Kabupaten Banjar"
        }
    
    data['Lokasi'] = data['Lokasi'].replace(kamus_penyesuaian)
    data['Lokasi'] = data['Lokasi'].replace({"Kota Banjar": "Banjar"})
    return data

def generate_cluster_category(n_cluster):
    BASE_LABELS = {
        2: ['Rendah', 'Tinggi'],
        3: ['Rendah', 'Sedang', 'Tinggi'],
        4: ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi'],
        5: ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
    }
    return BASE_LABELS.get(n_cluster, [f"Cluster {i}" for i in range(n_cluster)])

def merge_gdf(data):
    gdf = gpd.read_file('gadm_IDN/gadm41_IDN_2.shp') # Membuat shapefile Indonesia
    data_clustering = data[['Lokasi', 'Cluster']].rename(columns={'Lokasi': 'provinsi'}) # Ganti 'Lokasi' menjadi 'provinsi' untuk menyesuaikan
    gdf_provinsi = gdf.dissolve(by='NAME_2').reset_index() # Menggabungkan data kabupaten ke provinsi
    gdf_provinsi = gdf_provinsi.merge(data_clustering, left_on='NAME_2', right_on='provinsi', how='left') # Menggabungkan dengan data clustering
    gdf_provinsi['provinsi'] = gdf_provinsi['NAME_1']
    
    return gdf_provinsi

def map_folium(data, n_cluster, zoom=4, width=1500, height=400):
    gdf_provinsi = merge_gdf(data)

    color_map = {
        0: '#BE2A3E',
        1: '#F88F4D',
        2: '#F4D166',
        3: "#90B960",
        4: '#22763F'
    }

    gdf_provinsi['Cluster'] = pd.to_numeric(gdf_provinsi['Cluster'], errors='coerce')
    cluster_labels = generate_cluster_category(n_cluster)

    gdf_provinsi['color'] = gdf_provinsi['Cluster'].map(color_map)

    label_mapping = {i: label for i, label in enumerate(cluster_labels)}

    unique_clusters = gdf_provinsi[['Cluster', 'color']].drop_duplicates()
    unique_clusters = unique_clusters.sort_values(by='Cluster')
    unique_clusters['Cluster'] = unique_clusters['Cluster'].map(label_mapping).fillna('Tidak ada data')
    unique_clusters['color'] = unique_clusters['color'].fillna('lightgrey')

    # Replace cluster numbers with labels
    gdf_provinsi['Cluster'] = gdf_provinsi['Cluster'].map(label_mapping).fillna('Tidak ada data').astype(str)

    gdf_provinsi['color'] = gdf_provinsi['color'].fillna('lightgrey') # Warna abu-abu untuk kabupaten/kota tanpa data

    # Simplify the geometries in gdf_provinsi, keeping the 'NAME_2' column
    gdf_provinsi_simplified = gdf_provinsi[['NAME_2', 'geometry']].simplify(tolerance=0.01, preserve_topology=True)

    # Convert the simplified GeoSeries to a GeoDataFrame
    if isinstance(gdf_provinsi_simplified, gpd.GeoSeries):
        gdf_provinsi_simplified = gpd.GeoDataFrame(geometry=gdf_provinsi_simplified, crs=gdf_provinsi.crs)
        gdf_provinsi_simplified['NAME_2'] = gdf_provinsi['NAME_2'] # Re-add NAME_2 if simplify dropped it

    # Merge the 'Cluster' and 'provinsi' columns back into the simplified GeoDataFrame
    gdf_provinsi_simplified = gdf_provinsi_simplified.merge(gdf_provinsi[['NAME_2', 'Cluster', 'provinsi', 'color']], on='NAME_2', how='left')

    # Replace NaN in 'Cluster' with 'no data'
    gdf_provinsi_simplified['Cluster'] = gdf_provinsi_simplified['Cluster'].fillna('no data').astype(str)
    gdf_provinsi_simplified.rename(columns={'NAME_2': 'Lokasi', 'provinsi': 'Provinsi'}, inplace=True)

    m = folium.Map(location=[-2.5, 118], zoom_start=zoom, tiles='cartodbpositron')

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
            tooltip=folium.GeoJsonTooltip(fields=['Cluster', 'Lokasi', 'Provinsi']),
        )
        gj.add_to(fg)
        fg.add_to(m)

    m.add_child(folium.LatLngPopup())

    folium.LayerControl(position='bottomleft', collapsed=False).add_to(m)
    folium_static(m, width=width, height=height)

def graph_comparison(data):
    nama_kolom = data.columns
    lp = nama_kolom[nama_kolom.str.startswith('Luas Panen')]

    tahun = []
    for j in range(len(lp)):
        res = lp[j].rsplit(' ', 1) # Pisah kata terakhir yang dipisah spasi
        tahun.append(res[1])

    fig = go.Figure()

    for i in range(len(data)):
        city_names = data['Lokasi'].iloc[i]
        luas_panen_values = data.loc[i, data.columns.str.startswith('Luas Panen')]

    fig.add_trace(go.Scatter(x=tahun, y=luas_panen_values, mode='lines+markers', name=city_names))

    fig.update_layout(
        title='Luas Panen per Kota',
        xaxis_title='Tahun',
        yaxis_title='Luas Panen (Ha)',
        hovermode='x unified'
    )

    fig.show()

def plot_panen_trends(df, default_metric="Produksi", default_order="Terbesar ▶️", default_n_locations=5):
    fitur = ['Luas', 'Produksi', 'Produktivitas']
    
    # Add search functionality
    with st.container():
        search_terms = st.text_input(
            "🔍 Bandingkan Lokasi (pisahkan dengan koma):",
            placeholder="Contoh: Aceh, Kalimantan, Jawa Barat",
            help="Masukkan nama lokasi yang ingin dibandingkan trennya",
            key="trend_search"
        )
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metric = st.selectbox(
                "Pilih Fitur:",
                options=fitur,
                index=fitur.index(default_metric),
                key="trend_metric"
            )
            
        with col2:
            # Disable radio button when searching
            sort_order = st.radio(
                "Urutan:",
                options=["Terbesar ▶️", "Terkecil ◀️"],
                index=0 if default_order == "Terbesar ▶️" else 1,
                horizontal=True,
                key="sort_order",
                disabled=bool(search_terms)  # Disabled when searching
            )
            
        with col3:
            max_locations = len(df)
            n_locations = st.slider(
                "Jumlah Lokasi:",
                min_value=1,
                max_value=min(20, max_locations),
                value=default_n_locations,
                key="num_locations",
                disabled=bool(search_terms)  # Disabled when searching
            )

    # Prepare data
    year_pattern = r'_(\d{4})$'
    metric_cols = [col for col in df.columns 
                if metric in col and re.search(year_pattern, col)]
    
    if not metric_cols:
        st.warning(f"⚠️ Data {metric} tidak ditemukan")
        return
    
    years = sorted([int(re.search(year_pattern, col).group(1)) for col in metric_cols])
    
    # Filter data based on search terms if provided
    if search_terms:
        search_list = [term.strip() for term in search_terms.split(',') if term.strip()]
        
        # Check which locations exist and which don't
        found_locations = []
        missing_locations = []
        
        for loc in search_list:
            matches = df[df['Lokasi'].str.contains(loc, case=False, na=False)]
            if not matches.empty:
                found_locations.extend(matches['Lokasi'].unique().tolist())
            else:
                missing_locations.append(loc)
        
        # Remove duplicates while preserving order
        seen = set()
        found_locations = [x for x in found_locations if not (x in seen or seen.add(x))]
        
        # Show warning for missing locations
        if missing_locations:
            st.warning(f"⚠️ Lokasi berikut tidak ditemukan: {', '.join(missing_locations)}")
        
        if not found_locations:
            st.error("❌ Tidak ada lokasi yang ditemukan dari pencarian Anda")
            return
            
        df_trend = df[df['Lokasi'].isin(found_locations)]
        
        # In search mode, we don't sort by average but keep original order of search terms
        df_trend['Match_Order'] = df_trend['Lokasi'].apply(
            lambda x: next((i for i, loc in enumerate(search_list) if loc.lower() in x.lower()), len(search_list))
        )
        df_trend = df_trend.sort_values('Match_Order')
    else:
        df_trend = df.copy()
        df_trend['Average'] = df_trend[metric_cols].mean(axis=1)
        ascending = sort_order.startswith("Terkecil")
        df_trend = df_trend.sort_values('Average', ascending=ascending).head(n_locations)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5 + len(df_trend)*0.3))
    palette = sns.color_palette("tab20", len(df_trend))
    
    for i, (_, row) in enumerate(df_trend.iterrows()):
        ax.plot(
            years,
            row[metric_cols].values,
            marker='o',
            linestyle='-',
            color=palette[i % len(palette)],
            label=f"{row['Lokasi']} ({row['Kategori']})",
            linewidth=2,
            markersize=6
        )
    
    # Style plot
    units = {"Luas": "Ha", "Produksi": "Ton", "Produktivitas": "kg/Ha"}.get(metric, "")
    
    # Custom title based on mode
    if search_terms:
        display_locs = [loc for loc in search_list if any(loc.lower() in found.lower() for found in found_locations)]
        title = f"TREN PERBANDINGAN: {', '.join(display_locs).upper()}\n"
        title += f"{metric} ({min(years)}-{max(years)})"
    else:
        title = f"{metric} Tren ({min(years)}-{max(years)})\n"
        title += f"{n_locations} {'Top' if not ascending else 'Bottom'} Lokasi"
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Tahun")
    ax.set_ylabel(f"{metric} ({units})")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Improved legend placement
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=8,
        title="Lokasi",
        title_fontsize=9
    )
    
    plt.tight_layout()
    
    if fig:
        img_buffer = get_fig_buffer(fig)
        
        filename = f"Tren_{metric}_{min(years)}-{max(years)}.png"
        
        # Create download button
        st.download_button(
            label="🖼️ Download PNG",
            data=img_buffer,
            file_name=filename,
            mime="image/png",
        )
    st.pyplot(fig)

def avg_features(dataframe):
    kolom_lp = dataframe.filter(regex='^Luas Panen').columns
    kolom_prod = dataframe.filter(regex='^Produksi').columns
    kolom_pty = dataframe.filter(regex='^Produktivitas').columns
    
    # Calculate the row-wise average using mean with axis=1
    dataframe['Luas Panen'] = dataframe[kolom_lp].mean(axis=1).round(2)
    dataframe['Produksi'] = dataframe[kolom_prod].mean(axis=1).round(2)
    dataframe['Produktivitas'] = dataframe[kolom_pty].mean(axis=1).round(2)
    return dataframe

# Menampilkan hasil clustering dgn k optimal berisi provinsi, lp, prod, pty, dan kategori
def clustering_results_dataframe(dataframe, bestcluster, cluster_labels):
    dataframe_baru = avg_features(dataframe)
    # dataframe_baru['Cluster'] = cluster_labels
    # st.dataframe(dataframe_baru, hide_index=True)
    cluster_category = generate_cluster_category(bestcluster)

    label_mapping = {i: label for i, label in enumerate(cluster_category)}
    dataframe_baru['Kategori'] = dataframe_baru['Cluster'].map(label_mapping)

    gdf_provinsi = merge_gdf(dataframe_baru)

    gdf_provinsi['Cluster'] = pd.to_numeric(gdf_provinsi['Cluster'], errors='coerce')

    unique_clusters = gdf_provinsi[['Cluster']].drop_duplicates()
    unique_clusters = unique_clusters.sort_values(by='Cluster')
    unique_clusters['Cluster'] = unique_clusters['Cluster'].fillna('Tidak ada data')

    # Replace cluster numbers with labels
    gdf_provinsi['Cluster'] = gdf_provinsi['Cluster'].fillna('Tidak ada data').astype(str)
    gdf_provinsi.rename(columns={'NAME_2': 'Lokasi', 'provinsi': 'Provinsi'}, inplace=True)

    dataframe_baru = dataframe_baru.merge(gdf_provinsi[['Provinsi', 'Lokasi']], on='Lokasi', how='inner')
    dataframe_baru = dataframe_baru[['Provinsi', 'Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas', 'Kategori']]

    gdf_provinsi.rename(columns={'Cluster': 'Kategori'}, inplace=True)
    dataframe_baru.loc[dataframe_baru['Lokasi'] == 'Banjar', 'Provinsi'] = 'Kalimantan Selatan'
    dataframe_baru.loc[dataframe_baru['Lokasi'] == 'Pesisir Barat', 'Provinsi'] = 'Lampung'
    dataframe_baru.loc[dataframe_baru['Lokasi'] == 'Buton Tengah', 'Provinsi'] = 'Sulawesi Tenggara'

    # st.dataframe(dataframe_baru, hide_index=True)
    return dataframe_baru

def show_prod_dan_lp(df):
    luas = df.loc[:, df.columns.str.startswith('Luas Panen')].describe()
    prod = df.loc[:, df.columns.str.startswith('Produksi')].describe()

    # Extract years from columns like "Produksi 2020"
    # tahun = [col.split(' ')[-1] for col in df.columns if col.startswith('Produksi')]
    tahun = [re.search(r'\d{4}', col).group() for col in df.columns if 'Produksi' in col]

    # Initialize figure
    fig = go.Figure()

    mean_luas = luas.loc['mean']
    mean_prod = prod.loc['mean']

    # Add traces
    fig.add_trace(go.Scatter(x=tahun, y=mean_luas.values,
                                mode='lines+markers', name=f'Luas Panen', marker_color='crimson'))
    fig.add_trace(go.Bar(x=tahun, y=mean_prod.values,
                        name=f'Produksi', marker_color='darkseagreen'))

    # Update layout
    fig.update_layout(
        title='Produksi dan Luas Panen Kacang Hijau per Tahun',
        title_font_size=20,
        xaxis_title='Tahun',
        yaxis_title='Nilai',
        hovermode='x unified',
        font=dict(
            family="Calibri, monospace"
        )
    )

    return fig

def evaluate(cluster_optimal, waktu, silhouette, dbi, cluster_option):
    col1, col2 = st.columns(2)
    with col1:
        if cluster_option == "Rentang cluster":
            st.metric(label=f"Cluster Optimal", value=f"{cluster_optimal}")
        else:
            st.metric(label=f"Jumlah Cluster", value=f"{cluster_optimal}")
        st.write("")
        st.metric(label="Waktu Komputasi", value=f"{round(waktu['Waktu Komputasi (detik)'].mean(), 4)} detik")
    with col2:
        st.metric(label="Silhouette Score", value=f"{silhouette}")
        st.write("")
        st.metric(label="Davies-Bouldin Index", value=f"{dbi}")

def clustering_sample(df):
    df = columns_to_drop(df)
    dataframe = data_selection (df)
    df_copy = dataframe.copy()
    dataframe = dataframe.reset_index() # Reset index 'Lokasi'
    # dataframe = dataframe.replace(np.nan, 0)
    bar_chart = show_prod_dan_lp(df)
    df_copy = handle_null (df_copy)

    df_array = df_copy.drop(['Lokasi'], axis=1)
    df_array = df_array.copy().values
    
    metode = "Agglomerative Hierarchical Clustering"
    df_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, _, bestcluster_ahc, labels_ahc = AHC(df_array, range(2,6), "single")

    df_copy['Cluster'] = labels_ahc
    dataframe['Cluster'] = labels_ahc

    df_copy = sort_cluster(df_copy)
    df_result = penyesuaian(df_copy)
    dataframe = penyesuaian(dataframe)
    
    dataframe_baru = clustering_results_dataframe(df_result, bestcluster_ahc, labels_ahc)
    dataframe_lama = clustering_results_dataframe(dataframe, bestcluster_ahc, labels_ahc)

    cols = st.columns(2, gap="small", vertical_alignment="top")
    with cols[0]:
        map_folium(df_result, bestcluster_ahc)

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
        fig.update_layout(title_text="Kontribusi Produksi Kacang Hijau 2010 - 2024")
        st.plotly_chart(fig, use_container_width=True)

    with cols[1]:
        st.dataframe(dataframe_lama, hide_index=True)
        st.plotly_chart(bar_chart)

def add_cluster_to_df (df, cluster_labels):
    df['Cluster'] = cluster_labels
    df = sort_cluster(df)
    df = penyesuaian(df)
    return df

def show_map(df, cluster_labels, cluster_optimal, zoom=4, width=1500, height=400):
    df = add_cluster_to_df(df, cluster_labels)
    map_folium(df, cluster_optimal, zoom=zoom, width=width, height=height)

def cluster_and_category_result(df, cluster_labels, cluster_optimal, nama_kolom_kategori, nama_kolom_cluster):
    df = add_cluster_to_df(df, cluster_labels)
    # df = categorize_df(df, cluster_optimal, f'{nama_kolom_kategori}', f'{nama_cluster}')
    df = df.rename(columns={'Cluster': f'{nama_kolom_cluster}'})
    kategori = generate_cluster_category(cluster_optimal)
    label_mapping = {i: label for i, label in enumerate(kategori)}
    df[f'{nama_kolom_kategori}'] = df[f'{nama_kolom_cluster}'].map(label_mapping)
    return df

def group_df_by_cluster (df, kategori):
    cluster = df.groupby(kategori).size().reset_index(name='Jumlah')
    cluster.columns = ['Cluster', 'Jumlah']
    return cluster

def show_n_cluster(df, kategori, metode):
    # cluster = group_df_by_cluster (df, kategori)
    cluster = df.groupby(kategori).size().reset_index(name='Jumlah')
    cluster.columns = ['Cluster', 'Jumlah']

    # colors = ['gold', 'darkorange', 'mediumturquoise', 'crimson', 'lightgreen']
    colors = ['#BE2A3E', '#F88F4D', '#F4D166', "#90B960", '#22763F']

    fig = px.pie(cluster, values='Jumlah', names='Cluster', title=f'Jumlah Anggota Cluster {metode}')
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=17,
                    marker=dict(colors=colors, line=dict(color='#000000', width=1)))
    fig.update_layout(legend_title='Cluster',
                        title_font_size=20)
    st.plotly_chart(fig)

def compare_cluster(df, nama_kolom, height=900, direction='vertical'):
    fitur = ['Luas Panen (Hektar)', 'Produksi (Ton)', 'Produktivitas (Kuintal/Ha)']
    prefix_fitur = ['Luas Panen 2', 'Produksi 2', 'Produktivitas 2']

    cluster_colors = {
        0: '#BE2A3E',
        1: '#F88F4D',
        2: '#F4D166',
        3: "#90B960",
        4: '#22763F'
    }

    if direction == 'vertical':
        rows = 3
        cols = 1
        y = 1.1
    elif direction == 'horizontal':
        rows = 1
        cols = 3
        y = 1.3
    else:
        rows = 3
        cols = 1
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=fitur)
    n=0
    for prefix in prefix_fitur:
        n+=1
        # Menyimpan kolom yg memiliki nama "Luas Panen 2"
        feature_columns = [col for col in df.columns if f'{prefix}' in col]

        # mneyimpan rata-rata mean berdasarkan golongan cluster
        feature_mean = df.groupby(nama_kolom)[feature_columns].mean()

        # Menyimpan array tahun
        tahun = [re.search(r'\d{4}', col).group() for col in df.columns if f'{prefix}' in col]
        feature_mean = feature_mean.reset_index().reset_index(drop=True)

        unique_categories = feature_mean[nama_kolom]
        feature_mean = feature_mean.set_index(nama_kolom).transpose()
        for category in unique_categories:
            color = cluster_colors[category]
            if direction == 'vertical':
                fig.add_trace(go.Scatter(
                    x=tahun, 
                    y=feature_mean[category], 
                    mode='lines+markers', 
                    name=f"Cluster {category}",
                        marker_color=color,
                        showlegend=True if n == 3 else False
                    ), row=n, col=1)
            elif direction == 'horizontal':
                fig.add_trace(go.Scatter(
                    x=tahun, 
                    y=feature_mean[category], 
                    mode='lines+markers', 
                    name=f"Cluster {category}",
                        marker_color=color,
                        showlegend=True if n == 3 else False
                    ), row=1, col=n)
            else:
                st.write("Error creating chart.")
        
        fig.update_layout(
            height=height,
            showlegend=True,
            legend=dict(y=y, orientation='h')
        )
        
        # Update x-axis to show gridlines and customize their appearance
        fig.update_xaxes(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=1,
            griddash='solid',
            tickmode='linear'
        )

        # Update y-axis to show gridlines and customize their appearance
        fig.update_yaxes(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=1,
            griddash='solid'
        )
    
    st.plotly_chart(fig)

def plot_data_cluster(df, label_clusters, metode, pca_components=2):
    # Ensure the DataFrame is in the correct format (only numeric columns)
    X = df.select_dtypes(include=[np.number]).values

    y = label_clusters

    # Reduce dimensions to 2D if necessary using PCA
    if X.shape[1] > 2:
        pca = PCA(n_components=pca_components)
        X_pca = pca.fit_transform(X)
    else:
        X_pca = X  # If data is already 2D, just use it as is

    # Create a DataFrame for Plotly plotting
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    df_pca['Cluster'] = y.astype(str)

    # Sort by cluster for consistent legend order
    df_pca = df_pca.sort_values(by='Cluster')

    # Define the new color map
    scatter_color = {
        '0': '#BE2A3E',
        '1': '#F88F4D',
        '2': '#F4D166',
        '3': "#90B960",
        '4': '#22763F'
    }

    # fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
    #                  labels={'PC1': 'x', 'PC2': 'y'},
    #                  color_discrete_map=scatter_color, height=400,
    #                  title=f"Ruang Fitur Cluster dengan {metode}"
    #                 )

    # # Adjust opacity and size of the markers (dots)
    # fig.update_traces(marker=dict(opacity=0.6, size=10))  # 0.6 means 60% opacity (more transparent)
    # fig.update_layout(title_font_size=20)

    # # Show the figure
    # st.plotly_chart(fig)
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                    labels={'PC1': 'x', 'PC2': 'y'},
                    color_discrete_map=scatter_color, height=400,
                    title=f"Ruang Fitur Cluster dengan {metode}"
                    )

    # Adjust opacity and size of the markers (dots)
    fig.update_traces(marker=dict(opacity=0.6, size=10))  # 0.6 means 60% opacity (more transparent)

    # Update the layout for gridlines
    fig.update_layout(
        title_font_size=20,
        xaxis=dict(
            showgrid=True,             # Show gridlines on x-axis
            gridcolor='lightgrey',     # Color of gridlines
            gridwidth=1,               # Thickness of gridlines
        ),
        yaxis=dict(
            showgrid=True,             # Show gridlines on y-axis
            gridcolor='lightgrey',     # Color of gridlines
            gridwidth=1,               # Thickness of gridlines
        )
    )

    # Display the figure
    st.plotly_chart(fig)


