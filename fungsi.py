from sklearn.cluster import BisectingKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from streamlit_folium import folium_static
import folium
import time
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import re
import geopandas as gpd
import datetime
import streamlit as st
import pandas as pd
import numpy as np

COLOR_MAP = {
    0: "#D13035",
    1: "#F7D756",
    2: '#4BB35C',
    3: "#6CD9D5",
    4: '#1965B0'
}

def validate_columns_and_data(data):
    required_columns = ['Lokasi', 'Luas Panen ', 'Produksi ', 'Produktivitas ']
    # Cek jika kolom ditemukan ('Luas Panen', 'Produksi', etc.)
    column_rename_map = {
        'Luas Panen ': 'Luas Panen [tahun]',
        'Produksi ': 'Produksi [tahun]',
        'Produktivitas ': 'Produktivitas [tahun]'
    }
    
    missing_columns = []
    for col in required_columns:
        matching_columns = [c for c in data.columns if c.startswith(col)]
        if not matching_columns:
            mapped_name = column_rename_map.get(col, col)  # Use the mapped name if available
            missing_columns.append(mapped_name)
    
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
    kolom_lokasi = data['Lokasi']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.drop(columns=['Lokasi'], axis=1).values)

    data_scaled = pd.DataFrame(scaled_data, columns=data.drop(columns=['Lokasi'], axis=1).columns, index=data.index) # Mengubah hasil scaled_data (array) kembali menjadi DataFrame
    data_scaled['Lokasi'] = kolom_lokasi
    return data_scaled

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
            num_cluster = i
            labels = cluster_labels
            db_index_temp = dbi_avg

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

def linechart_evaluation (df_bkmeans, df_ahc): # untuk evaluasi perbandingan kedua metode
    method_dataframe = pd.concat([df_bkmeans, df_ahc], ignore_index=False)
    method_dataframe = method_dataframe.reset_index() # Reset index

    fig_silhouette = px.line(
        method_dataframe,
        x='Cluster',
        y='Silhouette Score',
        color='Metode',
        markers=True,
        # title='Silhouette Score',
        color_discrete_map={
            'Bisecting K-Means': '#BD4B46',
            'AHC': '#8D957E'
        }
    )
    fig_silhouette.update_layout(
        title=f'Silhouette Score',
        title_font_size=20
    )

    # Plot DBI Score
    fig_dbi = px.line(
        method_dataframe,
        x='Cluster',
        y='DBI Score',
        color='Metode',
        markers=True,
        # title='DBI Score',
        color_discrete_map={
            'Bisecting K-Means': '#BD4B46',
            'AHC': '#8D957E'
        }
    )
    fig_dbi.update_layout(
        title=f'DBI Score',
        title_font_size=20
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

        color = COLOR_MAP.get(i, '#808080')  # Default to gray if not found in COLOR_MAP

        fig.add_trace(go.Bar(
            x=ith_vals,
            y=np.arange(y_lower, y_upper),
            orientation='h',
            marker=dict(color=color),
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
        title=f'Visualisasi Silhouette {metode} {n_clusters} Cluster',
        title_font_size=20,
        xaxis=dict(title='Koefisien Silhouette', range=[-0.2, 1]),
        yaxis=dict(title='Label Cluster', showticklabels=False),
        # height=450,
        # margin=dict(l=50, r=50, t=80, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

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

def sort_cluster (data, nama_kolom='Cluster'): # Sort cluster based on mean
    mean = {}
    unique_clusters = np.sort(data[f'{nama_kolom}'].unique()) # Count unique clusters
    for i in range (len(unique_clusters)):
        cluster = data[data[f'{nama_kolom}'] == unique_clusters[i]]
        mean[i] = cluster.describe().loc['mean'].mean()

    # descending sort berdasarkan mean
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
        "Kota Tidore Kepulauan": "Tidore Kepulauan", "Kota Tual": "Tual", "Banjar": "Kabupaten Banjar", "Yapen Waropen": "Kepulauan Yapen"
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
    # data_clustering = data[['Lokasi', 'Cluster']].rename(columns={'Lokasi': 'provinsi'}) # Ganti 'Lokasi' menjadi 'provinsi' untuk menyesuaikan
    if 'Cluster' in data.columns:
        data_clustering = data[['Lokasi', 'Cluster']].rename(columns={'Lokasi': 'provinsi'})
    else:
        data_clustering = data[['Lokasi']].rename(columns={'Lokasi': 'provinsi'})
    gdf_provinsi = gdf.dissolve(by='NAME_2').reset_index() # Menggabungkan data kabupaten ke provinsi
    gdf_provinsi = gdf_provinsi.merge(data_clustering, left_on='NAME_2', right_on='provinsi', how='left') # Menggabungkan dengan data clustering
    gdf_provinsi['provinsi'] = gdf_provinsi['NAME_1']
    
    return gdf_provinsi

def map_folium(data, n_cluster, zoom=4, width=1500, height=400):
    with st.spinner("Memuat peta..."):
        gdf_provinsi = merge_gdf(data)

        gdf_provinsi['Cluster'] = pd.to_numeric(gdf_provinsi['Cluster'], errors='coerce')
        cluster_labels = generate_cluster_category(n_cluster)

        gdf_provinsi['color'] = gdf_provinsi['Cluster'].map(COLOR_MAP)

        label_mapping = {i: label for i, label in enumerate(cluster_labels)}

        unique_clusters = gdf_provinsi[['Cluster', 'color']].drop_duplicates()
        unique_clusters = unique_clusters.sort_values(by='Cluster')
        unique_clusters['Cluster'] = unique_clusters['Cluster'].map(label_mapping).fillna('Tidak ada data')
        unique_clusters['color'] = unique_clusters['color'].fillna('lightgrey')

        # Ganti nilai cluster numerik dengan label kategori
        gdf_provinsi['Cluster'] = gdf_provinsi['Cluster'].map(label_mapping).fillna('Tidak ada data').astype(str)

        gdf_provinsi['color'] = gdf_provinsi['color'].fillna('lightgrey') # Warna abu-abu untuk kabupaten/kota tanpa data

        # Menyederhanakan geometries di gdf_provinsi, simpan kolom 'NAME_2'
        gdf_provinsi_simplified = gdf_provinsi[['NAME_2', 'geometry']].simplify(tolerance=0.01, preserve_topology=True)

        # Ubah kembali ke GeoDataFrame jika hasilnya GeoSeries
        if isinstance(gdf_provinsi_simplified, gpd.GeoSeries):
            gdf_provinsi_simplified = gpd.GeoDataFrame(geometry=gdf_provinsi_simplified, crs=gdf_provinsi.crs)
            gdf_provinsi_simplified['NAME_2'] = gdf_provinsi['NAME_2'] # Tambahkan kolom 'NAME_2' kembali

        # Gabungkan kolom 'Cluster', 'provinsi', dan 'color' ke gdf_provinsi_simplified
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
                    'color': "#484848",
                    'weight': 0.3,
                    'fillOpacity': 0.8,
                },
                tooltip=folium.GeoJsonTooltip(fields=['Cluster', 'Lokasi', 'Provinsi']),
            )
            gj.add_to(fg)
            fg.add_to(m)

        m.add_child(folium.LatLngPopup())

        folium.LayerControl(position='bottomleft', collapsed=False).add_to(m)
        folium_static(m, width=width, height=height)

def show_map_explanation():
    with st.expander("Lihat penjelasan"):
        st.write('''
            Peta ini menggambarkan wilayah di Indonesia yang sudah dibagi berdasarkan jumlah cluster yang Anda input.
            Label tinggi / rendah pada peta adalah berdasarkan tingkat produksi.
        ''')

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

def pie_kontribusi(df, min_tahun, max_tahun):
    pie_df = df.groupby('Provinsi')['Produksi'].sum().reset_index()
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
        title_text=f"Kontribusi Produksi Kacang Hijau {min_tahun}-{max_tahun}",
        title_font_size=20
    )
    st.plotly_chart(fig)

def plot_tren_panen(df, default_metric="Produksi", default_order="Terbesar", default_n_locations=5):
    fitur = ['Luas Panen', 'Produksi', 'Produktivitas']
    with st.container():
        col1, col2, col3 = st.columns(3, gap='medium')
        with col1:
            indikator = st.selectbox(
                "Pilih indikator :",
                options=fitur,
                index=fitur.index(default_metric),
                key="indikator"
            )
        with col2:
            # Disable radio button when searching
            sort_order = st.radio(
                "Urutan :",
                options=["Terbesar", "Terkecil"],
                index=0 if default_order == "Terbesar" else 1,
                horizontal=True,
                key="sort_order"
            )
        with col3:
            max_locations = len(df)
            n_locations = st.slider(
                "Jumlah Lokasi:",
                min_value=1,
                max_value=min(13, max_locations),
                value=default_n_locations,
                key="num_locations"
            )

    year_pattern = r' (\d{4})$'
    metric_cols = [col for col in df.columns 
                if indikator in col and re.search(year_pattern, col)]
    
    if not metric_cols:
        st.warning(f"Indikator {indikator} tidak ditemukan")
        return
    
    tahun = sorted([int(re.search(year_pattern, col).group(1)) for col in metric_cols])
    
    df_tren = df.copy()

    sort_bool = sort_order.startswith("Terkecil")
    df_tren = df.copy().sort_values(indikator, ascending=sort_bool).head(n_locations)

    df_tren = df_tren.reset_index().drop('index', axis=1)

    satuan = {"Luas Panen": "Ha", "Produksi": "Ton", "Produktivitas": "kg/Ha"}.get(indikator, "")
        
    fig = go.Figure()
    for i in range(len(df_tren)):
        city_names = df_tren['Lokasi'].iloc[i]
        feature_values = df_tren.loc[i, df_tren.columns.str.startswith(f'{indikator} ')]

        fig.add_trace(go.Scatter(x=tahun, y=feature_values, mode='lines+markers', name=city_names))

    fig.update_layout(
        title=f'{indikator} {sort_order} per Kabupaten / Kota',
        xaxis_title='Tahun',
        yaxis_title=f'{indikator} ({satuan})',
        hovermode='x unified'
    )

    st.plotly_chart(fig, theme=None)

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

def add_provinsi_to_df(dataframe):
    gdf_provinsi = merge_gdf(dataframe)
    # gdf_provinsi['Cluster'] = pd.to_numeric(gdf_provinsi['Cluster'], errors='coerce')

    # unique_clusters = gdf_provinsi[['Cluster']].drop_duplicates()
    # unique_clusters = unique_clusters.sort_values(by='Cluster')
    # unique_clusters['Cluster'] = unique_clusters['Cluster'].fillna('Tidak ada data')

    # Replace cluster numbers with labels
    # gdf_provinsi['Cluster'] = gdf_provinsi['Cluster'].fillna('Tidak ada data').astype(str)
    gdf_provinsi.rename(columns={'NAME_2': 'Lokasi', 'provinsi': 'Provinsi'}, inplace=True)

    dataframe = dataframe.merge(gdf_provinsi[['Provinsi', 'Lokasi']], on='Lokasi', how='inner')

    # dataframe.loc[dataframe['Lokasi'] == 'Banjar', 'Provinsi'] = 'Kalimantan Selatan'
    # dataframe.loc[dataframe['Lokasi'] == 'Pesisir Barat', 'Provinsi'] = 'Lampung'
    # dataframe.loc[dataframe['Lokasi'] == 'Buton Tengah', 'Provinsi'] = 'Sulawesi Tenggara'
    # dataframe.loc[dataframe['Lokasi'] == 'Kota Banjar', 'Provinsi'] = 'Jawa Barat'

    return dataframe

def show_prod_dan_lp(df):
    luas = df.loc[:, df.columns.str.startswith('Luas Panen')].describe()
    prod = df.loc[:, df.columns.str.startswith('Produksi')].describe()

    # Extract years from columns like "Produksi 2020"
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
            st.metric(label=f"Cluster Optimal", value=f"{cluster_optimal}", help='Data dicluster berdasarkan Silhouette tertinggi dan Davies-Bouldin Index terendah.')
        else:
            st.metric(label=f"Jumlah Cluster", value=f"{cluster_optimal}", help='Berapa banyak cluster data dibagi.')
        st.write("")
        st.metric(label="Waktu Komputasi", value=f"{round(waktu['Waktu Komputasi (detik)'].mean(), 4)} detik", help='Rata-rata waktu yang dibutuhkan untuk melakukan proses clustering.')
    with col2:
        st.metric(label="Silhouette Score", value=f"{silhouette}", help='Nilai Silhouette Score berkisar antara -1 hingga 1. Nilai yang lebih tinggi menunjukkan bahwa objek lebih cocok dengan cluster mereka sendiri daripada dengan cluster tetangga.')
        st.write("")
        st.metric(label="Davies-Bouldin Index", value=f"{dbi}", help='Nilai Davies-Bouldin Index (DBI) yang lebih rendah menunjukkan bahwa cluster terpisah dengan baik satu sama lain dan lebih kompak.')
    st.write("")

def add_cluster_to_df (df, cluster_labels):
    df['Cluster'] = cluster_labels
    df = sort_cluster(df)
    df = penyesuaian(df)
    return df

def show_map(df, cluster_labels, cluster_optimal, zoom=4, width=1500, height=400):
    # df = add_cluster_to_df(df, cluster_labels)
    df['Cluster'] = cluster_labels
    df = sort_cluster(df)
    df = penyesuaian(df)
    map_folium(df, cluster_optimal, zoom=zoom, width=width, height=height)

def cluster_and_category_result(df, cluster_labels, cluster_optimal, nama_kolom_kategori, nama_kolom_cluster):
    # df = add_cluster_to_df(df, cluster_labels)
    # df = categorize_df(df, cluster_optimal, f'{nama_kolom_kategori}', f'{nama_cluster}')
    df['Cluster'] = cluster_labels
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
    cluster = df.groupby(kategori).size().reset_index(name='Jumlah')
    cluster.columns = ['Cluster', 'Jumlah']

    # colors = ['gold', 'darkorange', 'mediumturquoise', 'crimson', 'lightgreen']
    # colors = ['#BE2A3E', '#F88F4D', '#F4D166', "#90B960", '#22763F']
    # colors = ["#D13035", "#F7D756", '#4BB35C', "#6CD9D5", '#1965B0']
    colors = cluster['Cluster'].map(COLOR_MAP).fillna('#808080').tolist()

    fig = px.pie(cluster, values='Jumlah', names='Cluster', title=f'Jumlah Anggota Cluster {metode}')
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=17,
                    marker=dict(colors=colors, line=dict(color='#000000', width=1)))
    fig.update_layout(legend_title='Cluster',
                        title_font_size=20)
    st.plotly_chart(fig)

def compare_cluster(df, nama_kolom, height=900, direction='vertical'):
    fitur = ['Luas Panen (Hektar)', 'Produksi (Ton)', 'Produktivitas (Kuintal/Ha)']
    prefix_fitur = ['Luas Panen 2', 'Produksi 2', 'Produktivitas 2']

    # cluster_colors = {
    #     0: '#BE2A3E',
    #     1: '#F88F4D',
    #     2: '#F4D166',
    #     3: "#90B960",
    #     4: '#22763F'
    # }

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
            color = COLOR_MAP[category]
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
        
        # Tambah gridlines pada x-axis
        fig.update_xaxes(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=1,
            griddash='solid',
            tickmode='linear'
        )

        # Tambah gridlines pada y-axis
        fig.update_yaxes(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=1,
            griddash='solid'
        )
    
    st.plotly_chart(fig)

def plot_data_cluster(df, label_clusters, metode, pca_components=2):
    # dataframe hanya berisi fitur numerik untuk PCA
    X = df.select_dtypes(include=[np.number]).values

    y = label_clusters

    # Memperkecil dimensi data ke 2D menggunakan PCA
    if X.shape[1] > 2:
        pca = PCA(n_components=pca_components)
        X_pca = pca.fit_transform(X)
    else:
        X_pca = X  # Kalau sudah 2D, tidak perlu PCA

    # df untuk plotting
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    df_pca['Cluster'] = y.astype(str)

    # Sort by cluster
    df_pca = df_pca.sort_values(by='Cluster')

    # scatter_color = {
    #     '0': '#BE2A3E',
    #     '1': '#F88F4D',
    #     '2': '#F4D166',
    #     '3': "#90B960",
    #     '4': '#22763F'
    # }
    color_map_str = {str(k): v for k, v in COLOR_MAP.items()}

    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                    labels={'PC1': 'x', 'PC2': 'y'},
                    color_discrete_map=color_map_str, height=400,
                    title=f"Ruang Fitur Cluster dengan {metode}"
                    )

    # Set marker dots
    fig.update_traces(marker=dict(opacity=0.6, size=10))  # 0.6 means 60% opacity (more transparent)

    # Tambah gridlines
    fig.update_layout(
        title_font_size=20,
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=1,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=1,
        )
    )

    st.plotly_chart(fig)

def heatmap_corr(df):
    thecorr = df.corr().round(3)
    text_labels = thecorr.values.astype(str)

    heat = go.Heatmap(
        z=thecorr,
        x=thecorr.columns.values,
        y=thecorr.columns.values,
        zmin=-0.25,  # Sets the lower bound of the color domain
        zmax=1,
        xgap=1,  # Sets the horizontal gap (in pixels) between bricks
        ygap=1,
        colorscale='orrd',
        text=text_labels,
        texttemplate="%{text}"
    )

    layout = go.Layout(
        title_text='Matriks Korelasi Fitur Data Hasil Panen Kacang Hijau',
        width=600,
        height=500,
        yaxis_autorange='reversed',
        xaxis=dict(
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            tickfont=dict(size=16)
        ),
        margin=dict(t=50, b=50, l=50, r=50)
    )

    fig = go.Figure(data=[heat], layout=layout)
    fig.update_layout(
        title_font_size=20,
        font=dict(
            family="Calibri, monospace",
            size=18
        )
    )

    # Render the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def proses_clustering(df, metode, cluster_labels, cluster_optimal, cluster_option,
                      df_metode, dfwaktu, silhouette_df, dbi_df, df_temp):
    df['Cluster'] = cluster_labels
    df = sort_cluster(df)
    
    cluster_category = generate_cluster_category(cluster_optimal) # beri kategori string
    label_mapping = {i: label for i, label in enumerate(cluster_category)}
    df['Kategori'] = df['Cluster'].map(label_mapping)

    df = avg_features(df) # rata-rata fitur per lokasi

    # ANALISIS ALGORITMA CLUSTERING
    if cluster_option == "Jumlah cluster":
        subcol = st.columns([13,13], gap="medium", vertical_alignment='top', border=True)
        with subcol[0]:
            st.write(f"#### {metode}")
            st.write("")
            evaluate(cluster_optimal, dfwaktu, silhouette_df, dbi_df, cluster_option)
        
        with subcol[1]:
            df_array = df.drop(['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas', 'Cluster', 'Kategori'], axis=1).values
            visualize_silhouette(df_array, df['Cluster'], cluster_optimal, silhouette_df, metode)
        
    if cluster_option == "Rentang cluster":
        st.write(f"### {metode}")
        evaluate(cluster_optimal, dfwaktu, silhouette_df, dbi_df, cluster_option)
        subcol = st.columns([13,13], gap="medium", vertical_alignment='top', border=True)
        with subcol[0]:
            st.write("##### Hasil Silhouette dan Davies-Bouldin Index")
            fig_evaluate(df_metode, metode)
        with subcol[1]:
            df_array = df.drop(['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas', 'Cluster', 'Kategori'], axis=1).values
            visualize_silhouette(df_array, df['Cluster'], cluster_optimal, silhouette_df, metode)

    # PERBANDINGAN FITUR SETIAP CLUSTER
    # st.subheader("Hasil Clustering", divider=True, anchor="hasil_clustering")
    st.write("##### Perbandingan Fitur Setiap Cluster")
    compare_cluster(df, 'Cluster', height=400, direction='horizontal')
    
    st.write("##### Tabel Data Hasil Clustering")
    df_temp = cluster_and_category_result(df_temp, df['Cluster'], cluster_optimal, 'Kategori', 'Cluster')
    df_temp = avg_features(df_temp)
    st.dataframe(df_temp[['Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas', 'Kategori', 'Cluster']], hide_index=True)
    
    st.write("##### Pemetaan Tingkat Produksi Kacang Hijau")
    show_map(df, cluster_labels, cluster_optimal, zoom=5, height=500)
    show_map_explanation()

    # DETAIL HASIL CLUSTER
    subcol = st.columns(2, gap="medium", border=True)
    with subcol[0]:
        plot_data_cluster(df, df['Cluster'], metode)
    with subcol[1]:
        show_n_cluster(df, df["Cluster"], metode)

def greet():
    currentTime = datetime.datetime.now()

    if currentTime.hour < 12:
        st.subheader('Selamat pagi!')
    elif 12 <= currentTime.hour < 18:
        st.subheader('Selamat siang!')
    else:
        st.subheader('Selamat malam!')

def proses_clustering_perbandingan(linkage, df_copy, df_temp, df_array, n_cluster, cluster_option):
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
    df_copy = sort_cluster(df_copy, 'Cluster BKM')
    df_copy = sort_cluster(df_copy, 'Cluster AHC')

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
    df_temp = cluster_and_category_result(df_temp, df_copy['Cluster BKM'], bestcluster_bkmeans, 'Kategori (Bisecting K-Means)', 'Cluster BKM')
    df_temp = cluster_and_category_result(df_temp, df_copy['Cluster AHC'], bestcluster_ahc, 'Kategori (Agglomerative Clustering)', 'Cluster AHC')
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

    # DETAIL RUANG HASIL CLUSTER
    subcol = st.columns(2, gap="medium", border=True)
    with subcol[0]:
        plot_data_cluster(df_copy, df_copy['Cluster BKM'], metode1)
    with subcol[1]:
        plot_data_cluster(df_copy, df_copy["Cluster AHC"], metode2)