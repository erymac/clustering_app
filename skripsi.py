from sklearn.cluster import BisectingKMeans, AgglomerativeClustering
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.datasets import load_nfl
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import folium
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from datetime import date
from io import StringIO
import os

import streamlit as st
import pandas as pd
import numpy as np

def preprocess_data(data):
    # df = pd.read_csv('Kacang Hijau.csv', sep=';')
    df = data.replace(0, np.nan)

    column_null = df.isna().sum()/len(df) * 100
    columns_to_drop = column_null[column_null == 100].index
    df.drop(columns_to_drop, axis=1, inplace=True)

    # Drop data with NaNs > 30% and < 100%
    df_clean = df.set_index('Nama Kota') # Exclude 'Nama Kota' from the calculation
    row_null_pct = df_clean.transpose().isna().sum()/len(df_clean.transpose()) * 100 # Calculate % of missing values per row
    rows_to_drop = row_null_pct[(row_null_pct > 35) & (row_null_pct < 100)].index # Identify rows where NaNs > 30% and < 100%
    df_clean = df_clean.drop(index=rows_to_drop) # Drop those rows
    df_clean = df_clean.reset_index() # Reset index to restore 'Nama Kota' as a column
    df = df_clean.copy()

    # Handle missing values
    df = df.ffill()
    df = df.bfill()

    df.columns = df.columns.astype(str) # change columns type

    df_copy = df.copy()
    df_copy.drop(['Nama Kota'], axis=1, inplace=True)
    return df_copy

def normalize(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)

    data_scaled = pd.DataFrame(scaled_data, columns=data.columns, index=data.index) # Mengubah hasil scaled_data (array) kembali menjadi DataFrame
    data.update(data_scaled) # Memperbarui DataFrame asli
    return data

def BKMeans(data): # Fungsi metode Bisecting K-Means mengembalikan skor silhouette, DBI, dan waktu komputasi/pelatihan
    silhouette_temp = 0
    silhouette_bkmeans = []
    dbi_bkmeans = []
    waktu_bkmeans = []
    cluster_bkmeans = []
    waktu_avg = 0
    db_index_temp = float('inf')
    silhouette_avg_avg = 0
    dbi_avg = float('inf')
    for i in range(2, 11):
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
            silhouette_temp = silhouette_avg
            waktu_temp = waktu
            num_cluster = i

        if dbi_avg < db_index_temp:
            db_index_temp = db_index
            db_waktu_temp = waktu
            db_num_cluster = i

    df_bkmeans = pd.DataFrame(columns=['Cluster', 'Silhouette Score', 'DBI Score'])
    df_bkmeans['Cluster'] = cluster_bkmeans
    df_bkmeans['Silhouette Score'] = silhouette_bkmeans
    df_bkmeans['DBI Score'] = dbi_bkmeans
    df_bkmeans.set_index(['Cluster'], inplace=True)

    dfwaktu_bkmeans = pd.DataFrame(columns=['Cluster', 'Waktu Komputasi'])
    dfwaktu_bkmeans['Cluster'] = cluster_bkmeans
    dfwaktu_bkmeans['Waktu Komputasi'] = waktu_bkmeans
    dfwaktu_bkmeans.set_index(['Cluster'], inplace=True)
    return df_bkmeans, dfwaktu_bkmeans

def AHC(data): # Fungsi metode AHC mengembalikan skor silhouette, DBI, dan waktu komputasi/pelatihan
    silhouette_ahc = []
    dbi_ahc = []
    waktu_ahc = []
    exp_ahc = []
    cluster_ahc = []
    silhouette_temp = 0
    db_index_temp = float('inf')
    silhouette_avg_avg = 0
    waktu_avg = 0
    dbi_avg = float('inf')
    for i in range(2, 11):
        silhouette_total = 0
        dbi_total = 0
        waktu_total = 0
        cluster_ahc.append(i)
        for j in range (5):
            ahc_clusterer = AgglomerativeClustering(n_clusters=i, metric='euclidean', linkage='average')
            print (ahc_clusterer)

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
            silhouette_temp = silhouette_avg
            waktu_temp = waktu
            num_cluster = i
            labels = cluster_labels
            clusterer = ahc_clusterer

        if dbi_avg < db_index_temp:
            db_index_temp = db_index
            db_waktu_temp = waktu
            db_num_cluster = i
    
    df_ahc = pd.DataFrame(columns=['Cluster', 'Silhouette Score', 'DBI Score'])
    df_ahc['Cluster'] = cluster_ahc
    df_ahc['Silhouette Score'] = silhouette_ahc
    df_ahc['DBI Score'] = dbi_ahc

    dfwaktu_ahc = pd.DataFrame(columns=['Cluster', 'Waktu Komputasi'])
    dfwaktu_ahc['Cluster'] = cluster_ahc
    dfwaktu_ahc['Waktu Komputasi'] = waktu_ahc

    df_ahc.set_index(['Cluster'],inplace=True)
    dfwaktu_ahc.set_index(['Cluster'],inplace=True)
    return df_ahc, dfwaktu_ahc

#Home Page
def home():
    multi = '''Klasterisasi Data Kacang Hijau di Indonesia
    '''
    st.markdown('---')
    st.markdown(f"<h1 style='text-align: center; color: #111A19;'>{multi}</h1>", unsafe_allow_html=True)
    st.write("Kumpulan data luas panen, produksi, dan produktivitas sektor pangan dapat diperoleh dari situs [Basis Data Statistik Pertanian (BDSP)](https://bdsp2.pertanian.go.id/bdsp/id/lokasi). Pastikan file yang diunggah berbentuk csv / xlsx / xls.")

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[-1]
        if file_ext == '.csv':
            dataframe = pd.read_csv(uploaded_file, sep=';')
        elif file_ext == '.xlsx':
            dataframe = pd.read_excel(uploaded_file, engine='openpyxl')
        elif file_ext == '.xls':
            dataframe = pd.read_excel(uploaded_file, engine='xlrd')
        else:
            st.error("Jenis file tidak didukung. Harap unggah file Excel (.csv / .xls / .xlsx).")
        # st.dataframe(dataframe)

    option = st.selectbox(
        "Metode klasterisasi apa yang ingin digunakan?",
        ("Bisecting K-Means", "Agglomerative Hierarchical Clustering (AHC)", "Bisecting K-Means dan AHC"),
        index=None,
        placeholder="Pilih metode klasterisasi...",
    )
    
    if st.button("Mulai Klasterisasi"):
        df_copy = preprocess_data(dataframe)
        # st.dataframe(df_copy)
        df_copy = normalize(df_copy) # Normalize data
        if option == "Bisecting K-Means":
            df_bkmeans, dfwaktu_bkmeans = BKMeans(df_copy)
            # plot = visualize_model(df_bkmeans)
            col1, col2 = st.columns([3, 1])
            col1.subheader("Skor Silhouette dan DBI dari Setiap Cluster")
            col1.line_chart(df_bkmeans, color=["#BD4B46", "#8D957E"])
            
            col2.subheader("Waktu Komputasi")
            col2.write(dfwaktu_bkmeans)
        elif option == "Agglomerative Hierarchical Clustering (AHC)":
            df_ahc, dfwaktu_ahc = BKMeans(df_copy)
            col1, col2 = st.columns([3, 1])
            col1.subheader("Skor Silhouette dan DBI dari Setiap Cluster")
            col1.line_chart(df_ahc, color=["#BD4B46", "#8D957E"])
            
            col2.subheader("Waktu Komputasi")
            col2.write(dfwaktu_ahc)
        else:
            df_bkmeans, dfwaktu_bkmeans = BKMeans(df_copy)
            df_ahc, dfwaktu_ahc = BKMeans(df_copy)
            col1, col2 = st.columns(2)
            col1.subheader("Skor Silhouette dan DBI Metode Bisecting K-Means")
            col1.line_chart(df_bkmeans, color=["#BD4B46", "#BD4B46"])
            col1.line_chart(df_ahc, color=["#8D957E", "#8D957E"])
            
            table = st.table(df_bkmeans)
            col2.subheader("Skor Silhouette dan DBI Metode AHC")
            col2.line_chart(df_ahc, color=["#BD4B46", "#8D957E"])

def about():
    st.write("This is the about page")
    st.title(f"{current_page.title}")

def help():
    st.write("This is the help page")

pages = [
    st.Page(home, icon=":material/home:", title="Home"),
    st.Page(about, icon=":material/info:", title="About"),
    st.Page(help, icon=":material/settings:", title="Help")
]

current_page = st.navigation(pages=pages, position="hidden")

st.set_page_config(layout="wide", page_title='Clustering Data BDSP',)

num_cols = max(len(pages) + 1, 8)

columns = st.columns(num_cols, vertical_alignment="bottom")

columns[0].write("**Clustering Data BDSP**")
# columns[0].write(str(date.today()))

for col, page in zip(columns[5:], pages):
    col.page_link(page, icon=page.icon)

current_page.run()