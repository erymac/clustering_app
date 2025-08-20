from fungsi import preprocess_data, normalize, BKMeans, AHC, create_figure, compare, visualize_data, analisis, visualize_silhouette, penyesuaian, map_folium, sort_cluster
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

with open( "app\style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

#Home Page
def home():
    st.markdown('<h1 class="custom-header" style="align: center; margin-top: 36px; color: black; margin-bottom: 30px; font-family: Inter;">Clustering Data Kacang Hijau</h1>',
                unsafe_allow_html=True)
    # info = """
    # Data bisa diambil dari situs [Basis Data Statistik Pertanian (BDSP)](https://bdsp2.pertanian.go.id/bdsp/id/lokasi).
    # Pastikan file yang diunggah berbentuk csv / xlsx / xls.
    # """
    # st.markdown(info)
    data_path = 'https://bdsp2.pertanian.go.id/bdsp/id/lokasi'
    data_sample = 'Kacang Hijau.csv'
    st.markdown(
        f'''<p style="display: block; text-align: center;">
        <a href="{data_path}" target="_self">Sumber Data</a> | 
        <a> href="{data_sample}" target="_self">Contoh Data</a>
        </p>''',
        unsafe_allow_html=True
    )

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

    if "bkmeans" and "ahc" not in st.session_state:
        st.session_state.disabled = True

    st.markdown(
        """
        <div style='font-size:20px; text-align: center; font-family: Inter;'>
            Metode clustering apa yang ingin digunakan?
        </div>
        """,
        unsafe_allow_html=True
    )

    # subcol1, subcol2, subcol3, subcol4 = st.columns([4,2,1,3])
    # with subcol1:
    #     method1 = st.checkbox("Bisecting K-Means", key="bkmeans")
    #     method2 = st.checkbox("Agglomerative Hierarchical Clustering (AHC)", key="ahc")

    #     enabled = method1 or method2

    # with subcol2:
    #     cluster = st.selectbox(
    #         "",
    #         ("2 - 10", "2", "3", "4", "5", "6", "7", "8", "9", "10"),
    #         disabled = not enabled,
    #         index=None,
    #         placeholder="Pilih banyak cluster..."
    #         )

    # with subcol4:
    #     if method2:
    #         linkage = st.selectbox(
    #             "",
    #             ("Ward", "Average", "Complete", "Single"),
    #             index=None,
    #             placeholder="Pilih jenis linkage..."
    #             )

    # column1, column2, column3 = st.columns([3,1,3])
    # with column2 :
    #     mulai = st.button("Mulai Clustering")

    cols = st.columns([6,5,6,3], vertical_alignment="top", gap="medium")
    with cols[0]:
        metode = st.selectbox(
            "",
            ("Bisecting K-Means", "Agglomerative Hierarchical Clustering (AHC)", "Bisecting K-Means dan AHC"),
            index=None,
            placeholder="Pilih metode clustering..."
        )
        enabled = metode != ""

    #     method1 = st.checkbox("Bisecting K-Means", key="bkmeans")
    #     method2 = st.checkbox("Agglomerative Hierarchical Clustering (AHC)", key="ahc")
    #     enabled = method1 or method2
    with cols[1]:
        cluster = st.selectbox(
            "",
            ("2 - 10", "2", "3", "4", "5"),
            disabled = not enabled,
            index=None,
            placeholder="Pilih banyak cluster..."
            )
    with cols[2]:
        # if method2:
        if metode == "Agglomerative Hierarchical Clustering (AHC)" or metode == "Bisecting K-Means dan AHC":
            linkage = st.selectbox(
                "",
                ("Ward", "Average", "Complete", "Single"),
                index=None,
                placeholder="Pilih jenis linkage..."
                )
    with cols[3]:
        st.markdown("<div style='width: 1px; height: 28px; align-items: center; display: flex;'></div>", unsafe_allow_html=True)
        mulai = st.button("Mulai Clustering")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if mulai:
        if uploaded_file is not None:
            # dataframe = preprocess_data(dataframe)
            # df_copy = dataframe.copy()
            # df_array = normalize(df_copy) # Normalize data
            
            dataframe = preprocess_data(dataframe)
            df_copy = dataframe.copy()
            df_copy.drop(['Nama Kota'], axis=1, inplace=True)
            # df_array = normalize(df_copy.copy()) # Normalize data
            df_array = df_copy.copy().values
            print(type(df_array))

            if cluster == "2 - 10": # Menentukan jumlah cluster utk input
                n_cluster = range(2,11)
            elif cluster is None:
                st.warning("Pilih banyak cluster data.")
            else:
                cluster_int = int(cluster)
                n_cluster = range(cluster_int, cluster_int+1)
            # if method1 and not method2: # Jika memilih metode Bisecting K-Means
            if metode == "Bisecting K-Means":
                # metode = "Bisecting K-Means"
                df_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, _, bestcluster_bkmeans, labels_bkmeans = BKMeans(df_array, n_cluster)
                create_figure(cluster, df_bkmeans, dfwaktu_bkmeans, metode)
                analisis(metode, bestcluster_bkmeans, silhouette_bkmeans, dbi_bkmeans, cluster)

                dataframe['Cluster'] = labels_bkmeans
                dataframe = sort_cluster(dataframe)
                
                cols = st.columns(2, gap="small", vertical_alignment="top", border=True)
                with cols[0]:
                    df_temp = penyesuaian(dataframe)
                    # map_out(dataframe)
                    map_folium(df_temp, bestcluster_bkmeans)
                    # map_out(dataframe, bestcluster_bkmeans)
                with cols[1]:
                    visualize_data(df_copy, labels_bkmeans, metode)
                    visualize_silhouette(df_array, labels_bkmeans, bestcluster_bkmeans, silhouette_bkmeans, metode)                

            elif metode == "Agglomerative Hierarchical Clustering (AHC)":
                if linkage is not None:
                    linkage = linkage.lower()
                    metode = "Agglomerative Hierarchical Clustering"
                    df_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, _, bestcluster_ahc, labels_ahc = AHC(df_array, n_cluster, linkage)
                    create_figure(cluster, df_ahc, dfwaktu_ahc, metode)
                    analisis(metode, bestcluster_ahc, silhouette_ahc, dbi_ahc, bestcluster_ahc)

                    dataframe['Cluster'] = labels_ahc

                    subsubcol1, subsubcol2 = st.columns(2)
                    with subsubcol1:
                        visualize_silhouette(df_array, labels_ahc, bestcluster_ahc, silhouette_ahc, metode)
                    with subsubcol2:
                        visualize_data(df_copy, labels_ahc, metode)
                else:
                    st.warning("Jenis linkage metode AHC belum dipilih.")
            # elif method1 and method2: # Ketika memilih kedua metode, akan dibandingkan
            elif metode == "Bisecting K-Means dan AHC":
                if linkage is not None:
                    linkage = linkage.lower()
                    df_bkmeans, dfwaktu_bkmeans, silhouette_bkmeans, dbi_bkmeans, avg_silhouette_bkmeans, bestcluster_bkmeans, labels_bkmeans = BKMeans(df_array, n_cluster)
                    df_ahc, dfwaktu_ahc, silhouette_ahc, dbi_ahc, avg_silhouette_ahc, bestcluster_ahc, labels_ahc = AHC(df_array, n_cluster, linkage)
                    df_bkmeans['Metode'] = 'Bisecting K-Means'
                    df_ahc['Metode'] = 'AHC'

                    # Menyimpan nama metode
                    metode1 = 'Bisecting K-Means'
                    metode2 = 'AHC'

                    if cluster == "2 - 10":
                        method_dataframe = pd.concat([df_bkmeans, df_ahc], ignore_index=False)
                        method_dataframe = method_dataframe.reset_index() # Reset index
                        df_long = method_dataframe.melt(
                            id_vars=['Cluster', 'Metode'],
                            value_vars=['Silhouette Score', 'DBI Score'],
                            var_name='Skor',
                            value_name='Nilai'
                        )

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

                        subsubcol1, subsubcol2 = st.columns(2)
                        with subsubcol1:
                            st.plotly_chart(fig_silhouette, use_container_width=True)

                        with subsubcol2:
                            st.plotly_chart(fig_dbi, use_container_width=True)
                        result = compare(silhouette_bkmeans, silhouette_ahc, dbi_bkmeans, dbi_ahc, avg_silhouette_bkmeans, avg_silhouette_ahc, bestcluster_bkmeans, bestcluster_ahc)

                        subsubcol1, subsubcol2 = st.columns(2)
                        with subsubcol1:
                            visualize_data(df_copy, labels_bkmeans, metode1)
                            if result == 'BKMeans':
                                visualize_silhouette(df_array, labels_bkmeans, bestcluster_bkmeans, silhouette_bkmeans, metode1)
                                dataframe['Cluster'] = labels_bkmeans
                            elif result == 'AHC':
                                visualize_silhouette(df_array, labels_ahc, bestcluster_ahc, silhouette_ahc, metode2)
                                dataframe['Cluster'] = labels_ahc
                            else:
                                st.warning("Hasil metode tidak ditemukan.")
                        with subsubcol2:
                            visualize_data(df_copy, labels_ahc, metode2)

                    else:
                        waktu_bkmeans = dfwaktu_bkmeans.values[0][0]
                        waktu_ahc = dfwaktu_ahc.values[0][0]

                        singlcol1, singlcol2, singlcol3 = st.columns([5,1,10])
                        with singlcol1:
                            df_bkmeans['Waktu Komputasi (detik)'] = dfwaktu_bkmeans
                            df_bkmeans.reset_index(inplace=True, drop=True)
                            df_bkmeans.index = df_bkmeans['Metode']
                            # df_bkmeans.drop(columns='Metode', inplace=True)
                            
                            df_ahc['Waktu Komputasi (detik)'] = dfwaktu_ahc
                            df_ahc.reset_index(inplace=True, drop=True)
                            df_ahc.index = df_ahc['Metode']
                            # df_ahc.drop(columns='Metode', inplace=True)

                            df_compare = pd.concat([df_bkmeans, df_ahc], ignore_index=True)
                            df_compare = df_compare.set_index('Metode').T
                            st.dataframe(df_compare)
                        with singlcol3:
                            result = compare(silhouette_bkmeans, silhouette_ahc, dbi_bkmeans, dbi_ahc, avg_silhouette_bkmeans, avg_silhouette_ahc, bestcluster_bkmeans, bestcluster_ahc)
                            if result == "Waktu":
                                if waktu_bkmeans < waktu_ahc:
                                    st.info(f"""
                                    ##### Hasil perbandingan :  
                                    Nilai silhouette dan DBI Bisecting K-Means dan AHC sama namun rata-rata waktu komputasi Bisecting K-Means lebih cepat, sebesar {waktu_bkmeans:.4f} detik.
                                    """)
                                    result = 'BKMeans'
                                else:
                                    st.info(f"""
                                    ##### Hasil perbandingan :  
                                    Nilai silhouette dan DBI Bisecting K-Means dan AHC sama namun rata-rata waktu komputasi AHC lebih cepat, sebesar {waktu_ahc:.4f} detik.
                                    """)
                                    result = 'AHC'

                        subsubcol1, subsubcol2 = st.columns(2)
                        with subsubcol1:
                            visualize_data(df_copy, labels_bkmeans, metode1)
                            if result == 'BKMeans':
                                visualize_silhouette(df_array, labels_bkmeans, bestcluster_bkmeans, silhouette_bkmeans, metode1)
                                dataframe['Cluster'] = labels_bkmeans
                            elif result == 'AHC':
                                visualize_silhouette(df_array, labels_ahc, bestcluster_ahc, silhouette_ahc, metode2)
                                dataframe['Cluster'] = labels_ahc
                            else:
                                st.warning("Hasil metode tidak ditemukan.")
                        with subsubcol2:
                            visualize_data(df_copy, labels_ahc, metode2)
                else:
                    st.warning("Jenis linkage metode AHC belum dipilih.")
        else:
            st.error("File tidak ditemukan / belum diunggah.")

def about():
    st.write("This is the about page")

def help():
    st.write("This is the help page")

# pages = [
#     st.Page(home, icon=":material/home:", title="Home"),
#     st.Page(about, icon=":material/info:", title="About"),
#     st.Page(help, icon=":material/settings:", title="Help")
# ]

# Get page from URL query params
query_params = st.query_params
page = query_params.get("page", "home")  # Default to 'Home' if none

# Build nav bar
st.markdown("""
    <div class='nav-text'>
        <span class='nav-title'>Clustering Data Kacang Hijau</span>
        <span style='float: right;'>
            <a class='nav-link' href='?page=home' target='_self'>Home</a>
            <a class='nav-link' href='?page=about' target='_self'>About</a>
            <a class='nav-link' href='?page=help' target='_self'>Help</a>
        </span>
    </div>
""", unsafe_allow_html=True)

# --- Route content based on page ---
st.write("")

if page == "home":
    home()
elif page == "about":
    about()
elif page == "help":
    st.title("Help Page")
    st.write("Need help? Here's how to use the app...")
else:
    st.title("Page not found")