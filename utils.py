import streamlit as st
import re
import csv
import os
import io
import pandas as pd

from fungsi import (
    proses_clustering, validate_columns_and_data, preprocess_data, columns_to_drop, data_selection, BKMeans, AHC,
    normalize, proses_clustering_perbandingan, heatmap_corr, penyesuaian, pie_kontribusi, plot_tren_panen,
    avg_features, add_provinsi_to_df, show_prod_dan_lp, greet
    )

def show_navbar():
    st.markdown("""
    <nav class='navbar fixed-top navbar-dark' style='background-color: #183a1d; padding-top: 17px;'>
        <div class='container'>
            <span class='nav-title' style='color: #f0e0b1;'>Clustering Data Kacang Hijau</span>
            <div class='nav-text navbar-nav'>
                <ul class='nav justify-content-end'>
                    <li class='nav-item'>
                        <a class='nav-link' href='/'>Home</a>
                    </li>
                    <li class='nav-item'>
                        <a class='nav-link' href='/?page=analyze'>Analisis</a>
                    </li>
                    <li class='nav-item'>
                        <a class='nav-link' href='/?page=about'>Tentang</a>
                    </li>
                    <li class='nav-item'>
                        <a class='nav-link' href='/?page=profile'>Profile</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
            .navbar-dark .navbar-nav .nav-link {
                color: #f0e0b1;  /* Off-white text color */
                transition: color 0.3s ease-in-out, border-bottom 0.3s ease-in-out;
            }

            .navbar-dark .navbar-nav .nav-link:hover {
                color: #ff6f61;  /* Bright orange on hover */
            }

            .navbar {
                padding-top: 20px;  /* Adjust padding-top to make the navbar thicker at the top */
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("")


def what_page():
    # Get page from URL query params
    query_params = st.query_params
    page = query_params.get("page", ["home"])[0]  # Default ke 'home' jika parameter tidak ada
    return page

def hide_sidebar():
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                display: none
            }
        </style>
        """,
            unsafe_allow_html=True,
    )


def show_footer():
    st.markdown("""
    <div style='height:300px;'>
    </div>
    <div class='container'>
        <footer class='flex-wrap justify-content-between align-items-center py-3 my-4 border-top border-primary'>
            <p class='text-center text-body-secondary'> Made using Streamlit by Lisa © 2025. </p>
        </footer>
    </div>
    """, unsafe_allow_html=True)

def home_page():
    #Home Page
    st.set_page_config(
        layout="wide",
        page_title="Home",
        page_icon="app/images/kacang_hijau_icon.png"
    )

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
            label="Template Upload Dataset",
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
            **Sumber data** dapat diunduh dari [BDSP](https://bdsp2.pertanian.go.id/bdsp/id/lokasi) atau menggunakan contoh 
                    dataset yang diunduh melalui tombol "Template Upload Dataset".
            
            Berikut adalah alur penggunaan situs untuk melakukan clustering pada data kacang hijau :
            1. **Unggah Data**: Klik "Browse files" dan unggah dataset dalam bentuk excel (.csv / .xlsx).
            2. **Pilih Rentang Tahun**: Pilih rentang tahun yang ingin digunakan untuk proses clustering.
            3. **Pilih Algoritma dan Parameter**: Pilih algoritma dan jumlah cluster yang ingin diterapkan pada dataset Anda.
            4. **Mulai Clustering**: Dataset yang diunggah dapat diproses setelah pengguna memencet tombol "Mulai Clustering".
            5. **Lihat Hasil**: Setelah proses clustering selesai, hasil pengelompokan akan ditampilkan beserta metrik evaluasi.

            Jenis linkage Agglomerative Hierarchical Clustering :
            - Ward adalah metode yang meminimalkan variansi total dalam cluster.
            - Complete adalah metode yang meminimalkan jarak maksimum antara titik dalam cluster.
            - Average adalah metode yang meminimalkan jarak rata-rata antara titik dalam cluster.
            - Single adalah metode yang meminimalkan jarak minimum antara titik dalam cluster.

            """)
            st.session_state.instruction_shown = True

    if not st.session_state.get("instruction_shown", False):
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
            "Metode Clustering",
            options=["Bisecting K-Means", "Agglomerative Hierarchical Clustering"],
            default=["Agglomerative Hierarchical Clustering"],
            help="Pilih satu atau dua metode clustering untuk dibandingkan."
        )
        enabled = metode != ""
        if "Agglomerative Hierarchical Clustering" in metode:
            linkage = st.selectbox(
                "Parameter Linkage Agglomerative Hierarchical Clustering",
                options=["ward", "complete", "average", "single"],
                index=0, width="stretch",
                help="Linkage menentukan cara pengukuran jarak antar cluster.",
            )
        
    with cols[1]:
        cluster_option = st.radio(
            "Jumlah Cluster atau Pilih Rentang Cluster",
            options=["Jumlah cluster", "Rentang cluster"],
            horizontal=True,
            help="Pilih jumlah cluster data atau beberapa rentang jumlah cluster untuk dievaluasi."
        )

        if cluster_option == "Jumlah cluster": # Pilih angka jumlah cluster
            cluster_value = st.slider(
                "Jumlah Kelompok (Cluster) Data", 
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

def analyze_page():
    st.set_page_config(
        layout="wide", 
        page_title="About",
        page_icon="app/images/kacang_hijau_icon.png"
    )

    st.markdown('''<h1 class="custom-header" style="font-size:47px; align: center; color: black; margin-bottom: 26px; font-family: Inter;">
            Analisis Data
            </h1>''',
            unsafe_allow_html=True)

    st.write('''Halaman ini digunakan untuk menganalisis dataset hasil panen kacang hijau. Analisis mencakup korelasi antar fitur, 
            tren panen dari tahun ke tahun, serta visualisasi kontribusi produksi dan luas panen berdasarkan provinsi.''')

    uploaded_file = st.file_uploader("Unggah dataset berbentuk excel (.csv / .xlsx).")
    dataframe_mentah = None
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

    if uploaded_file is not None:
        try:
            validate_columns_and_data(dataframe_mentah)
            # st.success(":green[:material/done:] Data berhasil divalidasi dan diproses.")
            st.badge("Data berhasil divalidasi dan diproses.", icon=":material/check:", color="green")
            # df_copy = preprocess_data(dataframe_mentah)
            df_copy = columns_to_drop(dataframe_mentah) # temp untuk menampilkan tabel dataframe asli
            df_copy = data_selection (df_copy)
            df_copy = df_copy.reset_index()
            df_array = df_copy.drop(['Lokasi'], axis=1)
            df_array = df_array.values

            df_copy = avg_features(df_copy)
            dataframe_mentah = avg_features(dataframe_mentah)
            with st.spinner("Memproses data..."):
                cols = st.columns(2, gap="medium", vertical_alignment="top")
                with cols[0]:
                    heatmap_corr(df_copy[['Luas Panen', 'Produksi', 'Produktivitas']])
                with cols[1]:
                    st.write("##### Data Rata-rata Fitur per Lokasi")
                    nama_lokasi_awal = dataframe_mentah['Lokasi'].to_list()
                    dataframe_mentah = penyesuaian(dataframe_mentah)
                    dataframe_mentah = add_provinsi_to_df(dataframe_mentah)
                    st.dataframe(dataframe_mentah[['Provinsi', 'Lokasi', 'Luas Panen', 'Produksi', 'Produktivitas']], hide_index=True)

            with st.spinner("Memproses data..."):
                cols = st.columns(2, gap="medium", vertical_alignment="top")
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

            plot_tren_panen(df_copy)

        except ValueError as e:
            st.error(f"Terjadi kesalahan: {e}")

def about_page():
    st.set_page_config(
        layout="wide", 
        page_title="About",
        page_icon="app/images/kacang_hijau_icon.png"
    )

    st.title('Tentang Situs Ini')
    st.markdown("<br><br>", unsafe_allow_html=True)
    cols = st.columns([3,1])
    with cols[0]:
        tabs = st.tabs(["Latar Belakang & Informasi Terkait", "Mengenai Data", "Cara Kerja Situs"])
        with tabs[0]:
            with st.expander("Apa itu kacang hijau?"):
                column = st.columns(2)
                with column[0]:
                    st.image("app/images/kacang_hijau.jpg", caption="Kacang Hijau (Sumber: Alodokter)", width=385)
                with column[1]:
                    st.image("app/images/tanaman-kacang-hijau.jpg", caption="Tanaman Kacang Hijau (Sumber: Kompas)", width=300)
                st.write("""
                Kacang hijau adalah jenis tanaman kacang / legum dengan biji yang umumnya berwarna hijau dan tumbuh di daerah tropis.
                Kacang hijau kaya akan protein, serat, vitamin, dan mineral, serta mengandung antioksidan yang bermanfaat untuk kesehatan.
                Selain itu, kacang hijau juga dapat membantu mengontrol kadar gula darah, menurunkan kolesterol, dan 
                mendukung kesehatan pencernaan. Di Indonesia, kacang hijau sering digunakan dalam berbagai hidangan tradisional, 
                seperti bubur kacang hijau, es kacang hijau, dan kue-kue tradisional lainnya.
                """)
            with st.expander("Mengenai website dan tujuannya"):
                st.write("""
                Website ini dirancang untuk mengelompokkan data hasil panen kacang hijau berdasarkan tingkat tinggi / rendahnya 
                        luas panen, produksi, dan produktivitas. Pengelompokan ini dilakukan pada tingkat kabupaten/kota di 
                        seluruh wilayah Indonesia. Dengan menggunakan metode Bisecting K-Means dan Agglomerative Hierarchical 
                        Clustering, website ini bertujuan untuk memberikan wawasan yang lebih baik tentang pola dan tren dalam 
                        data pertanian kacang hijau.
                """)
                st.write("""
                Tujuan utama dari pengelompokan data hasil panen kacang hijau ini adalah untuk mengidentifikasi pola dan 
                        tren dalam produksi kacang hijau di berbagai wilayah di Indonesia. Dengan mengelompokkan data berdasarkan 
                        luas panen, produksi, dan produktivitas, diharapkan dapat ditemukan kelompok wilayah dengan karakteristik 
                        serupa. Informasi ini dapat digunakan untuk:
                1. Membantu petani dalam memahami kondisi pertanian di wilayah mereka dan mengadopsi praktik terbaik dari 
                wilayah lain dalam kelompok yang sama.
                2. Memberikan wawasan kepada pembuat kebijakan untuk merancang program dukungan yang lebih efektif bagi 
                petani di berbagai kelompok.
                3. Mendorong penelitian lebih lanjut tentang faktor-faktor yang mempengaruhi produksi kacang hijau di 
                berbagai wilayah di Indonesia.
                """)
            with st.expander("Data Mining dan Clustering"):
                st.write("""
                Data Mining adalah proses menemukan pola, tren, dan informasi berharga dari kumpulan data besar menggunakan 
                        teknik statistik, matematika, dan algoritma komputer. Tujuan utama dari data mining adalah mengubah 
                        data mentah menjadi informasi yang bermanfaat untuk pengambilan keputusan yang lebih baik.
                        
                Clustering adalah salah satu teknik dalam data mining yang digunakan untuk mengelompokkan data berdasarkan 
                        kemiripan sifat atau jarak antar data. Dalam clustering, data yang memiliki karakteristik serupa akan 
                        dikelompokkan bersama dalam satu cluster, sementara data yang berbeda akan ditempatkan di cluster yang 
                        berbeda. Teknik ini berguna untuk mengidentifikasi pola tersembunyi dalam data dan memahami struktur 
                        data secara lebih baik.
                """)
            with st.expander("Teknik Clustering"):
                st.write("""
                1. **Bisecting K-Means** adalah varian dari algoritma K-Means yang digunakan untuk pengelompokan data. 
                        Algoritma ini memulai dengan seluruh data dalam satu cluster dan kemudian secara iteratif membagi 
                        cluster tersebut menjadi dua sub-cluster hingga jumlah cluster yang diinginkan tercapai. Proses 
                        pembagian dilakukan dengan menerapkan algoritma K-Means pada cluster dengan varians tertinggi untuk dibagi. 
                        Metode ini sering kali menghasilkan cluster yang lebih baik dan lebih stabil dibandingkan dengan 
                        K-Means tradisional, terutama pada dataset yang besar dan kompleks.
                2. **Agglomerative Hierarchical Clustering** adalah metode pengelompokan data yang membangun cluster secara 
                        hierarki. Proses dimulai dengan setiap data sebagai cluster individu, kemudian secara iteratif 
                        menggabungkan dua cluster terdekat berdasarkan jarak atau kemiripan hingga jumlah cluster yang 
                        diinginkan tercapai. Karena proses penggabungan ini, metode ini umumnya memakan lebih banyak memori dan 
                        waktu komputasi dibandingkan dengan metode partisi seperti K-Means.
                """)
            with st.expander("Evaluasi dan Validasi Hasil Clustering"):
                st.write("""
                Untuk mengetahui seberapa bagus dan akurat model clustering yang digunakan, dapat dilakukan evaluasi berdasarkan hasil 
                        clustering yang diperoleh. Beberapa metode evaluasi yang umum digunakan antara lain:
                1. Silhouette Score: Mengukur seberapa mirip objek dalam cluster dengan objek di cluster lain. Nilai Silhouette berkisar 
                        antara -1 hingga 1. Nilai yang lebih tinggi menunjukkan clustering yang lebih baik. 
                2. Davies-Bouldin Index: Mengukur rasio rata-rata jarak antar cluster terhadap jarak dalam cluster. Nilai yang lebih 
                        rendah menunjukkan clustering yang lebih baik.
                """)

        with tabs[1]:
            with st.expander("Mengenai Sumber Data"):
                st.write("""
                Dataset dapat diperoleh dari situs (Basis Data Statistik Pertanian) dengan link https://bdsp2.pertanian.go.id/bdsp/id/lokasi 
                        milik Kementrian Pertanian Republik Indonesia. Pilih **Subsektor** Tanaman Pangan > **Komoditas** Kacang Hijau > 
                        **Indikator** Luas Panen, Produksi, Produktivitas > **Level** Kabupaten. Pilih **Satuan** yang sesuai dengan 
                        Indikator dan **Tahun** dari 2010 ke atas. Klik **Cari** dan unduh data dengan menekan tombol **Excel**.
                """)
                st.write("""
                Dataset sampel juga dapat diunduh melalui tombol "Template Upload Dataset" pada halaman utama situs ini.
                """)
            with st.expander("Syarat / Bentuk Dataset"):
                st.write("""
                Template dataset dapat dilihat pada file "Sampel Kacang Hijau 2010-2024.csv" yang dapat diunduh melalui tombol 
                        "Template Upload Dataset" pada halaman utama situs ini.
                """)
                st.write("""
                Dataset yang diunggah harus :
                1. Memiliki kolom **Lokasi**, **Luas Panen** [tahun], **Produksi** [tahun], dan **Produktivitas** [tahun]
                2. Semua kolom selain Lokasi berisi **numerik**
                """)
        with tabs[2]:
            st.write("""
                1. **Unggah Data**: Klik "Browse files" dan mengunggah dataset berbentuk excel (.csv / .xlsx).
                2. **Pilih Rentang Tahun**: Pilih rentang tahun yang ingin digunakan untuk proses clustering.
                3. **Pilih Algoritma dan Parameter**: Pilih algoritma dan jumlah cluster yang ingin diterapkan pada dataset Anda.
                4. **Mulai Clustering**: Dataset yang diunggah dapat diproses setelah pengguna memencet tombol "Mulai Clustering".
                5. **Lihat Hasil**: Setelah proses clustering selesai, hasil pengelompokan akan ditampilkan beserta metrik evaluasi.

                Jenis linkage Agglomerative Hierarchical Clustering :
                - Ward adalah metode yang meminimalkan variansi total dalam cluster.
                - Complete adalah metode yang meminimalkan jarak maksimum antara titik dalam cluster.
                - Average adalah metode yang meminimalkan jarak rata-rata antara titik dalam cluster.
                - Single adalah metode yang meminimalkan jarak minimum antara titik dalam cluster.

                Setiap visualisasi dan tabel yang ditampilkan dapat diunduh dengan memencet tombol :grey[:material/download:] yang tersedia pada 
                    pop-up yang muncul ketika mengarahkan kursor ke visualisasi.
                """)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    # Contact Information
    st.write("""
    Untuk pertanyaan lebih lanjut, Anda dapat menghubungi :  
    - Email: lisakurniadi16@gmail.com
    - Telepon: +62 812 9620 7168
    """)
    
def profile_page():

    st.set_page_config(
        layout="wide", 
        page_title="Profile",
        page_icon="app/images/kacang_hijau_icon.png"
        )

    st.markdown('''<h1 class="custom-header" style="font-size:47px; align: center; color: black; margin-bottom: 36px; font-family: Inter;">
                Profile
                </h1>''',
                unsafe_allow_html=True)

    greet()

    cols = st.columns([3,1], border=True, gap="medium")
    with cols[0]:
        tabs = st.tabs(["Keahlian", "Pendidikan"])
        with tabs[0]:
            st.write("""
                Saya Lisa, dan saat ini saya adalah mahasiswa semester akhir 
                Fakultas Teknologi Informasi di Universitas Tarumanagara. Saya memiliki 
                minat dalam bidang data science dan pengembangan web karena saya senang 
                menelaah dan menemukan hal-hal baru dalam data, serta mengubahnya menjadi 
                informasi baru yang bermanfaat. Saya juga menyukai proses membangun aplikasi 
                web yang interaktif dan user-friendly.    
                    
                Keahlian saya mencakup analisis dan komputasi data menggunakan bahasa pemrograman 
                R dan SQL, khususnya dalam bidang data mining dan machine learning. Saya juga 
                berpengalaman dalam menggunakan berbagai library Python untuk tugas data mining, 
                seperti Pandas, NumPy, dan Scikit-learn, serta untuk visualisasi data dengan Folium 
                dan Plotly. Keahlian saya juga mencakup penggunaan berbagai aplikasi yang mendukung 
                tugas data mining, seperti Oracle SQL Developer, Power BI, MATLAB, Visual Studio Code, dan R.
                """)
        with tabs[1]:
            st.write("""
                Selama masa sekolah saya di berpendidikan Santo Leo II, Jakarta. Saya lulus SMA pada tahun 2019 dan mulai kuliah di 
                    Universitas Tarumanagara pada tahun yang sama dengan memilih jurusan Teknik Informatika. Saat ini saya belum lulus,
                    tetapi saya sedang menyelesaikan tugas akhir saya melalui projek ini. Saya berharap dapat lulus pada tahun 2025.
                """)
    with cols[1]:
        column = st.columns([1,4,1], gap="large")
        with column[1]:
            st.image("app/images/profile.png", width=150)
        st.write("""
                Saya adalah mahasiswa aktif di Universitas Tarumanagara, Fakultas Teknologi Informasi, 
                jurusan Teknik Informatika. Saya lahir di Jakarta pada tanggal 16 Agustus 2001.
                
                **Kontak** :  
                lisakurniadi16@gmail.com  
                Telepon: 0812-9620-7168
                """)