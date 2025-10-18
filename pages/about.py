import streamlit as st
from utils import show_navbar, hide_sidebar, show_footer

st.set_page_config(
    layout="wide", 
    page_title="About",
    page_icon="app/images/kacang_hijau_icon.png"
    )
hide_sidebar()

with open( "app/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" crossorigin="anonymous">', unsafe_allow_html=True)

def content():
    st.write("Pelajari lebih lanjut mengenai situs ini.")

# Get page from URL query params
query_params = st.query_params
page = query_params.get("page", "home")  # Default to 'Home' if none

show_navbar()

st.title('Tentang Situs Ini')
st.markdown("<br><br>", unsafe_allow_html=True)
cols = st.columns([3,1])
with cols[0]:
    tabs = st.tabs(["Latar Belakang & Informasi Terkait", "Mengenai Data", "Cara Kerja Situs"])
    with tabs[0]:
        with st.expander("Mengenai website dan tujuannya"):
            st.write("""
            Website ini dirancang untuk mengelompokkan data hasil panen kacang hijau berdasarkan tingkat (tinggi / rendah) 
                    luas panen, produksi, dan produktivitas. Pengelompokan ini dilakukan pada tingkat kabupaten/kota di 
                    seluruh wilayah Indonesia. Dengan menggunakan metode Bisecting K-Means dan Agglomerative Hierarchical 
                    Clustering, website ini bertujuan untuk memberikan wawasan yang lebih baik tentang pola dan tren dalam 
                    data pertanian kacang hijau. Hasil dari pengelompokan ini diharapkan dapat membantu petani, pembuat 
                    kebijakan, dan pemangku kepentingan lainnya dalam membuat keputusan terkait dengan produksi kacang hijau.
            """)
        with st.expander("Tujuan pengelompokan data"):
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
                    kemiripan atau jarak antar data. Dalam clustering, data yang memiliki karakteristik serupa akan 
                    dikelompokkan bersama dalam satu cluster, sementara data yang berbeda akan ditempatkan di cluster yang 
                    berbeda. Teknik ini berguna untuk mengidentifikasi pola tersembunyi dalam data dan memahami struktur 
                    data secara lebih baik.
            """)
        with st.expander("Bisecting K-Means"):
            st.write("""
            Bisecting K-Means adalah varian dari algoritma K-Means yang digunakan untuk pengelompokan data. 
                    Algoritma ini memulai dengan seluruh data dalam satu cluster dan kemudian secara iteratif membagi 
                    cluster tersebut menjadi dua sub-cluster hingga jumlah cluster yang diinginkan tercapai. Proses 
                    pembagian dilakukan dengan menerapkan algoritma K-Means pada cluster dengan varians tertinggi untuk dibagi. 
                    Metode ini sering kali menghasilkan cluster yang lebih baik dan lebih stabil dibandingkan dengan 
                    K-Means tradisional, terutama pada dataset yang besar dan kompleks.
            """)
        with st.expander("Agglomerative Hierarchical Clustering"):
            st.write("""
            Agglomerative Hierarchical Clustering adalah metode pengelompokan data yang membangun cluster secara 
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
            # st.write("Dataset yang digunakan pada website ini dapat diperoleh dari situs (Basis Data " \
            # "Statistik Pertanian) yang dimiliki oleh Kementrian Pertanian Republik Indonesia. Berikut syarat dataset :")
            st.write("""
            Dataset dapat diperoleh dari situs (Basis Data Statistik Pertanian) dengan link https://bdsp2.pertanian.go.id/bdsp/id/lokasi 
                    milik Kementrian Pertanian Republik Indonesia. Pilih **Subsektor** Tanaman Pangan > **Komoditas** Kacang Hijau > 
                    **Indikator** Luas Panen, Produksi, Produktivitas > **Level** Kabupaten. Pilih **Satuan** yang sesuai dengan 
                    Indikator dan **Tahun** dari 2010 ke atas. Klik **Cari** dan unduh data dengan menekan tombol **Excel**.
            """)
        with st.expander("Syarat / Bentuk Dataset"):
            st.write("""
            Dataset yang diunggah harus :
            1. Memiliki kolom **Lokasi**, **Luas Panen**, **Produksi**, dan **Produktivitas**
            2. Semua kolom selain Lokasi berisi **numerik**
            """)
    with tabs[2]:
        st.write("""
            1. **Unggah Data**: Mulai dengan mengunggah dataset untuk proses clustering pada halaman home.
            2. **Pilih Algoritma dan Parameter**: Pilih algoritma dan jumlah cluster yang ingin diterapkan pada dataset Anda.
            3. **Mulai Clustering**: Setelah itu dataset Anda akan langsung diproses dan hasil clustering akan langsung keluar.

            Setiap visualisasi dan tabel yang ditampilkan dapat diunduh dengan menekan tombol :grey[:material/download:] yang tersedia pada 
                 pop-up yang muncul ketika mengarahkan kursor ke visualisasi.
            """)
    
st.markdown("<br><br>", unsafe_allow_html=True)
# Contact Information
st.write("""
Untuk pertanyaan lebih lanjut, Anda dapat menghubungi :  
- Email: lisakurniadi16@gmail.com
- Telepon: +62 812 9620 7168
""")
    
show_footer()