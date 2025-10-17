import streamlit as st
from utils import show_navbar, hide_sidebar, show_footer

st.set_page_config(layout="wide", page_title="About")
hide_sidebar()

with open( "app\style.css" ) as css:
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
    with st.expander("Mengenai website dan tujuannya"):
        st.write("""
        Website ini dirancang untuk mengelompokkan data hasil panen kacang hijau berdasarkan tingkat (tinggi / rendah) 
                 luas panen, produksi, dan produktivitas. Pengelompokan ini dilakukan pada tingkat kabupaten/kota di 
                 seluruh wilayah Indonesia. Dengan menggunakan metode Bisecting K-Means dan Agglomerative Hierarchical 
                 Clustering, website ini bertujuan untuk memberikan wawasan yang lebih baik tentang pola dan tren dalam 
                 data pertanian kacang hijau. Hasil dari pengelompokan ini diharapkan dapat membantu petani, pembuat 
                 kebijakan, dan pemangku kepentingan lainnya dalam membuat keputusan terkait dengan produksi kacang hijau.
        """)
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
        # st.write("Dataset yang digunakan pada website ini dapat diperoleh dari situs (Basis Data " \
        # "Statistik Pertanian) yang dimiliki oleh Kementrian Pertanian Republik Indonesia. Berikut syarat dataset :")
        st.write("""
        Dataset yang diunggah harus :
        1. Memiliki kolom **Lokasi**, **Luas Panen**, **Produksi**, dan **Produktivitas**
        2. Semua kolom selain Lokasi berisi **numerik**
        """)
    with st.expander("Cara Kerja Situs Ini"):
        st.write("""
        1. **Unggah Data**: Mulai dengan mengunggah dataset untuk proses clustering pada halaman home.
        2. **Pilih Algoritma dan Parameter**: Pilih algoritma dan jumlah cluster yang ingin diterapkan pada dataset Anda.
        3. **Mulai Clustering**: Setelah itu dataset Anda akan langsung diproses dan hasil clustering akan langsung keluar.
        """)
    
st.markdown("<br><br>", unsafe_allow_html=True)
# Footer or Contact Information
st.write("""
Untuk pertanyaan lebih lanjut, Anda dapat menghubungi :  
- Email: lisa.535190064@stu.untar.ac.id  
- Telepon: +62 812 9620 7168
""")
    
show_footer()
    
# # --- Route content based on page ---
# st.write("")

# if page == "home":
#     home()
# elif page == "about":
#     content()
# elif page == "profile":
#     # st.switch_page("profile.py")
#     # st.page_link("profile.py")
#     profile.content()
# else:
#     st.title("Page not found")