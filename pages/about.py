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
        st.write("Website ini dirancang untuk mengelompokkan data hasil panen kacang hijau" \
        " berdasarkan tingkat (tinggi / rendah) luas panen, produksi, dan produktivitasnya. Unggah " \
        "dataset pada halaman home dan lihat hasil pengelompokan data per kabupaten/kota serta visualisasi " \
        "grafik dan pemetaan wilayah Indonesia.")
    with st.expander("Sumber Data"):
        # st.write("Dataset yang digunakan pada website ini dapat diperoleh dari situs (Basis Data " \
        # "Statistik Pertanian) yang dimiliki oleh Kementrian Pertanian Republik Indonesia. Berikut syarat dataset :")
        st.write("""
        Dataset dapat diperoleh dari situs (Basis Data Statistik Pertanian) yang dimiliki oleh 
                 Kementrian Pertanian Republik Indonesia. Dataset yang diunggah harus :
        1. Memiliki kolom "Lokasi", "Luas Panen", "Produksi", dan "Produktivitas"
        2. Semua kolom selain "Lokasi" berisi numerik
        """)
    with st.expander("Cara Kerja Situs Ini"):
        st.write("""
        1. **Unggah Data**: Anda bisa mulai dengan mengunggah dataset untuk proses clustering pada halaman home.
        2. **Pilih Algoritma dan Parameter**: Pilih algoritma dan jumlah cluster yang Anda ingin terapkan pada dataset Anda.
        3. **Mulai Clustering**: Setelah itu dataset Anda akan langsung diproses dan hasil clustering akan langsung keluar.
        """)
    
st.markdown("<br><br>", unsafe_allow_html=True)
# Footer or Contact Information
st.write("""
Untuk pertanyaan lebih lanjut, Anda dapat menghubungi kami di:  
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