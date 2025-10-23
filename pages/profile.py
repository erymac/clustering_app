import streamlit as st
from fungsi import greet
from utils import show_navbar, hide_sidebar, show_footer
import os

st.set_page_config(
    layout="wide", 
    page_title="Profile",
    page_icon="app/images/kacang_hijau_icon.png"
    )
hide_sidebar()

with open( "app/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" crossorigin="anonymous">', unsafe_allow_html=True)

# Get page from URL query params
query_params = st.query_params
page = query_params.get("page", "home")  # Default to 'Home' if none

show_navbar()

st.markdown('''<h1 class="custom-header" style="font-size:47px; align: center; color: black; margin-bottom: 36px; font-family: Inter;">
            Profil
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
             minat dalam bidang data science dan pengembangan web karena Saya senang 
             menelaah dan menemukan hal-hal baru dalam data, serta mengubahnya menjadi 
             informasi baru yang bermanfaat. Saya juga menyukai proses membangun aplikasi 
             web yang interaktif dan user-friendly.
             """)
        # st.write("""
        #      Selain pemrograman dengan Python, saya juga memiliki keahlian dalam mengolah 
        #          data dan visualisasi menggunakan bahasa R.
        #      """)
    with tabs[1]:
        st.write("""
             Selama masa sekolah saya di berpendidikan Santo Leo II Jakarta. Saya lulus SMA pada tahun 2018 dan mulai kuliah di 
                 Universitas Tarumanagara pada tahun 2019 dengan memilih jurusan Teknik Informatika. Saat ini saya belum lulus,
                 tetapi saya sedang menyelesaikan tugas akhir saya melalui projek ini. Saya berharap dapat lulus pada tahun 2025.
             """)
with cols[1]:
    column = st.columns([1,4,1], gap=None)
    with column[1]:
        st.image("app/images/profile.png", width=150)
    st.write("""
            Kontak :  
            lisakurniadi16@gmail.com  
            Telepon: +62 812 9620 7168
            """)


show_footer()