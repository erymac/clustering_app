import streamlit as st
from fungsi import greet
from utils import show_navbar, hide_sidebar, show_footer, what_page
import os

st.set_page_config(
    layout="wide", 
    page_title="Profile",
    page_icon="app/images/kacang_hijau_icon.png"
    )

hide_sidebar()

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
                    <a class='nav-link' href='/analyze'>Analisis</a>
                </li>
                <li class='nav-item'>
                    <a class='nav-link' href='/about'>Tentang</a>
                </li>
                <li class='nav-item'>
                    <a class='nav-link' href='#'>Profile</a>
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

with open( "app/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" crossorigin="anonymous">', unsafe_allow_html=True)

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


show_footer()

