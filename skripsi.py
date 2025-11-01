from utils import show_navbar, hide_sidebar, show_footer, home_page, analyze_page, about_page, profile_page, set_page_config
import streamlit as st
set_page_config("Clustering Data Kacang Hijau")
hide_sidebar()

st.markdown("""
<nav class='navbar fixed-top navbar-dark' style='background-color: #183a1d; padding-top: 17px;'>
    <div class='container'>
        <span class='nav-title' style='color: #f0e0b1;'>Clustering Data Kacang Hijau</span>
        <div class='nav-text navbar-nav'>
            <ul class='nav justify-content-end'>
                <li class='nav-item'>
                    <a class='nav-link' href='?page=home' target="_self">Home</a>
                </li>
                <li class='nav-item'>
                    <a class='nav-link' href='?page=analyze' target="_self">Analisis</a>
                </li>
                <li class='nav-item'>
                    <a class='nav-link' href='?page=about' target="_self">Tentang</a>
                </li>
                <li class='nav-item'>
                    <a class='nav-link' href='?page=profile' target="_self">Profile</a>
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

with open("app/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" crossorigin="anonymous">', unsafe_allow_html=True)

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

# ====== ROUTER ======
page = st.query_params.get("page", "home").lower()

if page == "home":
    if not st.session_state.instruction_shown:
        instruction()
    home_page()
elif page == "analyze":
    analyze_page()
elif page == "about":
    about_page()
elif page == "profile":
    profile_page()
else:
    home_page()

show_footer()

