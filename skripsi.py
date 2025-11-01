from utils import show_navbar, hide_sidebar, show_footer, home_page, analyze_page, about_page, profile_page, set_page_config
import streamlit as st
page = None
if page is None:
    set_page_config("Clustering Data Kacang Hijau")
else:
    set_page_config(f"{page} - Clustering Data Kacang Hijau")

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

# ====== ROUTER ======
page = st.query_params.get("page", "home").lower()

if page == "home":
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

