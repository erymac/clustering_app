import streamlit as st
from utils import show_navbar, hide_sidebar

st.set_page_config(layout="wide", page_title="Profile")
hide_sidebar()

with open( "app\style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" crossorigin="anonymous">', unsafe_allow_html=True)

def content():
    st.title("About Me")

# Get page from URL query params
query_params = st.query_params
page = query_params.get("page", "home")  # Default to 'Home' if none

st.markdown("""
<nav class='navbar fixed-top navbar-dark' style='background-color: #984216;'>
    <div class='container'>
        <span class='nav-title'>Clustering Data Kacang Hijau</span>
        <div class='nav-text navbar-nav'>
            <ul class='nav justify-content-end '>
                <li class='nav-item'>
                    <a class='nav-link' href='/'>Home</a>
                </li>
                <li class='nav-item'>
                    <a class='nav-link' href='/analyze'>Analisis Data</a>
                </li>
                <li class='nav-item'>
                    <a class='nav-link' href='/about'>Tentang</a>
                </li>
                <li class='nav-item'>
                    <a class='nav-link' href='/profile'>Profil</a>
                </li>
            </ul>
        </div>
    </div>
</nav>
""", unsafe_allow_html=True)

content()
# # --- Route content based on page ---
# st.write("")

# if page == "home":
#     home()
# elif page == "about":
#     # st.switch_page("about.py")
#     # st.page_link("about.py")
#     about.content()
# elif page == "profile":
#     content()
# else:
#     st.title("Page not found")