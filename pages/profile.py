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

show_navbar()

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