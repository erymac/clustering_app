import streamlit as st

# def show_navbar():
#     st.markdown("""
#     <nav class='navbar fixed-top navbar-dark' style='background-color: #183a1d; padding-top: 17px;'>
#         <div class='container'>
#             <span class='nav-title' style='color: #f0e0b1;'>Clustering Data Kacang Hijau</span>
#             <div class='nav-text navbar-nav'>
#                 <ul class='nav justify-content-end'>
#                     <li class='nav-item'>
#                         <a class='nav-link' href='/'>Home</a>
#                     </li>
#                     <li class='nav-item'>
#                         <a class='nav-link' href='/analyze?page=analyze'>Analisis</a>
#                     </li>
#                     <li class='nav-item'>
#                         <a class='nav-link' href='/about?page=about'>Tentang</a>
#                     </li>
#                     <li class='nav-item'>
#                         <a class='nav-link' href='/profile?page=profile'>Profile</a>
#                     </li>
#                 </ul>
#             </div>
#         </div>
#     </nav>
#     """, unsafe_allow_html=True)

#     st.markdown("""
#         <style>
#             .navbar-dark .navbar-nav .nav-link {
#                 color: #f0e0b1;  /* Off-white text color */
#                 transition: color 0.3s ease-in-out, border-bottom 0.3s ease-in-out;
#             }

#             .navbar-dark .navbar-nav .nav-link:hover {
#                 color: #ff6f61;  /* Bright orange on hover */
#             }

#             .navbar {
#                 padding-top: 20px;  /* Adjust padding-top to make the navbar thicker at the top */
#             }
#         </style>
#     """, unsafe_allow_html=True)

#     st.markdown("")


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