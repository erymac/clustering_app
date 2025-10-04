import streamlit as st

def show_navbar():
    # # <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
    # st.markdown("""
    # <nav class='navbar fixed-top navbar-dark' style='background-color: #984216;'>
    #     <div class='container'>
    #         <span class='nav-title'>Clustering Data Kacang Hijau</span>
    #         <div class='nav-text navbar-nav'>
    #             <ul class='nav justify-content-end '>
    #                 <li class='nav-item'>
    #                     <a class='nav-link active' href='?page=home'>Home</a>
    #                 </li>
    #                 <li class='nav-item'>
    #                     <a class='nav-link active' href='?page=about' target='_self'>About</a>
    #                 </li>
    #                 <li class='nav-item'>
    #                     <a class='nav-link active' href='?page=profile' target='_self'>Profile</a>
    #                 </li>
    #             </ul>
    #         </div>
    #     </div>
    # </nav>
    # """, unsafe_allow_html=True)
    st.markdown("""
    <nav class='navbar fixed-top navbar-dark' style='background-color: #183a1d;'>
        <div class='container'>
            <span class='nav-title' style='color: #f0e0b1;'>Clustering Data Kacang Hijau</span>
            <div class='nav-text navbar-nav'>
                <ul class='nav justify-content-end'>
                    <li class='nav-item'>
                        <a class='nav-link' href='/'>Home</a>
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

    # Add a minimal hover effect with light underline and subtle pale green
    st.markdown("""
        <style>
            .navbar-dark .navbar-nav .nav-link {
                color: #f0e0b1;  /* Off-white text color */
                transition: color 0.3s ease-in-out, border-bottom 0.3s ease-in-out;
            }
            
            .navbar-dark .navbar-nav .nav-link:hover {
                color: #ff6f61;  /* Bright orange on hover */
            }
        </style>
    """, unsafe_allow_html=True)

def what_page():
    # Get page from URL query params
    query_params = st.query_params
    page = query_params.get("page", "home")  # Default to 'Home' if none
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

# def show_footer():
#     st.markdown("""
#     <hr style='margin: 0.5rem 0;'>
#     <div class='col mb-3' style='text-align: center; font-size: 0.9rem; padding: 0.5rem 0;'>
#         <ul class='nav flex-column'>
#             <li>
#                 <a> Made in 🎈Streamlit by Lisa. </a>
#             </li>
#             <li>
#                 <a> © 2025 </a>
#             </li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)

# def show_footer():
#     st.markdown("""
#     <footer class='d-flex flex-wrap justify-content-between align-items-center py-3 my-4 border-top'>
#         <div class='footer fixed-bottom'>
#             <span class='text-muted'>
#                 <p class='text-center text-body-secondary'> © 2025 Lisa </p>
#             </span>
#         </div>
#     </footer>
#     """, unsafe_allow_html=True)

def show_footer():
    st.markdown("""
    <div style='height:300px;'>
    </div>
    <div class='container'>
        <footer class='flex-wrap justify-content-between align-items-center py-3 my-4 border-top border-primary'>
            <p class='text-center text-body-secondary'> © 2025 Lisa </p>
        </footer>
    </div>
    """, unsafe_allow_html=True)


# Build nav bar
# st.markdown("""
#     <div class='nav-text'>
#         <span class='nav-title'>Clustering Data Kacang Hijau</span>
#         <span style='float: right;'>
#             <a class='nav-link' href='?page=home' target='_self'>Home</a>
#             <a class='nav-link' href='?page=about' target='_self'>About</a>
#             <a class='nav-link' href='?page=help' target='_self'>Help</a>
#         </span>
#     </div>
# """, unsafe_allow_html=True)



# st.markdown("""
# <nav class='navbar fixed-top navbar-expand-lg navbar-dark' style='background-color: #984216;'>
#     <button class='navbar-toggler' type='button' data-bs-toggle='collapse' data-bs-target='#navbarSupportedContent' aria-controls='navbarSupportedContent' aria-expanded='false' aria-label='Toggle navigation'>
#       <span class='navbar-toggler-icon'></span>
#     </button>
#     <span class='nav-title'>Clustering Data Kacang Hijau</span>
#     <div class='collapse navbar-collapse nav-text' id='navbarSupportedContent'>
#       <ul class='navbar-nav me-auto mb-2 mb-lg-0 '>
#         <li class='nav-item'>
#           <a class='nav-link active' href='?page=home'>Home</a>
#         </li>
#         <li class='nav-item'>
#           <a class='nav-link' href='?page=about'>About</a>
#         </li>
#         <li class='nav-item'>
#           <a class='nav-link' href='?page=profile'>Profile</a>
#         </li>
#       </ul>
#     </div>
# </nav>
# """, unsafe_allow_html=True)