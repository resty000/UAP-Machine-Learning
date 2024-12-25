import streamlit as st # type: ignore
from upload_data import app as upload_data_app
from analisis_data import app as analisis_data_app
from klasifikasi_data import app as klasifikasi_data_app

# Header utama aplikasi
st.title("Aplikasi Klasifikasi Tabular")
st.subheader("Eksplorasi dan Analisis Data Tabular dengan Mudah")

# Sidebar untuk navigasi
with st.sidebar:
    st.title("Navigasi")
    selected_page = st.radio(
        "Pilih halaman:",
        ["Beranda", "Upload Data", "Analisis Data", "Klasifikasi"]
    )

# Halaman beranda
if selected_page == "Beranda":
    st.header("Beranda")
    st.write("""
    Selamat datang di aplikasi *Klasifikasi Tabular*! 
    Aplikasi ini membantu Anda dalam:
    - **Mengunggah data**
    - **Menganalisis data**
    - **Melakukan klasifikasi**
    
    Gunakan menu navigasi di sebelah kiri untuk mulai.
    """)

# Halaman lainnya
elif selected_page == "Upload Data":
    upload_data_app()
elif selected_page == "Analisis Data":
    analisis_data_app()
elif selected_page == "Klasifikasi":
    klasifikasi_data_app()
