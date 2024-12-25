# upload_data.py
import streamlit as st # type: ignore
import pandas as pd

def app():
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Cek ekstensi file
        file_extension = uploaded_file.name.split('.')[-1]
        
        if file_extension == "csv":
            # Membaca file CSV
            data = pd.read_csv(uploaded_file)
            st.write("Data CSV berhasil diunggah:", data.head())
        
        elif file_extension == "xlsx":
            # Membaca file Excel
            data = pd.read_excel(uploaded_file)
            st.write("Data Excel berhasil diunggah:", data.head())
        
        # Simpan data ke session_state agar bisa digunakan di halaman lain
        st.session_state["data"] = data
    else:
        st.warning("Silakan unggah file CSV atau Excel terlebih dahulu.")