import streamlit as st  # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi utama untuk aplikasi analisis data
def app():
    st.header("Analisis Data")

    # Pastikan data tersedia sebelum melanjutkan
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Silakan unggah data terlebih dahulu di menu Upload Data.")
        return

    # Muat data dari session state
    data = st.session_state.data

    # Pilihan untuk melihat statistik data
    st.subheader("Statistik Data")
    if st.checkbox("Tampilkan Statistik Data"):
        st.write(data.describe(include='all'))

    # Pilihan untuk melihat data yang hilang
    st.subheader("Analisis Data yang Hilang")
    if st.checkbox("Tampilkan Informasi Data yang Hilang"):
        missing_data = data.isnull().sum()
        st.write(missing_data[missing_data > 0])
        st.bar_chart(missing_data)

    # Visualisasi distribusi data
    st.subheader("Distribusi Data")
    if st.checkbox("Tampilkan Distribusi Kolom"):
        column = st.selectbox("Pilih kolom untuk visualisasi:", data.columns)
        if pd.api.types.is_numeric_dtype(data[column]):
            fig, ax = plt.subplots()
            sns.histplot(data[column], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            data[column].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

    # Visualisasi hubungan antar fitur
    st.subheader("Hubungan Antar Fitur")
    if st.checkbox("Tampilkan Heatmap Korelasi"):
        numeric_data = data.select_dtypes(include=['number'])
        if numeric_data.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Tidak cukup data numerik untuk membuat heatmap korelasi.")

    # Pilihan untuk mengisi data yang hilang
    st.subheader("Penanganan Data yang Hilang")
    if st.checkbox("Isi Data yang Hilang"):
        column = st.selectbox("Pilih kolom untuk mengisi data yang hilang:", data.columns)

        # Jika kolom numerik
        if pd.api.types.is_numeric_dtype(data[column]):
            method = st.selectbox("Pilih metode pengisian:", ["Mean", "Median", "Mode", "Custom Value"])
            if method == "Custom Value":
                value = st.text_input("Masukkan nilai custom:")
                if st.button("Isi Data"):
                    data[column].fillna(value, inplace=True)
                    st.success(f"Data yang hilang pada kolom {column} telah diisi dengan nilai '{value}'.")
            else:
                if st.button("Isi Data"):
                    if method == "Mean":
                        data[column].fillna(data[column].mean(), inplace=True)
                    elif method == "Median":
                        data[column].fillna(data[column].median(), inplace=True)
                    elif method == "Mode":
                        data[column].fillna(data[column].mode()[0], inplace=True)
                    st.success(f"Data yang hilang pada kolom {column} telah diisi dengan metode {method}.")
        
        # Jika kolom kategorikal (non-numerik)
        else:
            method = st.selectbox("Pilih metode pengisian:", ["Mode", "Custom Value"])
            if method == "Custom Value":
                value = st.text_input("Masukkan nilai custom:")
                if st.button("Isi Data"):
                    data[column].fillna(value, inplace=True)
                    st.success(f"Data yang hilang pada kolom {column} telah diisi dengan nilai '{value}'.")
            elif method == "Mode":
                if st.button("Isi Data"):
                    # Mengisi dengan modus (nilai yang paling sering muncul)
                    mode_value = data[column].mode()[0]
                    data[column].fillna(mode_value, inplace=True)
                    st.success(f"Data yang hilang pada kolom {column} telah diisi dengan nilai modus '{mode_value}'.")

    # Simpan perubahan pada session state
    st.session_state.data = data
