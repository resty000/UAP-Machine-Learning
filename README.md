# 📊 **Aplikasi Klasifikasi Tabular - Prediksi Kinerja Karyawan Menggunakan Algoritma Deep Learning** 🧠

Selamat datang di aplikasi *Klasifikasi Tabular* 🚀! Aplikasi ini membantu Anda untuk melakukan eksplorasi, analisis, dan klasifikasi data tabular dengan mudah menggunakan berbagai algoritma machine learning. Berikut adalah panduan penggunaan dan pengaturan lingkungan untuk menjalankan aplikasi ini.

---

## 📂 **Dataset**
Aplikasi ini menggunakan dataset "Employee Performance for HR Analytics" yang dapat diakses di Kaggle: [Link Dataset](https://www.kaggle.com/datasets/sanjanchaudhari/employees-performance-for-hr-analytics/data) 📥.

---

## ⚙️ **Setting Environment**

### 1. **Buat Virtual Environment**
   Pertama-tama, buatlah virtual environment untuk memastikan dependensi aplikasi terisolasi dengan baik.
   ```bash
   python -m venv myenv
   ```

### 2. **Aktifkan Virtual Environment**
   Pada Windows, aktifkan environment dengan perintah:
   ```bash
   myenv/Scripts/activate
   ```

### 3. **Install Dependencies**
   Install library yang diperlukan untuk menjalankan aplikasi:
   ```bash
   pip install streamlit
   pip install pdm
   pip install joblib
   pip install scikit-learn
   pip install tensorflow
   pip install seaborn
   pip install pytorch-tabnet
   ```

### 4. **Inisialisasi PDM (Python Development Master)**
   Inisialisasi PDM di proyek Anda:
   ```bash
   pdm init
   ```

---

## 💻 **Source Code**

Aplikasi ini memiliki beberapa skrip utama untuk memproses, menganalisis, dan mengklasifikasikan data tabular:

### 📜 `klasifikasi_data.py`
File ini berisi kode untuk memproses data, melatih model dengan berbagai algoritma (Logistic Regression, Random Forest, XGBoost, Neural Network, dll.), serta menampilkan hasil klasifikasi dalam bentuk yang interaktif.

### 📜 `preprocessing_data.py`
Berisi fungsi untuk melakukan preprocessing data, termasuk penanganan missing values, encoding kategori, serta normalisasi data.

### 📜 `upload_data.py`
Fungsi untuk meng-upload data CSV atau Excel dan menyimpannya untuk digunakan di halaman lain dalam aplikasi.

### 📜 `analisis_data.py`
Untuk melakukan eksplorasi data secara visual dan statistik seperti melihat distribusi data dan korelasi antar fitur.

### 📜 `app.py`
File utama aplikasi Streamlit yang mengatur navigasi antar halaman seperti Upload Data, Analisis Data, dan Klasifikasi.

---

## 🚀 **Menggunakan Aplikasi**

1. **Upload Data**: 
   Pilih file CSV atau Excel yang ingin Anda analisis. Setelah data berhasil diunggah, aplikasi akan menyimpannya untuk digunakan pada langkah berikutnya.

2. **Analisis Data**: 
   Lihat statistik data, identifikasi nilai yang hilang, serta buat visualisasi distribusi data dan heatmap korelasi antar fitur.

3. **Klasifikasi Data**: 
   Pilih kolom fitur dan target untuk pelatihan model, lalu pilih algoritma yang diinginkan (seperti Logistic Regression, Neural Network, TabNet, dll.). Setelah model dilatih, hasil klasifikasi serta akurasi model akan ditampilkan.

---

## 💡 **Fitur Utama Aplikasi**:
- **Upload Data**: Mudah mengunggah file CSV atau Excel
- **Analisis Data**: Statistik dasar, analisis data hilang, dan visualisasi distribusi serta korelasi
- **Klasifikasi**: Berbagai algoritma machine learning untuk klasifikasi data
- **Feature Importance**: Menampilkan grafik pentingnya fitur setelah model dilatih
- **Augmentasi Data**: Memanfaatkan GAN untuk memperbesar dataset dan meningkatkan kinerja model

---

## 📦 **Dependencies**:
- **Python 3.x**
- **Streamlit**: Untuk membuat antarmuka interaktif
- **Pandas & Numpy**: Untuk manipulasi data
- **Scikit-learn**: Untuk model machine learning dasar
- **TensorFlow & Keras**: Untuk Neural Network dan model pembelajaran mendalam
- **XGBoost**: Untuk model boosting
- **PyTorch TabNet**: Untuk model TabNet yang efisien
- **Seaborn & Matplotlib**: Untuk visualisasi data

---

## 📜 **Lisensi**
Model:
https://drive.google.com/drive/folders/1jyzwgf5ZNYVkuDBNY910JWEAJ8P3lVVG?usp=sharing
---

Jangan ragu untuk berkontribusi atau memberi feedback! Selamat mencoba dan semoga aplikasi ini membantu 🎉!

