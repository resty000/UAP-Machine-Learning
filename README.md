# 📊 **Aplikasi Klasifikasi Tabular - Prediksi Kinerja Karyawan Menggunakan Algoritma Deep Learning** 🧠
## ✏️ **Latar Belakang**
Selamat datang di aplikasi *Klasifikasi Tabular* 🚀! Aplikasi ini membantu Anda untuk melakukan eksplorasi, analisis, dan klasifikasi data tabular dengan mudah menggunakan berbagai algoritma machine learning. Berikut adalah panduan penggunaan dan pengaturan lingkungan untuk menjalankan aplikasi ini.

---

## 🕵️ **Tujuan**
1. Menyediakan platform interaktif untuk klasifikasi data tabular.
2. Memungkinkan pengguna untuk membandingkan performa berbagai model klasifikasi.
3. Memberikan analisis visual yang membantu pengguna memahami data dan hasil model.

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

## 💻 ** Deskripsi Model**
Model yang Digunakan
Aplikasi ini mendukung berbagai algoritma machine learning untuk klasifikasi, termasuk:
- **Logistic Regression**: Algoritma baseline yang sederhana dan cepat.
- **Decision Tree**: Algoritma yang mudah diinterpretasi, cocok untuk data kecil.
- **Random Forest**: Ensambel model yang efektif untuk mengurangi overfitting.
- **Support Vector Machine (SVM)**: Algoritma yang kuat untuk data berdimensi tinggi.
- **Neural Network (MLP)**: Model berbasis jaringan saraf untuk pola kompleks.
- **TabNet**: Model berbasis PyTorch untuk data tabular.
- **XGBoost**: Algoritma gradient boosting yang populer dan sangat efektif.
- **Feedforward Neural Network (FNN)**: Jaringan saraf sederhana untuk klasifikasi tabular.

##  **Analisis Performa**
Performa model dievaluasi menggunakan metrik seperti:
- **Accuracy**: Persentase prediksi yang benar.
- **Precision**: Kemampuan model dalam menghindari false positives.
- **Recall**: Kemampuan model dalam menangkap true positives.
- **F1-Score**: Harmonik rata-rata antara precision dan recall.

---

## 📊 **Hasil dan Analisis**
Berikut adalah hasil evaluasi dari berbagai model yang digunakan. Dataset sampel digunakan untuk menguji akurasi, presisi, dan metrik lainnya.

| Model                    | Accuracy | Precision | Recall | F1-Score |
|--------------------------|----------|-----------|--------|----------|
| Logistic Regression      | 71.04%   | 50.47%     | 71.04%  | 59.01%    |
| Decision Tree            | 62.74%   | 63.37%     | 62.74%  | 63.04%    |
| Random Forest            | 69.55%   | 65.08%     | 69.55%  | 65.52%    |
| SVM                      | 71.04%   | 50.47%     | 71.04%  | 59.01%    | 
| Neural Network (MLP)     | 66.79%   | 63.82%     | 66.79%  | 64.82%    |
| Feedforward Neural Net   | 71.01%   | 65.15%     | 71.01%  | 61.09%    |
| XGBoost                  | 69.20%   | 65.36%     | 69.20%  | 66.06%    |
| TabNet                   | 70.01%   | 64.66%     | 70.01%  | 64.41%    |

Analisis Hasil:
1. Model dengan Akurasi Tertinggi:
Logistic Regression, SVM, dan Feedforward Neural Network memberikan akurasi tertinggi (sekitar 71%).
2. Model dengan F1-Score Tertinggi:
XGBoost memiliki F1-Score tertinggi (66.06%), diikuti oleh Random Forest (65.52%) dan Neural Network (64.82%).
3. Rekomendasi:
- XGBoost menjadi pilihan terbaik secara keseluruhan karena memiliki keseimbangan antara akurasi (69.20%) dan F1-Score (66.06%).
- Feedforward Neural Network adalah alternatif menarik karena memiliki performa hampir setara dengan Logistic Regression tetapi lebih baik dalam menangkap pola data.

**Statistik data**:
![Screenshot 2024-12-26 032733](https://github.com/user-attachments/assets/13486322-4f12-4110-af77-cf417149b6b1)

**Hubungan Antar Fitur**:
![image](https://github.com/user-attachments/assets/d605c9db-3045-4af6-aca7-ad89880aef19)

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

