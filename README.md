# Deteksi Tumor Otak dari Citra MRI

## Overview

Proyek ini bertujuan untuk mengembangkan sistem deteksi tumor otak non-invasif dari citra Magnetic Resonance Imaging (MRI) menggunakan teknik pengolahan citra digital dan machine learning. Proyek ini mencakup pipeline lengkap mulai dari preprocessing citra, ekstraksi fitur, hingga klasifikasi otomatis untuk mengidentifikasi keberadaan dan jenis tumor otak.

Dataset yang digunakan adalah **Brain Tumor MRI Dataset** yang tersedia publik di Kaggle, berisi citra MRI otak dengan empat kategori: Glioma Tumor, Meningioma Tumor, Pituitary Tumor, dan No Tumor (normal).

## Fitur Utama

*   **Preprocessing Citra:** Implementasi teknik noise reduction (Gaussian Blur), contrast enhancement (CLAHE), dan normalization (Min-Max Normalization) untuk meningkatkan kualitas citra MRI.
*   **Image Transformation & Segmentation:** Penerapan deteksi tepi (Canny, Sobel), operasi morfologi (Opening, Closing, Gradient), dan segmentasi (Otsu, Adaptive Threshold, K-means) untuk mengekstrak area penting dalam citra, khususnya area yang berpotensi mengandung tumor.
*   **Feature Extraction:** Ekstraksi fitur-fitur kunci dari citra (tekstur, bentuk, intensitas) yang relevan untuk membedakan antar kelas tumor dan non-tumor.
*   **Klasifikasi Machine Learning:** Pelatihan dan evaluasi beberapa model klasifikasi (SVM, Random Forest, K-Nearest Neighbors, Neural Network) menggunakan fitur yang terekstrak untuk mengklasifikasikan citra MRI ke dalam salah satu dari empat kategori.
*   **Streamlit Web App:** Antarmuka pengguna web sederhana untuk mengunggah citra MRI dan mendapatkan hasil prediksi secara real-time.

## Struktur Proyek
brain-tumor-streamlit-app/ 
  ├── app.py 
  ├── requirements.txt 
    └── models/ 
      ├── best_model.pkl 
      └── scaler.pkl 

*   **Catatan:** Notebook Google Colab asli tempat model dilatih mungkin berada di luar struktur ini, namun file `.pkl` yang dihasilkan harus ditempatkan di sini.

## Dataset

Dataset yang digunakan dapat diunduh dari Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). Dataset ini dibagi menjadi direktori `Training` dan `Testing` dengan sub-direktori untuk setiap kelas tumor (Glioma, Meningioma, Pituitary) dan kelas normal (No Tumor).

## Persyaratan Sistem

*   Python 3.6 atau lebih tinggi
*   Pustaka Python yang tercantum dalam `requirements.txt`
*   Google Colab (opsional, untuk melatih model dan mendapatkan file `.pkl`)
*   VS Code atau IDE Python lainnya (untuk menjalankan aplikasi Streamlit)

## Cara Menjalankan

Ikuti langkah-langkah ini untuk menjalankan aplikasi deteksi tumor otak:

1.  **Clone Repositori:**

bash git clone cd brain-tumor-streamlit-app

2.  **Dapatkan Model dan Scaler:**
    *   **Penting:** Repositori ini tidak menyertakan file model dan scaler yang sudah dilatih (`.pkl`) karena ukurannya mungkin besar dan tergantung pada hasil pelatihan Anda.
    *   Anda harus melatih model (mengikuti langkah-langkah di notebook Colab Anda) dan menyimpan objek `best_model` serta `scaler` ke file `models/best_model.pkl` dan `models/scaler.pkl`. Pastikan file-file ini berada di lokasi yang benar dalam struktur folder proyek Anda.
    *   Instruksi rinci untuk menyimpan model dari Colab dapat ditemukan di bagian relevan dalam notebook pelatihan Anda atau dokumentasi terkait pickling/joblib di Python.

3.  **Buat dan Aktifkan Virtual Environment (Direkomendasikan):**

bash

# Membuat virtual environment
python -m venv .venv
# Mengaktifkan virtual environment (Windows)
.venv\Scripts\activate
# Mengaktifkan virtual environment (macOS/Linux)
source .venv/bin/activate
4.  **Instal Dependensi:**

bash pip install -r requirements.txt

5.  **Jalankan Aplikasi Streamlit:**
    Pastikan Anda berada di direktori root proyek (`brain-tumor-streamlit-app/`) di terminal, lalu jalankan:

bash streamlit run app.py

6.  Aplikasi akan terbuka di browser web Anda (biasanya di `http://localhost:8501`).

## Hasil (Berdasarkan Notebook)

Berdasarkan analisis dan pelatihan yang dilakukan di notebook Colab, pipeline preprocessing dan ekstraksi fitur berhasil menangkap karakteristik citra MRI. Model klasifikasi terbaik yang diidentifikasi dalam analisis (misalnya, SVM RBF, Random Forest, dll.) mencapai akurasi sebesar **[Sebutkan Akurasi Terbaik dari Output Colab Anda, misal: 90.5%]** dalam mengklasifikasikan citra MRI ke dalam empat kategori (Glioma, Meningioma, Pituitary, No Tumor).

## Analisis Teknik

Proyek ini mengeksplorasi berbagai teknik pemrosesan citra:

*   **Preprocessing:** Gaussian Blur untuk noise reduction, CLAHE untuk contrast enhancement, Min-Max Normalization untuk standardisasi intensitas. Analisis kuantitatif menunjukkan peningkatan kontras dan PSNR.
*   **Transformation & Segmentation:** Canny Edge Detection efektif untuk mengidentifikasi batas, operasi morfologi membantu membersihkan hasil, dan Otsu Thresholding menunjukkan performa yang baik dalam segmentasi area tumor.
*   **Feature Extraction:** Fitur tekstur (GLCM), bentuk (kontur), dan intensitas diekstraksi. Analisis visual distribusi fitur per kelas dan matriks korelasi memberikan wawasan tentang kemampuan diskriminatif fitur-fitur tersebut.

## Kontribusi

Kontribusi pada proyek ini disambut baik. Silakan buka *issue* atau *pull request*.

## Lisensi

Proyek ini dilisensikan di bawah [Sebutkan Lisensi, misal: MIT License] - lihat file [LICENSE](LICENSE) untuk detailnya.

---

**Instruksi Tambahan:**

*   Ganti `<URL_REPOSitori_ANDA>` dengan URL sebenarnya dari repositori GitHub Anda.
*   Buat file `LICENSE` di root folder proyek Anda dan cantumkan teks lisensinya (misal, teks MIT License).
*   Ganti `[Sebutkan Akurasi Terbaik dari Output Colab Anda, misal: 90.5%]` dengan nilai akurasi sebenarnya yang Anda peroleh dari output notebook Colab Anda.
*   Sesuaikan nama folder proyek (`brain-tumor-streamlit-app`) jika Anda menggunakan nama lain.
*   Jika Anda menggunakan metode penyimpanan selain `joblib.dump`, sesuaikan penjelasan di bagian "Dapatkan Model dan Scaler" dan di kode `app.py` yang memuat model.

README ini memberikan gambaran lengkap dan panduan bagi siapa pun yang ingin memahami atau menjalankan proyek Anda.
