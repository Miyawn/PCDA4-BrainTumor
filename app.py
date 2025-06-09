import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from skimage import feature
from sklearn.preprocessing import StandardScaler
from scipy import stats # Import scipy untuk skewness dan kurtosis

# --- Fungsi Preprocessing ---
def noise_reduction(image, method='gaussian'):
    if method == 'gaussian':
        # Sesuaikan ukuran kernel jika diperlukan
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'median':
        return cv2.medianBlur(image, 5)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)

def enhance_contrast(image, method='clahe'):
    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    elif method == 'hist_eq':
        return cv2.equalizeHist(image)
    # Bisa tambahkan adaptive_eq jika mau
    # elif method == 'adaptive_eq':
    #     return cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4)).apply(image)

def normalize_image(image, method='minmax'):
    if method == 'minmax':
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    # Bisa tambahkan zscore jika mau
    # elif method == 'zscore':
    #     mean = np.mean(image)
    #     std = np.std(image)
    #     return ((image - mean) / std * 50 + 128).astype(np.uint8)

def preprocess_single_image(image, target_size=(128, 128)):
    # Pastikan citra adalah grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize ke ukuran target
    image = cv2.resize(image, target_size)
    # Terapkan pipeline preprocessing
    denoised = noise_reduction(image, 'gaussian')
    enhanced = enhance_contrast(denoised, 'clahe')
    normalized = normalize_image(enhanced, 'minmax')
    return normalized

# --- Fungsi Feature Extraction ---
def extract_texture_features(image):
    # Resize sudah dilakukan di preprocess_single_image, tapi pastikan ukurannya
    # sesuai dengan yang digunakan saat pelatihan GLCM
    # if image.shape != (128, 128):
    #    image = cv2.resize(image, (128, 128)) # Opsional, jika yakin preprocess_single_image sudah resize
    distances = [1, 2] # Gunakan jarak yang sama seperti saat pelatihan
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # Gunakan sudut dalam radian untuk skimage
    features = []

    # Pastikan citra memiliki kedalaman bit yang sesuai untuk GLCM (misal: 8-bit)
    image_int = image.astype(np.uint8)

    for distance in distances:
        glcm = feature.graycomatrix(image_int, [distance], angles, symmetric=True, normed=True)
        contrast = feature.graycoprops(glcm, 'contrast').mean()
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        energy = feature.graycoprops(glcm, 'energy').mean()
        correlation = feature.graycoprops(glcm, 'correlation').mean()
        features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
    return features

def extract_shape_features(binary_image):
    # Pastikan citra adalah biner (0 atau 255)
    if binary_image.max() <= 1:
        binary_image = (binary_image * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return [0.0] * 10 # Return float to match other features
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0 or area == 0:
        return [float(area), float(perimeter)] + [0.0] * 8 # Return float for initial features too

    compactness = (perimeter ** 2) / (4 * np.pi * area)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h != 0 else 0.0 # Handle division by zero
    extent = area / (w * h) if w * h != 0 else 0.0 # Handle division by zero

    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    equiv_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0.0

    moments = cv2.moments(largest_contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        mu20 = moments['mu20'] / moments['m00']
        mu02 = moments['mu02'] / moments['m00'] # Perbaikan indeks
        mu11 = moments['mu11'] / moments['m00']
        # Hindari division by zero jika mu20 + mu02 sangat kecil
        denom = mu20 + mu02
        eccentricity = ((mu20 - mu02)**2 + 4*mu11**2)**0.5 / (denom) if denom != 0 else 0.0
    else:
        cx, cy, eccentricity = 0, 0, 0.0

    return [float(area), float(perimeter), float(compactness), float(aspect_ratio), float(extent),
            float(solidity), float(equiv_diameter), float(cx), float(cy), float(eccentricity)]

def extract_intensity_features(image):
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    min_intensity = np.min(image)
    max_intensity = np.max(image)

    # Pastikan image flattenable
    flattened_image = image.flatten()

    hist, _ = np.histogram(flattened_image, bins=256, range=[0, 256])
    hist = hist / np.sum(hist)
    # Hindari log(0)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    # Pastikan data memiliki variansi untuk skewness/kurtosis
    if len(flattened_image) < 2 or np.std(flattened_image) == 0:
         skewness = 0.0
         kurtosis = 0.0
    else:
        skewness = stats.skew(flattened_image)
        kurtosis = stats.kurtosis(flattened_image)


    return [float(mean_intensity), float(std_intensity), float(min_intensity), float(max_intensity),
            float(entropy), float(skewness), float(kurtosis)]


# --- Fungsi Segmentasi (diperlukan untuk shape features) ---
# Salin fungsi image_segmentation dari notebook Anda
def image_segmentation(image, method='otsu'):
    if method == 'otsu':
        # Pastikan image adalah grayscale sebelum Otsu
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Pastikan image bertipe yang sesuai, misal cv2.CV_8U
        image_8u = cv2.convertScaleAbs(image)
        _, binary = cv2.threshold(image_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    elif method == 'adaptive':
         if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         image_8u = cv2.convertScaleAbs(image)
         return cv2.adaptiveThreshold(image_8u, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    elif method == 'kmeans':
        # K-means memerlukan data float32
        data = image.reshape((-1, 1))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        # K-means pada dasarnya mengelompokkan intensitas, hasilnya perlu di-reshape
        # Label yang dihasilkan adalah 0 atau 1 (untuk 2 klaster)
        ret, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Pilih label mana yang merepresentasikan foreground (tumor)
        # Salah satu cara adalah melihat center intensitasnya
        # Cluster dengan intensitas lebih tinggi kemungkinan foreground
        segmented = labels.reshape(image.shape)
        # Pastikan output adalah biner (0 atau 255)
        if centers[0] > centers[1]:
            return np.uint8(segmented * 255)
        else:
            return np.uint8((1 - segmented) * 255)


# --- Fungsi untuk menggabungkan ekstraksi fitur ---
def extract_features_from_image(image):
    # Preprocess citra terlebih dahulu
    processed_img = preprocess_single_image(image)

    # Ekstraksi fitur tekstur dari citra yang sudah diproses
    texture_feat = extract_texture_features(processed_img)

    # Ekstraksi fitur bentuk memerlukan citra biner/segmentasi
    # Gunakan Otsu pada citra yang sudah diproses
    segmented_img = image_segmentation(processed_img, 'otsu')
    shape_feat = extract_shape_features(segmented_img)

    # Ekstraksi fitur intensitas dari citra yang sudah diproses
    intensity_feat = extract_intensity_features(processed_img)

    # Gabungkan semua fitur menjadi satu array
    all_features = np.hstack([texture_feat, shape_feat, intensity_feat])
    return all_features

# --- Load Model dan Scaler ---
# Pastikan file 'best-model.pkl' dan 'scaler.pkl' ada di dalam folder 'models/'
model_path = 'models/best-model.pkl'
scaler_path = 'models/scaler.pkl'

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    model_loaded = True
    st.success("Model dan scaler berhasil dimuat.")
except FileNotFoundError:
    st.error(f"Error: Model atau scaler belum ditemukan. Pastikan file '{os.path.basename(model_path)}' dan '{os.path.basename(scaler_path)}' ada di dalam folder 'models/'.")
    model_loaded = False
except Exception as e:
    st.error(f"Error saat memuat model atau scaler: {e}")
    model_loaded = False


# --- Streamlit UI ---
st.title("Deteksi Tumor Otak dari Citra MRI")
st.write("Unggah citra MRI (format JPG, JPEG, atau PNG) untuk dideteksi keberadaan dan jenis tumor.")

uploaded_file = st.file_uploader("Pilih citra MRI...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca citra dari file yang diunggah
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Gunakan cv2.IMREAD_UNCHANGED untuk membaca citra apa adanya (bisa grayscale atau warna)
    opencv_image_raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    # Tampilkan citra asli
    st.image(opencv_image_raw, caption='Citra yang Diunggah', use_column_width=True)

    if model_loaded:
        try:
            # Lakukan preprocessing dan ekstraksi fitur
            with st.spinner("Memproses citra dan mengekstrak fitur..."):
                # Pastikan citra diubah ke grayscale sebelum preprocessing jika belum
                if len(opencv_image_raw.shape) == 3:
                    opencv_image_gray = cv2.cvtColor(opencv_image_raw, cv2.COLOR_BGR2GRAY)
                else:
                    opencv_image_gray = opencv_image_raw # Citra sudah grayscale

                # Preprocessing (sekaligus resize)
                processed_image = preprocess_single_image(opencv_image_gray, target_size=(128, 128))

                # Ekstraksi fitur
                features = extract_features_from_image(opencv_image_gray) # Gunakan citra asli grayscale untuk ekstraksi fitur jika fungsi Anda membutuhkannya sebelum preprocessing, atau processed_image jika fitur diekstrak dari hasil preprocessing

                # Handle potential NaN or infinite values
                features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

                # Scaling fitur
                features_scaled = scaler.transform(features.reshape(1, -1))

            # Prediksi
            with st.spinner("Melakukan klasifikasi..."):
                prediction = model.predict(features_scaled)
                # Pastikan urutan kelas sesuai dengan yang digunakan saat pelatihan model
                classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
                predicted_class = classes[prediction[0]]

            st.subheader("Hasil Deteksi:")
            if predicted_class == 'notumor':
                st.success(f"Hasil: Tidak terdeteksi tumor.")
            else:
                st.warning(f"Hasil: Terdeteksi **{predicted_class.replace('tumor', ' tumor').title()}**.")

            st.write("Catatan: Hasil ini adalah berdasarkan model Machine Learning dan bukan diagnosis medis. Selalu konsultasikan dengan profesional medis.")

        except Exception as e:
            st.error(f"Terjadi error saat memproses citra atau melakukan prediksi: {e}")

    else:
        st.warning("Tidak dapat melakukan klasifikasi karena model belum dimuat.")