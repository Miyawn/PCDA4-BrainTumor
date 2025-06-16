import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
<<<<<<< HEAD
from skimage import feature
from sklearn.preprocessing import StandardScaler
from scipy import stats # Import scipy untuk skewness dan kurtosis
=======
import joblib
from skimage import feature
from sklearn.preprocessing import StandardScaler
from scipy import stats
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Deteksi Tumor Otak",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3

# --- Fungsi Preprocessing ---
def noise_reduction(image, method='gaussian'):
    if method == 'gaussian':
<<<<<<< HEAD
        # Sesuaikan ukuran kernel jika diperlukan
=======
>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3
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
<<<<<<< HEAD
    # Bisa tambahkan adaptive_eq jika mau
    # elif method == 'adaptive_eq':
    #     return cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4)).apply(image)
=======
>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3

def normalize_image(image, method='minmax'):
    if method == 'minmax':
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
<<<<<<< HEAD
    # Bisa tambahkan zscore jika mau
    # elif method == 'zscore':
    #     mean = np.mean(image)
    #     std = np.std(image)
    #     return ((image - mean) / std * 50 + 128).astype(np.uint8)

def preprocess_single_image(image, target_size=(128, 128)):
    # Pastikan citra adalah grayscale
=======

def preprocess_single_image(image, target_size=(128, 128)):
>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize ke ukuran target
    image = cv2.resize(image, target_size)
    # Terapkan pipeline preprocessing
    denoised = noise_reduction(image, 'gaussian')
    enhanced = enhance_contrast(denoised, 'clahe')
    normalized = normalize_image(enhanced, 'minmax')
    return normalized

<<<<<<< HEAD
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
=======
# --- Fungsi Preprocessing dengan Visualisasi ---
def preprocess_with_visualization(image, target_size=(128, 128)):
    """Preprocessing dengan menyimpan setiap tahap untuk visualisasi"""
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    # Resize ke ukuran target
    resized = cv2.resize(gray_image, target_size)
    
    # Noise reduction techniques
    gaussian_blur = noise_reduction(resized, 'gaussian')
    median_blur = noise_reduction(resized, 'median')
    bilateral_blur = noise_reduction(resized, 'bilateral')
    
    # Contrast enhancement techniques
    clahe_enhanced = enhance_contrast(gaussian_blur, 'clahe')
    hist_eq_enhanced = enhance_contrast(gaussian_blur, 'hist_eq')
    
    # Normalization
    normalized = normalize_image(clahe_enhanced, 'minmax')
    
    return {
        'original': resized,
        'gaussian_blur': gaussian_blur,
        'median_blur': median_blur,
        'bilateral_blur': bilateral_blur,
        'clahe_enhanced': clahe_enhanced,
        'hist_eq_enhanced': hist_eq_enhanced,
        'normalized': normalized
    }

# --- Fungsi Image Transformation ---
def apply_edge_detection(image):
    """Deteksi tepi menggunakan berbagai teknik"""
    # Canny edge detection
    canny_edges = cv2.Canny(image, 50, 150)
    
    # Sobel edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    
    # Laplacian edge detection
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    return {
        'canny': canny_edges,
        'sobel': sobel_combined,
        'laplacian': laplacian
    }

def apply_morphological_operations(image):
    """Operasi morfologi untuk membersihkan noise"""
    # Buat binary image terlebih dahulu
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Define kernels
    kernel_small = np.ones((3,3), np.uint8)
    kernel_medium = np.ones((5,5), np.uint8)
    
    # Morphological operations
    erosion = cv2.erode(binary, kernel_small, iterations=1)
    dilation = cv2.dilate(binary, kernel_small, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_medium)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium)
    
    return {
        'binary': binary,
        'erosion': erosion,
        'dilation': dilation,
        'opening': opening,
        'closing': closing
    }

def apply_segmentation_techniques(image):
    """Teknik segmentasi berbeda"""
    segmentations = {}
    
    # Otsu Thresholding
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmentations['otsu'] = otsu
    
    # Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    segmentations['adaptive'] = adaptive
    
    # K-Means Clustering
    data = image.reshape((-1, 1))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    kmeans_result = labels.reshape(image.shape)
    if centers[0] > centers[1]:
        segmentations['kmeans'] = np.uint8(kmeans_result * 255)
    else:
        segmentations['kmeans'] = np.uint8((1 - kmeans_result) * 255)
    
    return segmentations

# --- Fungsi Feature Extraction ---
def extract_texture_features(image):
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    features = []

>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3
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
<<<<<<< HEAD
    # Pastikan citra adalah biner (0 atau 255)
=======
>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3
    if binary_image.max() <= 1:
        binary_image = (binary_image * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
<<<<<<< HEAD
        return [0.0] * 10 # Return float to match other features
=======
        return [0.0] * 10
>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0 or area == 0:
<<<<<<< HEAD
        return [float(area), float(perimeter)] + [0.0] * 8 # Return float for initial features too

    compactness = (perimeter ** 2) / (4 * np.pi * area)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h != 0 else 0.0 # Handle division by zero
    extent = area / (w * h) if w * h != 0 else 0.0 # Handle division by zero
=======
        return [float(area), float(perimeter)] + [0.0] * 8

    compactness = (perimeter ** 2) / (4 * np.pi * area)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h != 0 else 0.0
    extent = area / (w * h) if w * h != 0 else 0.0
>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3

    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    equiv_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0.0

    moments = cv2.moments(largest_contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        mu20 = moments['mu20'] / moments['m00']
<<<<<<< HEAD
        mu02 = moments['mu02'] / moments['m00'] # Perbaikan indeks
        mu11 = moments['mu11'] / moments['m00']
        # Hindari division by zero jika mu20 + mu02 sangat kecil
=======
        mu02 = moments['mu02'] / moments['m00']
        mu11 = moments['mu11'] / moments['m00']
>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3
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

<<<<<<< HEAD
    # Pastikan image flattenable
=======
>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3
    flattened_image = image.flatten()

    hist, _ = np.histogram(flattened_image, bins=256, range=[0, 256])
    hist = hist / np.sum(hist)
<<<<<<< HEAD
    # Hindari log(0)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    # Pastikan data memiliki variansi untuk skewness/kurtosis
=======
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3
    if len(flattened_image) < 2 or np.std(flattened_image) == 0:
         skewness = 0.0
         kurtosis = 0.0
    else:
        skewness = stats.skew(flattened_image)
        kurtosis = stats.kurtosis(flattened_image)

<<<<<<< HEAD

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
=======
    return [float(mean_intensity), float(std_intensity), float(min_intensity), float(max_intensity),
            float(entropy), float(skewness), float(kurtosis)]

# --- Fungsi Segmentasi ---
def image_segmentation(image, method='otsu'):
    if method == 'otsu':
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3
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
<<<<<<< HEAD
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
=======
        data = image.reshape((-1, 1))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        ret, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        segmented = labels.reshape(image.shape)
>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3
        if centers[0] > centers[1]:
            return np.uint8(segmented * 255)
        else:
            return np.uint8((1 - segmented) * 255)

<<<<<<< HEAD

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
=======
# --- Fungsi untuk menggabungkan ekstraksi fitur ---
def extract_features_from_image(image):
    processed_img = preprocess_single_image(image)
    texture_feat = extract_texture_features(processed_img)
    segmented_img = image_segmentation(processed_img, 'otsu')
    shape_feat = extract_shape_features(segmented_img)
    intensity_feat = extract_intensity_features(processed_img)
    all_features = np.hstack([texture_feat, shape_feat, intensity_feat])
    return all_features, processed_img, segmented_img

# --- Load Model Package ---
@st.cache_resource
def load_model_package():
    try:
        # Try to load the complete model package first
        model_package_path = 'models/brain_tumor_model.pkl'
        if os.path.exists(model_package_path):
            model_package = joblib.load(model_package_path)
            return model_package, True, "Paket model berhasil dimuat"
        
        # Fallback to individual components
        model_path = 'models/best_model.pkl'
        scaler_path = 'models/feature_scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Create a minimal package
            model_package = {
                'model': model,
                'scaler': scaler,
                'class_names': ['glioma', 'meningioma', 'normal', 'pituitary'],
                'best_classifier_name': 'Tidak Diketahui'
            }
            return model_package, True, "Komponen individual berhasil dimuat"
        
        # Try original file names from your code
        model_path_alt = 'models/best-model.pkl'
        scaler_path_alt = 'models/scaler.pkl'
        
        if os.path.exists(model_path_alt) and os.path.exists(scaler_path_alt):
            with open(model_path_alt, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path_alt, 'rb') as f:
                scaler = pickle.load(f)
            
            model_package = {
                'model': model,
                'scaler': scaler,
                'class_names': ['glioma', 'meningioma', 'normal', 'pituitary'],
                'best_classifier_name': 'Tidak Diketahui'
            }
            return model_package, True, "Komponen lama berhasil dimuat"
        
        return None, False, "File model tidak ditemukan"
        
    except Exception as e:
        return None, False, f"Kesalahan saat memuat model: {str(e)}"

# --- Prediction Function ---
def predict_brain_tumor(image_array, model_package):
    try:
        # Extract features
        features, processed_img, segmented_img = extract_features_from_image(image_array)
        
        # Handle NaN/inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        features_scaled = model_package['scaler'].transform(features.reshape(1, -1))
        
        # Predict
        prediction = model_package['model'].predict(features_scaled)[0]
        
        # Get probabilities if available
        if hasattr(model_package['model'], 'predict_proba'):
            probabilities = model_package['model'].predict_proba(features_scaled)[0]
            prob_dict = {model_package['class_names'][i]: float(prob) for i, prob in enumerate(probabilities)}
        else:
            prob_dict = {model_package['class_names'][i]: 1.0 if i == prediction else 0.0 
                        for i in range(len(model_package['class_names']))}
        
        result = {
            'predicted_class': model_package['class_names'][prediction],
            'confidence': max(prob_dict.values()),
            'probabilities': prob_dict,
            'processed_image': processed_img,
            'segmented_image': segmented_img,
            'features': features
        }
        
        return result
        
    except Exception as e:
        return {'error': str(e)}

# --- Fungsi untuk menampilkan visualisasi preprocessing ---
def display_preprocessing_steps(image_array):
    """Menampilkan setiap tahap preprocessing"""
    st.subheader("🔧 Tahapan Image Preprocessing")
    
    # Dapatkan hasil preprocessing
    preprocessing_results = preprocess_with_visualization(image_array)
    
    # Penjelasan teknik
    st.markdown("""
    **Teknik noise reduction** seperti Gaussian dan median filter dipilih untuk mengurangi gangguan sinyal (noise) yang umum muncul 
    akibat proses pengambilan gambar MRI tanpa menghilangkan informasi penting. **Contrast enhancement** seperti CLAHE 
    (Contrast Limited Adaptive Histogram Equalization) digunakan karena kontras yang rendah pada beberapa bagian otak dapat 
    menyulitkan dalam mengamati struktur tumor secara jelas. **Normalisasi** digunakan untuk menyeragamkan rentang intensitas 
    antar citra, yang sangat penting agar model analisis berikutnya tidak bias terhadap nilai intensitas tertentu.
    """)
    
    # Tampilkan original image
    st.write("#### 📷 Gambar Original (Grayscale & Resized)")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(preprocessing_results['original'], caption='Gambar Original (128x128)', use_container_width=True, clamp=True)
    
    # Noise Reduction Techniques
    st.write("#### 🔍 Noise Reduction Techniques")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(preprocessing_results['gaussian_blur'], 
                caption='Gaussian Blur\n(Mengurangi noise dengan blur yang merata)', 
                use_container_width=True, clamp=True)
    
    with col2:
        st.image(preprocessing_results['median_blur'], 
                caption='Median Filter\n(Efektif untuk salt-and-pepper noise)', 
                use_container_width=True, clamp=True)
    
    with col3:
        st.image(preprocessing_results['bilateral_blur'], 
                caption='Bilateral Filter\n(Mempertahankan tepi sambil mengurangi noise)', 
                use_container_width=True, clamp=True)
    
    # Contrast Enhancement
    st.write("#### ✨ Contrast Enhancement Techniques")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(preprocessing_results['clahe_enhanced'], 
                caption='CLAHE (Contrast Limited Adaptive Histogram Equalization)\nMeningkatkan kontras lokal tanpa over-amplification', 
                use_container_width=True, clamp=True)
    
    with col2:
        st.image(preprocessing_results['hist_eq_enhanced'], 
                caption='Histogram Equalization\nMeratakan distribusi intensitas global', 
                use_container_width=True, clamp=True)
    
    # Final Normalized Image
    st.write("#### 📏 Normalization")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(preprocessing_results['normalized'], 
                caption='Normalized Image (Min-Max Scaling)\nMenyeragamkan rentang intensitas 0-255', 
                use_container_width=True, clamp=True)
    
    return preprocessing_results['normalized']

def display_transformation_steps(processed_image):
    """Menampilkan tahapan image transformation"""
    st.subheader("🔄 Image Transformation & Feature Extraction")
    
    st.markdown("""
    Setelah preprocessing, dilakukan **image transformation** untuk mengekstrak fitur visual penting. **Edge detection** seperti Canny dan Sobel 
    digunakan karena batas tumor biasanya muncul dalam bentuk perubahan intensitas yang tajam. **Morphological operations** dipilih untuk 
    membersihkan noise setelah segmentasi dan memperjelas bentuk objek. **Segmentation techniques** seperti Otsu Thresholding dan K-Means 
    Clustering diterapkan untuk memisahkan area tumor dari latar belakang, sehingga informasi spasial tumor bisa dianalisis lebih lanjut.
    """)
    
    # Edge Detection
    st.write("#### 🔍 Edge Detection Techniques")
    edge_results = apply_edge_detection(processed_image)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(edge_results['canny'], 
                caption='Canny Edge Detection\nMendeteksi tepi dengan noise reduction dan hysteresis', 
                use_container_width=True, clamp=True)
    
    with col2:
        st.image(edge_results['sobel'], 
                caption='Sobel Edge Detection\nMendeteksi gradien intensitas dalam arah X dan Y', 
                use_container_width=True, clamp=True)
    
    with col3:
        st.image(edge_results['laplacian'], 
                caption='Laplacian Edge Detection\nMendeteksi perubahan intensitas kedua (second derivative)', 
                use_container_width=True, clamp=True)
    
    # Morphological Operations
    st.write("#### 🔬 Morphological Operations")
    morph_results = apply_morphological_operations(processed_image)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.image(morph_results['binary'], 
                caption='Binary Image\nThreshold dasar untuk operasi morfologi', 
                use_container_width=True, clamp=True)
    
    with col2:
        st.image(morph_results['erosion'], 
                caption='Erosion\nMengecilkan objek putih', 
                use_container_width=True, clamp=True)
    
    with col3:
        st.image(morph_results['dilation'], 
                caption='Dilation\nMemperbesar objek putih', 
                use_container_width=True, clamp=True)
    
    with col4:
        st.image(morph_results['opening'], 
                caption='Opening\nMenghilangkan noise kecil', 
                use_container_width=True, clamp=True)
    
    with col5:
        st.image(morph_results['closing'], 
                caption='Closing\nMengisi lubang kecil dalam objek', 
                use_container_width=True, clamp=True)
    
    # Segmentation Techniques
    st.write("#### 🎯 Segmentation Techniques")
    seg_results = apply_segmentation_techniques(processed_image)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(seg_results['otsu'], 
                caption='Otsu Thresholding\nOtomatis menentukan threshold optimal', 
                use_container_width=True, clamp=True)
    
    with col2:
        st.image(seg_results['adaptive'], 
                caption='Adaptive Thresholding\nThreshold berbeda untuk setiap region', 
                use_container_width=True, clamp=True)
    
    with col3:
        st.image(seg_results['kmeans'], 
                caption='K-Means Clustering\nPengelompokan pixel berdasarkan intensitas', 
                use_container_width=True, clamp=True)
    
    return seg_results['otsu']  # Return segmented image for feature extraction

# --- Main App ---
def main():
    st.title("🧠 Deteksi Tumor Otak dari Citra MRI")
    st.markdown("---")
    
    # Load model
    model_package, model_loaded, load_message = load_model_package()
    
    # Sidebar info
    with st.sidebar:
        st.header("📋 Informasi")
        if model_loaded:
            st.success(load_message)
            if 'best_classifier_name' in model_package:
                st.info(f"**Model**: {model_package['best_classifier_name']}")
            if 'training_date' in model_package:
                st.info(f"**Tanggal Pelatihan**: {model_package['training_date']}")
        else:
            st.error(load_message)
            st.error("Pastikan file model berada di folder 'models/'")
        
        st.markdown("### 🎯 Kelas yang Didukung")
        if model_loaded:
            for cls in model_package['class_names']:
                if cls == 'notumor' or cls == 'normal':
                    st.write(f"• Normal (Tanpa Tumor)")
                else:
                    st.write(f"• {cls.title()}")
    
    # Main content
    st.header("📤 Unggah Gambar MRI")
    uploaded_file = st.file_uploader(
        "Pilih gambar MRI...", 
        type=["jpg", "jpeg", "png"],
        help="Unggah gambar scan MRI otak dalam format grayscale atau berwarna"
    )
    
    if uploaded_file is not None:
        # Read and display original image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert PIL to OpenCV format
        if len(image_array.shape) == 3:
            opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            opencv_image = image_array
        
        # Display original image
        st.subheader("📷 Gambar MRI Original")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption='Gambar MRI Asli', use_container_width=True)
        
        # Display preprocessing steps
        processed_image = display_preprocessing_steps(opencv_image)
        
        # Display transformation steps
        segmented_image = display_transformation_steps(processed_image)
        
        # Prediction section
        if model_loaded:
            st.header("🔍 Hasil Analisis dan Prediksi")

            with st.spinner("Memproses gambar dan mengekstraksi fitur..."):
                result = predict_brain_tumor(opencv_image, model_package)

            if 'error' not in result:
                col1, col2 = st.columns(2)

                with col1:
                    predicted_class = result['predicted_class']
                    confidence = result['confidence']

                    if predicted_class in ['notumor', 'normal']:
                        st.success(f"✅ **Tidak Ada Tumor Terdeteksi**")
                        st.success(f"Tingkat Kepercayaan: {confidence:.2%}")
                    else:
                        st.warning(f"⚠️ **{predicted_class.title()} Terdeteksi**")
                        st.warning(f"Tingkat Kepercayaan: {confidence:.2%}")

                with col2:
                    st.subheader("📊 Probabilitas Klasifikasi")
                    prob_df = pd.DataFrame(
                        list(result['probabilities'].items()),
                        columns=['Kelas', 'Probabilitas']
                    )

                    class_translation = {
                        'notumor': 'Normal (Tanpa Tumor)',
                        'normal': 'Normal (Tanpa Tumor)',
                        'glioma': 'Glioma',
                        'meningioma': 'Meningioma',
                        'pituitary': 'Pituitary'
                    }
                    prob_df['Kelas'] = prob_df['Kelas'].map(class_translation).fillna(prob_df['Kelas'])
                    prob_df = prob_df.sort_values('Probabilitas', ascending=False)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.barh(prob_df['Kelas'], prob_df['Probabilitas'])
                    ax.set_xlabel('Probabilitas')
                    ax.set_xlim(0, 1)

                    colors = ['green' if 'Normal' in cls else 'red' for cls in prob_df['Kelas']]
                    for bar, color in zip(bars, colors):
                        bar.set_color(color)
                        bar.set_alpha(0.7)

                    plt.tight_layout()
                    st.pyplot(fig)

                # Analisis fitur
                st.subheader("🔬 Analisis Fitur yang Diekstraksi")
                st.write(f"**Total Fitur yang Diekstraksi**: {len(result['features'])}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**🎨 Fitur Tekstur (10 fitur)**")
                    st.write("- Contrast")
                    st.write("- Dissimilarity") 
                    st.write("- Homogeneity")
                    st.write("- Energy")
                    st.write("- Correlation")
                    st.write("*Berbasis Gray Level Co-occurrence Matrix (GLCM)*")

                with col2:
                    st.markdown("**📐 Fitur Bentuk (10 fitur)**")
                    st.write("- Area")
                    st.write("- Perimeter")
                    st.write("- Compactness")
                    st.write("- Aspect Ratio")
                    st.write("- Extent")
                    st.write("- Solidity")
                    st.write("- Equivalent Diameter")
                    st.write("- Centroid (X, Y)")
                    st.write("- Eccentricity")
                    st.write("*Berbasis analisis kontur*")

                with col3:
                    st.markdown("**💡 Fitur Intensitas (7 fitur)**")
                    st.write("- Mean Intensity")
                    st.write("- Standard Deviation")
                    st.write("- Min/Max Intensity")
                    st.write("- Entropy")
                    st.write("- Skewness")
                    st.write("- Kurtosis")
                    st.write("*Berbasis statistik histogram*")

                # Tampilkan ringkasan nilai fitur
                st.markdown("#### 📈 Nilai Fitur yang Diekstraksi")
                feature_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation'] * 2 + \
                                ['Area', 'Perimeter', 'Compactness', 'Aspect_Ratio', 'Extent',
                                'Solidity', 'Equiv_Diameter', 'Centroid_X', 'Centroid_Y', 'Eccentricity'] + \
                                ['Mean_Intensity', 'Std_Intensity', 'Min_Intensity', 'Max_Intensity',
                                'Entropy', 'Skewness', 'Kurtosis']

                features_df = pd.DataFrame({
                    'Fitur': feature_names[:len(result['features'])],
                    'Nilai': result['features']
                })

                st.dataframe(features_df.head(15), use_container_width=True)

                if len(features_df) > 15:
                    with st.expander("📄 Lihat Semua Fitur"):
                        st.dataframe(features_df, use_container_width=True)

            else:
                st.error(f"Kesalahan saat prediksi: {result['error']}")
        else:
            st.error("Tidak dapat melakukan prediksi - model belum dimuat")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p><strong>⚠️ Disclaimer:</strong> Aplikasi ini merupakan bagian dari tugas besar mata kuliah Pengolahan Citra Digital dan dikembangkan untuk tujuan edukatif. Model yang digunakan bukan untuk diagnosis medis dan hasil prediksi tidak dapat dijadikan acuan klinis.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
>>>>>>> 3a8c3a58a8e1bd428a0c3625ca942311054403e3
