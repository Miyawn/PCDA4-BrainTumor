import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
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

# --- Fungsi Preprocessing ---
def noise_reduction(image, method='gaussian'):
    if method == 'gaussian':
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

def normalize_image(image, method='minmax'):
    if method == 'minmax':
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def preprocess_single_image(image, target_size=(128, 128)):
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
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    features = []

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
    if binary_image.max() <= 1:
        binary_image = (binary_image * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return [0.0] * 10
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0 or area == 0:
        return [float(area), float(perimeter)] + [0.0] * 8

    compactness = (perimeter ** 2) / (4 * np.pi * area)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h != 0 else 0.0
    extent = area / (w * h) if w * h != 0 else 0.0

    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    equiv_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0.0

    moments = cv2.moments(largest_contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        mu20 = moments['mu20'] / moments['m00']
        mu02 = moments['mu02'] / moments['m00']
        mu11 = moments['mu11'] / moments['m00']
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

    flattened_image = image.flatten()

    hist, _ = np.histogram(flattened_image, bins=256, range=[0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    if len(flattened_image) < 2 or np.std(flattened_image) == 0:
         skewness = 0.0
         kurtosis = 0.0
    else:
        skewness = stats.skew(flattened_image)
        kurtosis = stats.kurtosis(flattened_image)

    return [float(mean_intensity), float(std_intensity), float(min_intensity), float(max_intensity),
            float(entropy), float(skewness), float(kurtosis)]

# --- Fungsi Segmentasi ---
def image_segmentation(image, method='otsu'):
    if method == 'otsu':
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        data = image.reshape((-1, 1))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        ret, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        segmented = labels.reshape(image.shape)
        if centers[0] > centers[1]:
            return np.uint8(segmented * 255)
        else:
            return np.uint8((1 - segmented) * 255)

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
    col1, col2 = st.columns([1, 1])
    
    with col1:
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
            
            st.image(image, caption='Gambar MRI Asli', use_container_width=True)
            
            # Convert PIL to OpenCV format
            if len(image_array.shape) == 3:
                opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                opencv_image = image_array
    
    with col2:
        if uploaded_file is not None and model_loaded:
            st.header("🔍 Hasil Analisis")
            
            with st.spinner("Memproses gambar dan mengekstraksi fitur..."):
                result = predict_brain_tumor(opencv_image, model_package)
            
            if 'error' not in result:
                # Display prediction
                predicted_class = result['predicted_class']
                confidence = result['confidence']
                
                if predicted_class in ['notumor', 'normal']:
                    st.success(f"✅ **Tidak Ada Tumor Terdeteksi**")
                    st.success(f"Tingkat Kepercayaan: {confidence:.2%}")
                else:
                    st.warning(f"⚠️ **{predicted_class.title()} Terdeteksi**")
                    st.warning(f"Tingkat Kepercayaan: {confidence:.2%}")
                
                # Show probabilities
                st.subheader("📊 Probabilitas Klasifikasi")
                prob_df = pd.DataFrame(
                    list(result['probabilities'].items()),
                    columns=['Kelas', 'Probabilitas']
                )
                # Translate class names to Indonesian
                class_translation = {
                    'notumor': 'Normal (Tanpa Tumor)',
                    'normal': 'Normal (Tanpa Tumor)',
                    'glioma': 'Glioma',
                    'meningioma': 'Meningioma',
                    'pituitary': 'Pituitary'
                }
                prob_df['Kelas'] = prob_df['Kelas'].map(class_translation).fillna(prob_df['Kelas'])
                prob_df = prob_df.sort_values('Probabilitas', ascending=False)
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.barh(prob_df['Kelas'], prob_df['Probabilitas'])
                ax.set_xlabel('Probabilitas')
                ax.set_xlim(0, 1)
                
                # Color bars
                colors = ['green' if 'Normal' in cls else 'red' for cls in prob_df['Kelas']]
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                    bar.set_alpha(0.7)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show processed images
                st.subheader("🖼️ Tahapan Pemrosesan Gambar")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(result['processed_image'], 
                            caption='Gambar Setelah Preprocessing', 
                            use_container_width=True, 
                            clamp=True)
                
                with col_b:
                    st.image(result['segmented_image'], 
                            caption='Gambar Tersegmentasi', 
                            use_container_width=True, 
                            clamp=True)
                
                # Feature information
                with st.expander("🔬 Analisis Fitur"):
                    st.write(f"**Total Fitur yang Diekstraksi**: {len(result['features'])}")
                    st.write("- Fitur Tekstur: 10 (berbasis GLCM)")
                    st.write("- Fitur Bentuk: 10 (berbasis Kontur)")
                    st.write("- Fitur Intensitas: 7 (Statistik)")
                
            else:
                st.error(f"Kesalahan saat prediksi: {result['error']}")
        
        elif uploaded_file is not None and not model_loaded:
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