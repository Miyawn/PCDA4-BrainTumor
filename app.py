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

# --- Custom CSS for enhanced aesthetics with the new theme ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Overall background and text color */
    .stApp {
        background-color: #FFF0F6; /* Blush Tint */
        color: #4A4A4A; /* Deep Gray */
    }

    /* Main content area padding - NEW/ADJUSTED */
    .main .block-container {
        padding-top: 3rem; /* Adjusted for more space from the top */
        padding-right: 5rem; /* Adjusted for more space on the sides */
        padding-left: 5rem;  /* Adjusted for more space on the sides */
        padding-bottom: 3rem;
    }

    /* Header styling - ADJUSTED MARGINS */
    h1, h2, h3, h4, h5, h6 {
        color: #F06292; /* Warm Pink for headers */
        margin-top: 1.8em; /* More space above headings */
        margin-bottom: 0.7em; /* Consistent space below headings */
    }
    h1 {
        padding-top: 10px; /* Space from top of page title */
        margin-bottom: 0.5em; /* Slightly less space for main title */
    }

    /* Main title */
    .stApp > header {
        background-color: transparent;
    }
    .stApp > header h1 {
        color: #F48FB1; /* Pink Rose for main title */
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Sidebar styling (kept as is, focus on main) */
    .stSidebar {
        background-color: #FCE4EC; /* Light Pink */
        border-right: 2px solid #F48FB1; /* Pink Rose border */
        color: #4A4A4A; /* Deep Gray */
        padding-top: 30px; /* More padding at the top of sidebar */
    }
    .stSidebar .stSelectbox > label, .stSidebar .stFileUploader > label {
        color: #4A4A4A;
    }
    .stSidebar .stInfo, .stSidebar .stSuccess, .stSidebar .stError {
        background-color: rgba(244, 143, 177, 0.1); /* Slightly transparent Pink Rose */
        border-left: 5px solid;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        color: inherit; /* Inherit text color */
    }
    .stSidebar .stSuccess { border-color: #4CAF50; } /* Green for success */
    .stSidebar .stInfo { border-color: #2196F3; } /* Blue for info */
    .stSidebar .stError { border-color: #F44336; } /* Red for error */

    /* File uploader button */
    .stFileUploader > div > button {
        background-color: #F48FB1; /* Pink Rose */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1em;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    .stFileUploader > div > button:hover {
        background-color: #F06292; /* Warm Pink */
        transform: translateY(-2px);
    }
    /* File uploader input text */
    .stFileUploader > div > div > div > div {
        color: #4A4A4A; /* Deep Gray */
    }


    /* Tabs styling - REFINED FOR VISUAL CLARITY */

/* ===================================================================
   KODE CSS UNTUK STYLING STREAMLIT TABS
   =================================================================== */

    /* Mengatur wadah utama semua tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; /* Jarak antar tab */
        justify-content: center; /* Posisi di tengah */
        margin-bottom: 20px;
    }

    /* Mengatur tombol tab secara umum */
    .stTabs [data-baseweb="tab"] {
        height: 55px; /* Sedikit lebih tinggi */
        white-space: nowrap;
        border-radius: 8px; /* Sudut lebih melengkung */
        gap: 10px;
        padding: 0px 25px;
        transition: all 0.2s ease-in-out;
        border: 2px solid transparent; /* Border transparan untuk transisi hover */
    }

    /* === PENGATURAN FONT UNTUK 3 STATE === */

    /* 1. FONT WARNA UNTUK TAB DIAM (INACTIVE) */
    .stTabs [data-baseweb="tab"] > div[data-testid="stText"] {
        background-color: #FCE4EC; /* Latar pink muda */
        color: #C2185B !important; /* FONT: Warna pink tua agar kontras */
        font-weight: 600;
        opacity: 0.8;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        width: 100%;
        border-radius: 6px;
        transition: all 0.2s ease-in-out;
    }

    /* 2. FONT WARNA SAAT TAB DISOROT (HOVER) */
    .stTabs [data-baseweb="tab"]:hover > div[data-testid="stText"] {
        background-color: #F48FB1; /* Latar pink medium saat hover */
        color: #FFFFFF !important; /* FONT: Warna putih saat hover */
        opacity: 1;
    }

    /* 3. FONT WARNA UNTUK TAB AKTIF (ACTIVE) */
    .stTabs [data-baseweb="tab"][aria-selected="true"] > div[data-testid="stText"] {
        background-color: #E91E63; /* Latar pink magenta solid */
        color: #FFFFFF !important; /* FONT: Warna putih untuk kontras maksimal */
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    /* Menghapus background bawaan dari Streamlit agar style kita berlaku penuh */
    .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: transparent !important;
        border: none !important;
    }

    /* Container for sections - ADJUSTED PADDING and subtle border/shadow */
    .stContainer {
        background-color: #FCE4EC; /* Light Pink */
        padding: 40px; /* Generous padding for more breathing room */
        border-radius: 12px; /* Slightly more rounded corners */
        margin-bottom: 25px; /* More space between containers */
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1); /* More pronounced, soft shadow */
        border: 1px solid rgba(244, 143, 177, 0.4); /* Slightly stronger border */
    }

    /* Metrics (for prediction result) */
    [data-testid="stMetric"] {
        background-color: #FFF0F6; /* Blush Tint - slightly different from container */
        padding: 25px; /* Increased padding */
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08); /* Slightly more prominent shadow */
        text-align: center;
        border: 1px solid #F48FB1; /* Pink Rose border */
    }
    [data-testid="stMetric"] label {
        font-size: 1.2em;
        font-weight: bold;
        color: #F48FB1; /* Pink Rose */
    }
    [data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 2.8em; /* Slightly larger value */
        font-weight: 700;
        color: #F06292; /* Warm Pink (default for metric value) */
    }
    [data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-size: 1.6em; /* Slightly larger delta */
        color: #4A4A4A; /* Deep Gray */
    }
    /* Specific colors for prediction results */
    .stSuccess [data-testid="stMetricValue"] { color: #4CAF50; } /* Green for normal */
    .stWarning [data-testid="stMetricValue"] { color: #F06292; } /* Warm Pink for tumor */
    .stSuccess [data-testid="stMetricDelta"] { color: #4A4A4A; }
    .stWarning [data-testid="stMetricDelta"] { color: #4A4A4A; }

    /* Expander styling */
    .stExpander {
        background-color: #FCE4EC; /* Light Pink */
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
    }
    .stExpander [data-baseweb="accordion"] > div > div {
        color: #F06292; /* Warm Pink */
        font-weight: 600;
    }
    .stExpander [data-baseweb="accordion"] > div > div[aria-expanded="true"] {
        color: #F48FB1; /* Pink Rose when expanded */
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    .dataframe th {
        background-color: #F06292; /* Warm Pink */
        color: white;
        font-weight: bold;
    }
    .dataframe td {
        background-color: #FFF0F6; /* Blush Tint */
        color: #4A4A4A; /* Deep Gray */
    }

    /* Footer styling */
    .stApp footer {
        color: #777;
        font-size: 0.9em;
        text-align: center;
        padding-top: 30px;
    }
    .stApp footer p {
        color: #777;
    }
    /* Matplotlib text color - ADJUSTED FOR THEME */
    .stPlotlyChart {
        color: #4A4A4A; /* Deep Gray for chart elements */
    }
    /* Small text/caption styling - ADDED */
    small {
        color: #777777; /* Slightly lighter gray for captions */
        opacity: 0.95; /* Ensure full opacity for clarity */
    }

    /* Streamlit widgets general styling */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #FFF0F6; /* Blush Tint */
        border: 1px solid #F48FB1; /* Pink Rose border */
        color: #4A4A4A; /* Deep Gray text */
        border-radius: 5px;
        padding: 10px;
    }

    /* Horizontal Rule - ADJUSTED */
    hr {
        border-top: 2px solid rgba(244, 143, 177, 0.6); /* Slightly thicker, more visible pink divider */
        margin-top: 2.5em; /* More space around dividers */
        margin-bottom: 2.5em;
    }

    /* Image display specific adjustments */
    div[data-testid="stImage"] {
        margin-top: 20px;
        margin-bottom: 20px;
    }

    </style>
""", unsafe_allow_html=True)

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

# --- Fungsi untuk menampilkan visualisasi preprocessing ---
def display_preprocessing_steps(image_array):
    """Menampilkan setiap tahap preprocessing"""
    st.markdown("### Tahapan Image Preprocessing")
    
    # Dapatkan hasil preprocessing
    preprocessing_results = preprocess_with_visualization(image_array)
    
    # Penjelasan teknik
    st.markdown("""
    **Noise Reduction**: Mengurangi gangguan sinyal (noise) tanpa menghilangkan informasi penting.
    **Contrast Enhancement**: Meningkatkan kontras yang rendah untuk memudahkan observasi struktur tumor.
    **Normalization**: Menyeragamkan rentang intensitas antar citra agar model tidak bias.
    """)
    
    st.divider()

    # Tampilkan original image
    st.write("#### 📷 Gambar Original (Grayscale & Resized)")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(preprocessing_results['original'], caption='Gambar Original (128x128)', use_container_width=True, clamp=True)
    
    st.divider()

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
    
    st.divider()

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
    
    st.divider()

    # Final Normalized Image
    st.write("#### 📏 Normalized Image (Output Preprocessing)")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(preprocessing_results['normalized'], 
                  caption='Normalized Image (Min-Max Scaling)\nMenyeragamkan rentang intensitas 0-255', 
                  use_container_width=True, clamp=True)
    
    return preprocessing_results['normalized']

def display_transformation_steps(processed_image):
    """Menampilkan tahapan image transformation"""
    st.markdown("### Image Transformation & Feature Extraction")
    
    st.markdown("""
    **Edge Detection**: Batas tumor biasanya muncul dalam bentuk perubahan intensitas yang tajam.
    **Morphological Operations**: Membersihkan noise setelah segmentasi dan memperjelas bentuk objek.
    **Segmentation Techniques**: Memisahkan area tumor dari latar belakang, sehingga informasi spasial tumor bisa dianalisis.
    """)
    
    st.divider()

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
    
    st.divider()

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
    
    st.divider()

    # Segmentation Techniques
    st.write("#### 🎯 Segmentation Techniques (Untuk Ekstraksi Fitur Bentuk)")
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
    st.markdown("Analisis citra MRI untuk membantu mendeteksi keberadaan dan jenis tumor otak.")
    st.markdown("---")
    
    # Load model
    model_package, model_loaded, load_message = load_model_package()
    
    # Sidebar info
    with st.sidebar:
        st.header("📋 Informasi Model")
        st.markdown("Aplikasi ini menggunakan model Machine Learning untuk klasifikasi citra MRI.")
        st.divider()
        if model_loaded:
            st.success("✅ Model berhasil dimuat!")
            st.info(load_message)
            if 'best_classifier_name' in model_package:
                st.write(f"**Nama Model**: `{model_package['best_classifier_name']}`")
            if 'training_date' in model_package:
                st.write(f"**Tanggal Pelatihan**: `{model_package['training_date']}`")
        else:
            st.error("❌ Gagal memuat model.")
            st.error(load_message)
            st.warning("Pastikan file model (`brain_tumor_model.pkl` atau `best_model.pkl` dan `feature_scaler.pkl`) berada di folder 'models/'.")
        
        st.markdown("### 🎯 Kelas Klasifikasi")
        if model_loaded:
            for cls in model_package['class_names']:
                if cls == 'notumor' or cls == 'normal':
                    st.write(f"• **Normal** (Tanpa Tumor)")
                else:
                    st.write(f"• **{cls.title()}**")

        st.markdown("""
        <div style='text-align: center; margin-top: 150px;'>
            <p style='color: #888;'>Dibangun dengan ❤️ oleh PCD A4 IF ITK</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    st.header("📤 Upload Gambar MRI")
    st.markdown("Pilih dan unggah gambar MRI otak untuk memulai proses deteksi. Pastikan format gambar adalah JPG, JPEG, atau PNG.")
    uploaded_file = st.file_uploader(
        "Pilih gambar MRI ...", # Removed detailed instruction here as it's now above.
        type=["jpg", "jpeg", "png"],
        help="Gambar MRI akan diproses untuk ekstraksi fitur dan prediksi jenis tumor."
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
        
        st.subheader("🖼️ Gambar MRI yang Diunggah")
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image(image, caption='Gambar MRI Asli', use_container_width=True, width=300)
        st.markdown("---")

        tab_preprocess, tab_transform, tab_predict = st.tabs(["⚙️ Preprocessing", "✨ Transformasi Citra & Fitur", "📊 Hasil Prediksi"])

        with tab_preprocess:
            with st.container(border=True):
                processed_image = display_preprocessing_steps(opencv_image)
            
        with tab_transform:
            with st.container(border=True):
                segmented_image = display_transformation_steps(processed_image)
            
        with tab_predict:
            if model_loaded:
                with st.container(border=True):
                    st.markdown("### Menganalisis Gambar dan Melakukan Prediksi")
                    
                    with st.spinner("Memproses gambar, mengekstraksi fitur, dan memprediksi..."):
                        result = predict_brain_tumor(opencv_image, model_package)

                    if 'error' not in result:
                        col_pred1, col_pred2 = st.columns([1, 2])

                        with col_pred1:
                            predicted_class = result['predicted_class']
                            confidence = result['confidence']

                            st.markdown("#### Hasil Deteksi Utama:")
                            if predicted_class in ['notumor', 'normal']:
                                st.success(f"✅ **Tidak Ada Tumor Terdeteksi**")
                                st.metric(label="Tingkat Kepercayaan", value=f"{confidence:.2%}", delta="Sehat!")
                            else:
                                st.warning(f"⚠️ **{predicted_class.title()} Terdeteksi**")
                                st.metric(label="Tingkat Kepercayaan", value=f"{confidence:.2%}", delta="Perlu Pemeriksaan Lanjut!")

                            st.markdown("#### Citra Hasil Segmentasi:")
                            st.image(result['segmented_image'], caption='Citra Hasil Segmentasi (Otsu)', use_container_width=True, clamp=True)


                        with col_pred2:
                            st.subheader("📊 Probabilitas Klasifikasi Lengkap")
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

                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Use new theme colors for the bar chart
                            colors_for_bars = ['#4CAF50' if 'Normal' in cls else '#F06292' for cls in prob_df['Kelas']] # Green for normal, Warm Pink for others
                            bars = ax.barh(prob_df['Kelas'], prob_df['Probabilitas'], color=colors_for_bars)
                            
                            ax.set_xlabel('Probabilitas', color='#4A4A4A') # Deep Gray
                            ax.set_ylabel('Kelas Tumor', color='#4A4A4A') # Deep Gray
                            ax.set_xlim(0, 1)
                            ax.set_facecolor('#FCE4EC') # Light Pink - Chart background
                            fig.patch.set_facecolor('#FFF0F6') # Blush Tint - Figure background

                            # Set axis text color
                            ax.tick_params(axis='x', colors='#4A4A4A')
                            ax.tick_params(axis='y', colors='#4A4A4A')
                            ax.spines['bottom'].set_color('#4A4A4A')
                            ax.spines['left'].set_color('#4A4A4A')
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)

                            # Add subtle grid lines
                            ax.xaxis.grid(True, linestyle='--', alpha=0.6, color='#CCCCCC') # Lighter gray grid
                            ax.set_axisbelow(True) # Ensure grid is behind bars
                            
                            plt.title('Distribusi Probabilitas Kelas', color='#F06292', fontsize=16) # Warm Pink
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        st.divider()
                        # Analisis fitur
                        st.subheader("🔬 Analisis Fitur yang Diekstraksi")
                        st.info(f"**Total Fitur yang Diekstraksi**: {len(result['features'])}")

                        feat_col1, feat_col2, feat_col3 = st.columns(3)

                        with feat_col1:
                            st.markdown("### **🎨 Fitur Tekstur (10 fitur)**")
                            st.markdown("""
                            - Contrast
                            - Dissimilarity
                            - Homogeneity
                            - Energy
                            - Correlation
                            <br><small>*Berbasis Gray Level Co-occurrence Matrix (GLCM) pada jarak 1 dan 2, berbagai sudut.*</small>
                            """, unsafe_allow_html=True)

                        with feat_col2:
                            st.markdown("### **📐 Fitur Bentuk (10 fitur)**")
                            st.markdown("""
                            - Area
                            - Perimeter
                            - Compactness
                            - Aspect Ratio
                            - Extent
                            - Solidity
                            - Equivalent Diameter
                            - Centroid (X, Y)
                            - Eccentricity
                            <br><small>*Berbasis analisis kontur pada citra tersegmentasi.*</small>
                            """, unsafe_allow_html=True)

                        with feat_col3:
                            st.markdown("### **💡 Fitur Intensitas (7 fitur)**")
                            st.markdown("""
                            - Mean Intensity
                            - Standard Deviation
                            - Min/Max Intensity
                            - Entropy
                            - Skewness
                            - Kurtosis
                            <br><small>*Berbasis statistik dari distribusi intensitas piksel.*</small>
                            """, unsafe_allow_html=True)

                        # Tampilkan ringkasan nilai fitur
                        st.markdown("#### 📈 Nilai Fitur yang Diekstraksi (Top 15)")
                        feature_names = ['Contrast_d1', 'Dissimilarity_d1', 'Homogeneity_d1', 'Energy_d1', 'Correlation_d1',
                                         'Contrast_d2', 'Dissimilarity_d2', 'Homogeneity_d2', 'Energy_d2', 'Correlation_d2',
                                         'Area', 'Perimeter', 'Compactness', 'Aspect_Ratio', 'Extent',
                                         'Solidity', 'Equiv_Diameter', 'Centroid_X', 'Centroid_Y', 'Eccentricity',
                                         'Mean_Intensity', 'Std_Intensity', 'Min_Intensity', 'Max_Intensity',
                                         'Entropy', 'Skewness', 'Kurtosis']

                        features_df = pd.DataFrame({
                            'Fitur': feature_names[:len(result['features'])],
                            'Nilai': result['features']
                        })

                        st.dataframe(features_df.head(15), use_container_width=True)

                        if len(features_df) > 15:
                            with st.expander("📄 Klik untuk Melihat Semua Fitur Detail"):
                                st.dataframe(features_df, use_container_width=True)

                    else:
                        st.error(f"❌ Kesalahan saat prediksi: {result['error']}")
            else:
                st.error("Tidak dapat melakukan prediksi - model belum dimuat. Mohon cek kembali folder 'models/'.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: left; color: #888; padding-top: 20px;'>
            <p><strong>⚠️ Disclaimer:</strong> </p> 
            <p>Aplikasi ini merupakan bagian dari tugas besar mata kuliah Pengolahan Citra Digital dan dikembangkan untuk tujuan edukatif. <br>Model yang digunakan bukan untuk diagnosis medis dan hasil prediksi tidak dapat dijadikan acuan klinis.</p>
            <p>&copy; 2025 Deteksi Tumor Otak. Hak Cipta Dilindungi.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()