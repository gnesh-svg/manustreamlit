import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_sauvola, threshold_niblack

# Page Setup
st.set_page_config(page_title="Manuscript Master Pro", layout="wide")

def calculate_metrics(original_gray, processed_final):
    mse = np.mean((original_gray.astype(np.float32) - processed_final.astype(np.float32)) ** 2)
    psnr = 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    score, _ = ssim(original_gray, processed_final, full=True)
    return mse, psnr, score

# --- Sidebar ---
st.sidebar.title("🎮 Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "tif", "bmp"])

# View Toggle
st.sidebar.markdown("---")
focus_mode = st.sidebar.checkbox("🔍 Focus Mode (Enlarge Output)", value=False)

# NEW: Auto-Enhance and Manual Adjustments
st.sidebar.subheader("Adjustments")
auto_mode = st.sidebar.button("✨ Auto-Enhance (Histogram Equalize)")
brightness = st.sidebar.slider("Brightness", -100, 100, 0)
contrast = st.sidebar.slider("Contrast", -100, 100, 0)

# 1. Noise Filter
st.sidebar.subheader("1. Noise Filter")
filter_type = st.sidebar.selectbox("Strategy", ("Gaussian Blur", "Non-Local Means", "Median Filter", "Bilateral"))
filter_strength = st.sidebar.slider("Strength", 1, 25, 5)
use_clahe = st.sidebar.checkbox("Enable CLAHE Enhancement", value=True)

# 2. Binarization
st.sidebar.subheader("2. Binarization")
thresh_type = st.sidebar.selectbox("Strategy", ("Hybrid (Sauvola+Otsu)", "Otsu (Global)", "Sauvola (Local)", "Niblack (Local)", "Adaptive Gaussian"))
window_size = st.sidebar.slider("Window Size", 3, 101, 25, step=2)

# 3. Canny Sensitivity
st.sidebar.subheader("3. Edge Sensitivity")
edge_val = st.sidebar.slider("Canny Threshold", 10, 250, 100)

st.title("📜 Manuscript Master App")

if uploaded_file is not None:
    # Load Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    # Pre-processing Logic
    if auto_mode:
        # Convert to YUV to equalize brightness (Y channel) without messing up colors
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        adjusted = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        st.sidebar.success("Auto-Enhance Applied!")
    else:
        # Apply Manual Brightness/Contrast
        alpha = (contrast + 127) / 127
        beta = brightness
        adjusted = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    # Stage 1: Filtering
    k = filter_strength if filter_strength % 2 != 0 else filter_strength + 1
    if filter_type == "Gaussian Blur":
        filtered = cv2.GaussianBlur(gray, (k, k), 0)
    elif filter_type == "Non-Local Means":
        filtered = cv2.fastNlMeansDenoising(gray, h=filter_strength)
    elif filter_type == "Median Filter":
        filtered = cv2.medianBlur(gray, k)
    else:
        filtered = cv2.bilateralFilter(gray, k, 75, 75)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        filtered = clahe.apply(filtered)

    # Stage 2: Binarization
    w = window_size if window_size % 2 != 0 else window_size + 1
    if thresh_type == "Otsu (Global)":
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresh_type == "Sauvola (Local)":
        binary = (filtered > threshold_sauvola(filtered, window_size=w)).astype(np.uint8) * 255
    elif thresh_type == "Niblack (Local)":
        binary = (filtered > threshold_niblack(filtered, window_size=w, k=0.2)).astype(np.uint8) * 255
    elif thresh_type == "Adaptive Gaussian":
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, w, 2)
    else:
        otsu_val, _ = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        local_t = threshold_sauvola(filtered, window_size=w)
        binary = np.where((filtered < local_t) & (filtered < otsu_val), 0, 255).astype(np.uint8)

    # Stage 3: Edge Validation
    edges = cv2.Canny(filtered, edge_val/2, edge_val)
    mask = cv2.dilate(edges, np.ones((2,2), np.uint8))
    final = np.where((binary == 0) & (mask == 0), 255, binary).astype(np.uint8)

    # --- Display ---
    if focus_mode:
        st.subheader("🎯 Focus View: Final Output")
        st.image(final, use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🖼️ Adjusted Original")
            st.image(adjusted, channels="BGR", use_container_width=True)
        with col2:
            st.markdown("### 🎞️ Filtered")
            st.image(filtered, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Final Result")
        st.image(final, use_container_width=True)

    # Metrics
    mse, psnr, ssim_val = calculate_metrics(gray, final)
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("MSE", f"{mse:,.1f}")
    m_col2.metric("PSNR", f"{psnr:,.1f} dB")
    m_col3.metric("SSIM", f"{ssim_val:,.3f}")

    # Export
    res, img_encoded = cv2.imencode(".png", final)
    st.sidebar.markdown("---")
    st.sidebar.download_button("💾 Download Final Image", data=img_encoded.tobytes(), file_name="processed.png", mime="image/png")

else:
    st.info("Upload a file in the sidebar to get started!")
