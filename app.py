import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_sauvola, threshold_niblack

st.set_page_config(page_title="Manuscript Master Pro", layout="wide")

def calculate_metrics(original_gray, processed_final):
    mse = np.mean((original_gray.astype(np.float32) - processed_final.astype(np.float32)) ** 2)
    psnr = 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    score, _ = ssim(original_gray, processed_final, full=True)
    return mse, psnr, score

# --- Sidebar Controls ---
st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Manuscript Image", type=["jpg", "jpeg", "png", "tif", "bmp"])

# Focus Mode Toggle
st.sidebar.markdown("---")
focus_mode = st.sidebar.checkbox("🔍 Focus Mode (Enlarge Output)", value=False)

st.sidebar.header("🛠️ Parameters")
filter_type = st.sidebar.selectbox("Filter Strategy", ("Gaussian Blur", "Non-Local Means", "Median Filter", "Bilateral"))
filter_strength = st.sidebar.slider("Filter Strength", 1, 25, 5)
use_clahe = st.sidebar.checkbox("Enable CLAHE", value=True)
thresh_type = st.sidebar.selectbox("Binarization", ("Hybrid (Sauvola+Otsu)", "Otsu (Global)", "Sauvola (Local)", "Niblack (Local)", "Adaptive Gaussian"))
window_size = st.sidebar.slider("Window Size", 3, 101, 25, step=2)
edge_val = st.sidebar.slider("Canny Sensitivity", 10, 250, 100)

st.title("📜 Manuscript Master App")

if uploaded_file is not None:
    # --- Image Processing Logic ---
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

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

    edges = cv2.Canny(filtered, edge_val/2, edge_val)
    mask = cv2.dilate(edges, np.ones((2,2), np.uint8))
    final = np.where((binary == 0) & (mask == 0), 255, binary).astype(np.uint8)

    # --- ENLARGE LOGIC ---
    if focus_mode:
        st.warning("⚠️ Focus Mode Active: Stage 1 and 2 are hidden.")
        st.subheader("🎯 FINAL BINARY (ENLARGED)")
        # Display with massive width
        st.image(final, use_container_width=True)
    else:
        # Standard Multi-Row View
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🖼️ Stage 1: Original")
            st.image(img_bgr, channels="BGR", use_container_width=True)
        with col2:
            st.markdown("### 🎞️ Stage 2: Filtered")
            st.image(filtered, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🎯 Stage 3: Final Binary")
        st.image(final, use_container_width=True)

    # Metrics Panel
    mse, psnr, ssim_val = calculate_metrics(gray, final)
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("MSE", f"{mse:,.1f}")
    m_col2.metric("PSNR", f"{psnr:,.1f} dB")
    m_col3.metric("SSIM", f"{ssim_val:,.3f}")

    # Export
    res, img_encoded = cv2.imencode(".png", final)
    st.sidebar.download_button("💾 Download Result", data=img_encoded.tobytes(), file_name="output.png", mime="image/png")

else:
    st.info("Please upload an image.")
