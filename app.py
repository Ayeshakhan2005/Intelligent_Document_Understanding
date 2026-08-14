import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Intelligent Document Understanding")
st.title("📄 Intelligent Document Understanding")
st.write("Upload one or more document images, process them with OCR, and ask questions about their contents.")

uploaded_files = st.file_uploader("Upload Document Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("Selected Documents")
    st.write(f"{len(uploaded_files)} document(s) selected.")
    
    all_text = ""
    
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
        
        # Convert to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # 1. UPSCALE 3x for blurry
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # 2. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
        
        # 4. Adaptive Threshold - BEST for uneven lighting/blur
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        
        # 5. OCR with LSTM
        custom_config = r'--oem 1 --psm 6'
        text = pytesseract.image_to_string(gray, config=custom_config)
        
        all_text += f"\n\n--- From {uploaded_file.name} ---\n\n" + text
    
    st.subheader("Extracted Text")
    st.text_area("Result", all_text, height=300)
    
    # Download button
    st.download_button("Download Text", all_text, "extracted_text.txt")
else:
    st.info("Please upload images to begin.")
