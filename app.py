import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import re

st.set_page_config(page_title="Intelligent Document Understanding", layout="wide")
st.title("📄 Intelligent Document Understanding")

uploaded_file = st.file_uploader("Upload ONE Document Image", type=["jpg", "jpeg", "png"]) # CHANGED: 1 file at a time

if uploaded_file:
    bytes_data = uploaded_file.getvalue()
    
    # Show image
    image = Image.open(io.BytesIO(bytes_data))
    st.image(image, caption=uploaded_file.name, use_container_width=True)
    
    # OCR with AGGRESSIVE preprocessing for newspaper
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 1. UPSCALE 4x instead of 3x
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Denoise HARD
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 4. Threshold + Dilation to make text thicker
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((1,1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    
    # 5. OCR with PSM 3 for full page
    custom_config = r'--oem 1 --psm 3'
    text = pytesseract.image_to_string(gray, config=custom_config)
    
    # FIX hyphens and junk
    text = text.replace('-\n', '').replace('-\r\n', '')
    text = text.replace(' 1s ', ' is ').replace(' ic ', ' is ') # fix common OCR errors
    
    st.subheader("Extracted Text")
    st.text_area("Result", text, height=300)
    st.download_button("Download Text", text, "extracted_text.txt")

    # ===== Q&A PART =====
    st.subheader("💬 Ask Questions About This Document")
    question = st.text_input("Ask anything:")

    if question and text:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if len(s.strip()) > 10]
        q_words = set(question.lower().split())
        scored = [(len(q_words.intersection(set(s.lower().split()))), s) for s in sentences]
        scored.sort(reverse=True, key=lambda x: x[0])
        
        if scored and scored[0][0] > 0:
            st.write(f"**Answer based on {uploaded_file.name}:**")
            st.success(scored[0][1])
        else:
            st.warning("Answer not found in this document. Try different keywords.")
else:
    st.info("Please upload an image to begin.")
