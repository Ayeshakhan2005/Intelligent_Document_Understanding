import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import re

st.set_page_config(page_title="Intelligent Document Understanding", layout="wide")
st.title("📄 Intelligent Document Understanding")
st.write("Upload document images, extract text with OCR, and ask questions about them.")

uploaded_files = st.file_uploader("Upload Document Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("Selected Documents")
    st.write(f"{len(uploaded_files)} document(s) selected.")
    
    all_text = ""
    
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.getvalue()
        
        # Show image
        image = Image.open(io.BytesIO(bytes_data))
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        
        # OCR with heavy preprocessing for blurry images
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # 1. UPSCALE 3x 
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # 2. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
        
        # 4. Adaptive Threshold - BEST for blurry + uneven light
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        
        # 5. OCR with LSTM Engine
        custom_config = r'--oem 1 --psm 6'
        text = pytesseract.image_to_string(gray, config=custom_config)
        
        # FIX 1: Join hyphenated words from newspaper columns
        text = text.replace('-\n', '')
        text = text.replace('-\r\n', '')
        
        all_text += f"\n\n--- From {uploaded_file.name} ---\n\n" + text
    
    st.subheader("Extracted Text")
    st.text_area("Result", all_text, height=300)
    st.download_button("Download Text", all_text, "extracted_text.txt")

    # ===== SMARTER Q&A PART =====
    st.subheader("💬 Ask Questions About Your Document")
    question = st.text_input("Ask anything about the uploaded document:")

    if question and all_text:
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', all_text) if len(s.strip()) > 10]
        
        # Score sentences by how many question words they contain
        q_words = set(question.lower().split())
        scored = []
        for s in sentences:
            score = len(q_words.intersection(set(s.lower().split())))
            if score > 0:
                scored.append((score, s))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        
        if scored:
            st.write("**Answer based on document:**")
            st.success(scored[0][1]) # FIX 2: Green box for answer
            
            if len(scored) > 1:
                with st.expander("See more context"):
                    for _, s in scored[1:3]:
                        st.write("- " + s)
        else:
            st.warning("I couldn't find an answer in the document for that question. Try different keywords.")
else:
    st.info("Please upload images to begin.")
