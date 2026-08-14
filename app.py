import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import re

st.set_page_config(page_title="Intelligent Document Understanding", layout="wide")
st.title("📄 Intelligent Document Understanding PRO")
st.write("Upload document images. Uses 2-pass OCR for better accuracy on old text.")

uploaded_file = st.file_uploader("Upload ONE Document Image", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3) # better than fastNlMeans for text
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray

if uploaded_file:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(bytes_data))
    st.image(image, caption=uploaded_file.name, use_container_width=True)
    
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    processed = preprocess_image(img)
    
    with st.spinner("Reading text..."):
        # PASS 1: psm 3 = Fully automatic page segmentation
        config1 = r'--oem 1 --psm 3'
        text1 = pytesseract.image_to_string(processed, config=config1)
        
        # PASS 2: psm 6 = Assume uniform block of text. Catch what pass 1 missed
        config2 = r'--oem 1 --psm 6'
        text2 = pytesseract.image_to_string(processed, config=config2)
    
    # Combine and clean
    all_text = text1 + "\n" + text2
    all_text = all_text.replace('-\n', '').replace(' 1s ', ' is ').replace(' ic ', ' is ')
    all_text = re.sub(r'[^\x00-\x7F]+', ' ', all_text) # remove weird symbols
    
    st.subheader("Extracted Text")
    st.text_area("Result", all_text, height=350)
    st.download_button("Download Text", all_text, "extracted_text.txt")

    # Q&A
    st.subheader("💬 Ask Questions About This Document")
    question = st.text_input("Ask anything:")
    if question and all_text:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', all_text) if len(s.strip()) > 15]
        q_words = set(question.lower().split())
        scored = [(len(q_words.intersection(set(s.lower().split()))), s) for s in sentences]
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored and scored[0][0] > 0:
            st.success(scored[0][1])
            with st.expander("See more context"):
                for _, s in scored[1:3]:
                    st.write("- " + s)
        else:
            st.warning("Answer not found in document.")
else:
    st.info("Please upload an image to begin.")
