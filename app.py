import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import re

st.set_page_config(page_title="Intelligent Document Understanding", layout="wide")
st.title("📄 Intelligent Document Understanding - Column Fix")

def get_text_from_image(img):
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # KEY: psm 4 = Assume a single column of text of variable sizes
    # psm 3 = Fully automatic. We run both and take the better one
    config = r'--oem 1 --psm 4'
    text = pytesseract.image_to_string(gray, config=config)
    return text

uploaded_file = st.file_uploader("Upload ONE Document Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(bytes_data))
    st.image(image, caption=uploaded_file.name, use_container_width=True)
    
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]
    
    # SPLIT INTO 2 COLUMNS
    mid = w // 2
    left_col = img[:, :mid]
    right_col = img[:, mid:]
    
    with st.spinner("Reading column 1..."):
        text_left = get_text_from_image(left_col)
    with st.spinner("Reading column 2..."):
        text_right = get_text_from_image(right_col)
    
    all_text = text_left + "\n\n" + text_right
    all_text = all_text.replace('-\n', '')
    all_text = re.sub(r'\s+', ' ', all_text) # remove extra spaces
    
    st.subheader("Extracted Text")
    st.text_area("Result", all_text, height=400)
    st.download_button("Download Text", all_text, "extracted_text.txt")

    st.subheader("💬 Ask Questions")
    question = st.text_input("Ask anything:")
    if question and all_text:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', all_text) if len(s.strip()) > 10]
        q_words = set(question.lower().split())
        scored = [(len(q_words.intersection(set(s.lower().split()))), s) for s in sentences]
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored and scored[0][0] > 0:
            st.success(scored[0][1])
else:
    st.info("Please upload an image to begin.")
