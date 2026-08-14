import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import re

st.set_page_config(page_title="Intelligent Document Understanding", layout="wide")
st.title("📄 Intelligent Document Understanding PRO")

@st.cache_resource # load model only once
def load_reader():
    return easyocr.Reader(['en'], gpu=False) # 'en' for english

reader = load_reader()

uploaded_file = st.file_uploader("Upload ONE Document Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(bytes_data))
    st.image(image, caption=uploaded_file.name, use_container_width=True)
    
    # Preprocessing
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # EASYOCR - way better than tesseract
    with st.spinner("Reading text with AI... this takes 10s"):
        results = reader.readtext(gray, detail=0, paragraph=True)
    
    text = "\n".join(results)
    text = text.replace('-\n', '').replace(' 1s ', ' is ').replace(' ic ', ' is ')
    
    st.subheader("Extracted Text")
    st.text_area("Result", text, height=300)
    st.download_button("Download Text", text, "extracted_text.txt")

    # Q&A
    st.subheader("💬 Ask Questions About This Document")
    question = st.text_input("Ask anything:")
    if question and text:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if len(s.strip()) > 10]
        q_words = set(question.lower().split())
        scored = [(len(q_words.intersection(set(s.lower().split()))), s) for s in sentences]
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored and scored[0][0] > 0:
            st.success(scored[0][1])
        else:
            st.warning("Answer not found.")
else:
    st.info("Please upload an image to begin.")
