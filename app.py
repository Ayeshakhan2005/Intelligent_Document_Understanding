import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np

st.set_page_config(page_title="Newspaper OCR")
st.title("📰 Intelligent Document Understanding")

uploaded_file = st.file_uploader("Upload Newspaper Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # IMAGE PREPROCESSING - FIXES BLURRY/CLEAN IMAGES
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.medianBlur(img, 3)
    
    if st.button("Extract Text"):
        with st.spinner("Reading..."):
            text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
        st.success("Done!")
        st.text_area("Extracted Text", text, height=400)
