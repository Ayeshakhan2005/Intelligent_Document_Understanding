import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np

st.title("Intelligent Document Understanding")
uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)
    
    # BETTER PREPROCESSING FOR BAD PHOTOS
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.dilate(img, np.ones((1,1), np.uint8), iterations=1)
    
    if st.button("Extract Text"):
        text = pytesseract.image_to_string(img, config='--psm 6')
        st.text_area("Extracted Text", text, height=400)
