import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np

st.title("Intelligent Document Understanding")
uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)
    
    # CLEAN IMAGE FOR OCR
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    if st.button("Extract Text"):
        text = pytesseract.image_to_string(img, config='--psm 6')
        st.session_state.extracted_text = text
        st.success("Text Extracted")
    
    if st.session_state.extracted_text:
        st.text_area("Extracted Text", st.session_state.extracted_text, height=200)
        
        st.subheader("Ask Questions")
        question = st.text_input("Ask a question about the document:")
        
        if st.button("Get Answer") and question:
            # SIMPLE Q&A: finds sentence with keywords
            text = st.session_state.extracted_text.lower()
            keywords = question.lower().split()
            sentences = st.session_state.extracted_text.split('.')
            
            answer = "Answer not found in document."
            for s in sentences:
                if any(k in s.lower() for k in keywords):
                    answer = s.strip()
                    break
            
            st.write("**Answer:**", answer)
