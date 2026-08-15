import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re

st.title("Intelligent Document Understanding")
uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original")
    
    # THIS PREPROCESSING WORKED FOR YOU BEFORE
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    st.image(img, caption="Processed for OCR")
    
    if st.button("Extract Text"):
        text = pytesseract.image_to_string(img, config='--psm 6')
        st.session_state.extracted_text = text
        st.success("Text Extracted")
    
    if st.session_state.extracted_text:
        st.text_area("Extracted Text", st.session_state.extracted_text, height=200)
        
        st.subheader("Ask Questions")
        question = st.text_input("Ask a question about the document:")
        
        if st.button("Get Answer") and question:
            text = st.session_state.extracted_text
            # Split into proper sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if len(s.strip()) > 10]
            
            # Find best matching sentence
            q_words = set([w.lower() for w in question.split() if len(w) > 2])
            best_answer = "Answer not found in document."
            best_score = 0
            
            for s in sentences:
                score = sum(1 for w in q_words if w in s.lower())
                if score > best_score:
                    best_score = score
                    best_answer = s
            
            if best_score > 0:
                st.write("**Answer:**", best_answer)
            else:
                st.write("**Answer:**", "Answer not found in document.")
