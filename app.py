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

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original")
    
    # STRONG PREPROCESSING FOR BLURRY
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 15)
    
    if st.button("Extract Text"):
        text = pytesseract.image_to_string(img, config='--psm 6 --oem 3')
        st.session_state.extracted_text = clean_text(text)
        st.success("Text Extracted")
    
    if st.session_state.extracted_text:
        st.text_area("Extracted Text", st.session_state.extracted_text, height=200)
        
        st.subheader("Ask Questions")
        question = st.text_input("Ask a question about the document:")
        
        if st.button("Get Answer") and question:
            text = st.session_state.extracted_text
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]
            
            # SCORE each sentence by how many question words it has
            q_words = set([w.lower() for w in question.split() if len(w) > 3])
            best_score = 0
            best_answer = "Answer not found in document."
            
            for s in sentences:
                s_lower = s.lower()
                score = sum(1 for w in q_words if w in s_lower)
                if score > best_score:
                    best_score = score
                    best_answer = s
            
            st.write("**Answer:**", best_answer)
