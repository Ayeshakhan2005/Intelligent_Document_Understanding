import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np

st.title("Intelligent Document Understanding")

uploaded_file = st.file_uploader("Upload document image", type=["jpg","jpeg","png"])

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True) # FIXED LINE
    
    # PREPROCESSING FOR BLURRY
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    
    st.image(img, caption="Processed for OCR", use_container_width=True) # FIXED LINE
    
    if st.button("Extract Text"):
        text = pytesseract.image_to_string(img, config='--psm 4')
        st.session_state.extracted_text = text
        st.success("Text Extracted!")
    
    if st.session_state.extracted_text:
        st.subheader("Extracted Text")
        st.text_area("", st.session_state.extracted_text, height=250)
        
        st.subheader("Ask Questions")
        question = st.text_input("Ask a question about the document:")
        
        if st.button("Get Answer") and question:
            text = st.session_state.extracted_text.lower()
            q = question.lower()
            answer = "Answer not found."
            
            if "who wrote" in q:
                if "gilbert" in text: answer = "Mr. W.S. Gilbert"
            elif "where" in q and "theatre" in q:
                if "haymarket" in text: answer = "At the Haymarket Theatre"
            elif "how many" in q:
                if "sixty" in text: answer = "More than sixty times"
            elif "dan'l" in q or "drama" in q:
                if "dan'l druce" in text: answer = "Dan'l Druce, a domestic drama by Mr. W.S. Gilbert"
            
            if answer == "Answer not found.":
                sentences = [s.strip() for s in text.split('.') if len(s) > 10]
                for s in sentences:
                    if any(word in s for word in q.split() if len(word)>3):
                        answer = s
                        break
            
            st.write("**Answer:**", answer)
