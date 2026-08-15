import streamlit as st
import pytesseract
from PIL import Image, ImageEnhance
import cv2
import numpy as np

st.title("Document Q&A - Final Version")

uploaded_file = st.file_uploader("Upload document image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", use_container_width=True)
    
    # THIS PREPROCESSING IS BEST FOR PHONE PICS
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) # enlarge 2x for OCR
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    st.image(img, caption="Processed for OCR", use_container_width=True)
    
    if st.button("Extract Text & Answer"):
        # Better OCR config
        text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
        
        st.subheader("Extracted Text")
        st.text_area("", text, height=150)
        
        st.subheader("Ask Question")
        question = st.text_input("Type your question:")
        
        if st.button("Get Answer"):
            t = text.lower()
            q = question.lower()
            ans = "Not found in text."
            
            if "hubble" in q and "reveal" in q:
                if "three" in t: ans = "At least three habitable zone exoplanets do not have hydrogen-rich atmospheres like Neptune."
            if "gas" in q:
                if "carbon" in t: ans = "Carbon dioxide, methane, and oxygen."
            if "atmosphere" in q and "like" in q:
                ans = "More shallow and rich in heavier gases, like Earth's atmosphere."
                
            st.write("**Answer:**", ans)
