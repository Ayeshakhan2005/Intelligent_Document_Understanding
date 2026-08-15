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

def find_answer(question, context):
    q = question.lower()
    context = context.replace('\n', ' ')
    
    # Rule 1: WHO questions
    if "who" in q:
        match = re.search(r'by (Mr\.? [A-Z][a-z\. ]+)', context)
        if match: return match.group(1)
        match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+) acted', context)
        if match: return match.group(1)
    
    # Rule 2: WHERE questions  
    if "where" in q:
        match = re.search(r'at the ([A-Z][a-z ]+ Theatre)', context)
        if match: return "At the " + match.group(1)
    
    # Rule 3: WHAT questions
    if "what" in q:
        if "about" in q or "is" in q:
            match = re.search(r'This (interesting [^\.]+)', context)
            if match: return match.group(1)
    
    # Rule 4: HOW MANY
    if "how many" in q:
        match = re.search(r'(\d+) times', context)
        if match: return match.group(1) + " times"
        
    # Fallback: best sentence
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', context) if len(s) > 15]
    q_words = set([w for w in q.split() if len(w) > 3])
    best = max(sentences, key=lambda s: sum(1 for w in q_words if w in s.lower()), default="")
    return best if best else "Answer not found"


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original")
    
    # WORKING OCR
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
        question = st.text_input("Ask a question:")
        
        if st.button("Get Answer") and question:
            answer = find_answer(question, st.session_state.extracted_text)
            st.write("**Answer:**", answer)
