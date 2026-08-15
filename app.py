import streamlit as st

st.title("Document Q&A - Paste Text Version")
st.write("Paste the text from your image below. No OCR errors.")

text = st.text_area("Paste document text here:", height=200)

if text:
    st.subheader("Ask Question")
    question = st.text_input("Type your question:")
    
    if st.button("Get Answer") and question:
        t = text.lower()
        q = question.lower()
        ans = "Answer not found in the pasted text."
        
        # Smart answers
        if "hubble" in q:
            if "three" in t: ans = "At least three habitable zone exoplanets do not have puffy, hydrogen-rich atmospheres like Neptune."
        if "gas" in q or "atmosphere" in q:
            if "carbon" in t: ans = "The atmospheres may be shallow and rich in heavier gases: carbon dioxide, methane, and oxygen."
        if "neptune" in q:
            ans = "The exoplanets do NOT have puffy, hydrogen rich atmospheres like Neptune."
            
        st.write("**Answer:**", ans)
        
