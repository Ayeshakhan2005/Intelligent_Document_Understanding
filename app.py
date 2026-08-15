import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import re
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Intelligent Document Understanding",
    layout="wide"
)

st.title("📄 Intelligent Document Understanding")

# Load semantic model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


def get_text_from_image(img):
    # Upscale
    img = cv2.resize(
        img,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mild denoising
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    # Automatic page layout
    text = pytesseract.image_to_string(
        processed,
        config="--oem 1 --psm 3"
    )

    return text


def clean_text(text):
    text = text.replace("-\n", "")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def make_chunks(text, size=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i + size])

        if len(chunk.strip()) > 20:
            chunks.append(chunk)

    return chunks


def find_relevant_context(question, text):
    chunks = make_chunks(text)

    if not chunks:
        return None

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    chunk_embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    scores = np.dot(
        chunk_embeddings,
        question_embedding
    )

    best_indices = np.argsort(scores)[::-1][:3]

    relevant = []

    for index in best_indices:
        if scores[index] >= 0.25:
            relevant.append(chunks[index])

    if not relevant:
        return None

    return "\n\n".join(relevant)


uploaded_file = st.file_uploader(
    "Upload ONE Document Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    bytes_data = uploaded_file.getvalue()

    image = Image.open(
        io.BytesIO(bytes_data)
    )

    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True
    )

    file_bytes = np.asarray(
        bytearray(bytes_data),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    with st.spinner("Extracting text..."):
        all_text = get_text_from_image(img)
        all_text = clean_text(all_text)

    st.subheader("📝 Extracted Text")

    st.text_area(
        "Result",
        all_text,
        height=400
    )

    st.download_button(
        "Download Text",
        all_text,
        "extracted_text.txt"
    )

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Ask anything about the document:"
    )

    if question:

        with st.spinner("Finding the most relevant information..."):

            context = find_relevant_context(
                question,
                all_text
            )

        if context:
            st.success("Most relevant information:")
            st.write(context)
        else:
            st.warning(
                "I could not find relevant information in the document."
            )

else:
    st.info("Please upload an image to begin.")
