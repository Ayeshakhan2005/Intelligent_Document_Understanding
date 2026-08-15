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


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


def get_text_from_image(img):

    img = cv2.resize(
        img,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    text = pytesseract.image_to_string(
        processed,
        config="--oem 1 --psm 3"
    )

    return text


def clean_text(text):

    text = text.replace("-\n", "")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def find_relevant_context(question, text):

    words = text.split()

    chunks = []

    for i in range(0, len(words), 100):

        chunk = " ".join(words[i:i + 100])

        if len(chunk) > 20:
            chunks.append(chunk)

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

    best = np.argsort(scores)[::-1][:3]

    results = [
        chunks[i]
        for i in best
        if scores[i] >= 0.25
    ]

    if not results:
        return None

    return "\n\n".join(results[:2])


uploaded_file = st.file_uploader(
    "Upload a document image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)


if uploaded_file:

    data = uploaded_file.getvalue()

    image = Image.open(
        io.BytesIO(data)
    )

    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True
    )

    file_bytes = np.asarray(
        bytearray(data),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    with st.spinner("Extracting text..."):

        text = get_text_from_image(img)
        text = clean_text(text)

    st.subheader("📝 Extracted Text")

    st.text_area(
        "Result",
        text,
        height=400
    )

    st.download_button(
        "Download Text",
        text,
        "extracted_text.txt"
    )

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Ask anything about the document:"
    )

    if question:

        context = find_relevant_context(
            question,
            text
        )

        if context:
            st.success("Relevant information:")
            st.write(context)
        else:
            st.warning(
                "I could not find relevant information."
            )

else:

    st.info("Please upload an image to begin.")
