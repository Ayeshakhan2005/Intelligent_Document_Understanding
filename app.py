import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re

st.set_page_config(layout="wide")
st.title("Intelligent Document Q&A")


# ============================================================
# OCR FUNCTIONS
# ============================================================

def prepare_images(image):
    """
    Create several versions of the uploaded image.
    Different images work better with different preprocessing.
    """

    rgb = np.array(image.convert("RGB"))

    # OpenCV grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Upscale
    gray_up = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Slight denoising
    denoised = cv2.fastNlMeansDenoising(
        gray_up,
        None,
        10,
        7,
        21
    )

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    # Otsu threshold
    _, otsu = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Sharpened grayscale
    blur = cv2.GaussianBlur(
        gray_up,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(
        gray_up,
        1.5,
        blur,
        -0.5,
        0
    )

    return {
        "original": gray_up,
        "adaptive": adaptive,
        "otsu": otsu,
        "sharpened": sharpened
    }


def score_ocr_text(text):
    """
    Estimate which OCR result is most useful.
    """

    if not text:
        return 0

    # Words from Latin and Urdu/Arabic ranges
    words = re.findall(
        r"[A-Za-zÀ-ÿ\u0600-\u06FF]{2,}",
        text
    )

    if not words:
        return 0

    score = 0

    # Reward actual words
    score += len(words) * 2

    # Penalize excessive garbage symbols
    symbols = re.findall(
        r"[~`^_=*|\\<>]{2,}",
        text
    )

    score -= len(symbols) * 3

    # Penalize extremely short garbage lines
    lines = text.splitlines()

    for line in lines:
        line = line.strip()

        if line and len(line) <= 2:
            score -= 1

    return score


def extract_text_from_image(image):
    """
    Run Tesseract several ways and choose the strongest result.
    """

    versions = prepare_images(image)

    results = []

    # Different page segmentation modes
    configs = [
        "--oem 3 --psm 3",
        "--oem 3 --psm 4",
        "--oem 3 --psm 6",
        "--oem 3 --psm 11"
    ]

    for name, processed_image in versions.items():

        for config in configs:

            try:

                text = pytesseract.image_to_string(
                    processed_image,
                    config=config
                )

                if text and text.strip():

                    score = score_ocr_text(text)

                    results.append(
                        (
                            score,
                            text
                        )
                    )

            except Exception:
                continue

    if not results:
        return ""

    # Highest scoring OCR result
    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return results[0][1]


def clean_text(text):
    """
    Clean OCR formatting without changing the actual content.
    """

    if not text:
        return ""

    # Join words broken at line endings
    text = re.sub(
        r"-\s*\n\s*",
        "",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# QUESTION ANSWERING
# ============================================================

def normalize_words(text):
    """
    Extract English, Urdu/Arabic and numeric words.
    """

    return set(
        re.findall(
            r"[A-Za-zÀ-ÿ\u0600-\u06FF0-9]+",
            text.lower()
        )
    )


def split_into_passages(text):
    """
    Break extracted document into useful passages.
    """

    # First split by paragraphs
    paragraphs = re.split(
        r"\n{2,}",
        text
    )

    passages = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Further split long paragraphs into sentences
        sentences = re.split(
            r"(?<=[.!?])\s+",
            paragraph
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) >= 10:
                passages.append(sentence)

    return passages


def find_best_passages(question, text):
    """
    Find document passages most relevant to the question.
    """

    passages = split_into_passages(text)

    if not passages:
        return []

    question_words = normalize_words(question)

    # Words that usually don't help identify the answer
    stop_words = {
        "what",
        "where",
        "when",
        "who",
        "why",
        "how",
        "which",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "and",
        "or",
        "does",
        "did",
        "do",
        "can",
        "could",
        "would",
        "should"
    }

    important_words = (
        question_words - stop_words
    )

    scored_passages = []

    for passage in passages:

        passage_words = normalize_words(
            passage
        )

        normal_overlap = len(
            question_words & passage_words
        )

        important_overlap = len(
            important_words & passage_words
        )

        # Give important words more weight
        score = (
            normal_overlap
            + important_overlap * 3
        )

        # Small bonus when important words
        # occur in the passage
        if important_words:

            coverage = (
                important_overlap
                / len(important_words)
            )

            score += coverage * 5

        scored_passages.append(
            (
                score,
                passage
            )
        )

    scored_passages.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        passage
        for score, passage in scored_passages[:5]
        if score > 0
    ]


def answer_question(question, text):
    """
    Generate a useful answer from the extracted document.
    """

    if not text.strip():
        return "No readable text was extracted from the image."

    best_passages = find_best_passages(
        question,
        text
    )

    if not best_passages:
        return (
            "I could not find information related "
            "to this question in the extracted document."
        )

    # For questions asking for a direct fact,
    # return the most relevant passage first.
    question_lower = question.lower()

    direct_question_words = [
        "who",
        "where",
        "when",
        "how many",
        "how much",
        "what",
        "which"
    ]

    is_direct_question = any(
        word in question_lower
        for word in direct_question_words
    )

    if is_direct_question:

        return best_passages[0]

    # For general questions, provide the
    # strongest relevant passages.
    return "\n\n".join(
        best_passages[:3]
    )


# ============================================================
# STREAMLIT APP
# ============================================================

uploaded_file = st.file_uploader(
    "Upload document image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    # Show the main preprocessing preview
    processed_versions = prepare_images(
        image
    )

    with col2:

        st.image(
            processed_versions["adaptive"],
            caption="Processed for OCR",
            use_container_width=True
        )

    # ========================================================
    # EXTRACT TEXT
    # ========================================================

    if st.button(
        "Extract Text",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing image and extracting text..."
        ):

            extracted_text = (
                extract_text_from_image(image)
            )

            extracted_text = clean_text(
                extracted_text
            )

            st.session_state.extracted_text = (
                extracted_text
            )

        if extracted_text:

            st.success(
                "Text extracted successfully!"
            )

        else:

            st.error(
                "No readable text was detected."
            )

    # ========================================================
    # DISPLAY EXTRACTED TEXT
    # ========================================================

    if "extracted_text" in st.session_state:

        extracted_text = (
            st.session_state.extracted_text
        )

        st.subheader(
            "📝 Extracted Text"
        )

        st.text_area(
            "Document text",
            extracted_text,
            height=350
        )

        st.download_button(
            "⬇️ Download Extracted Text",
            extracted_text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )

        # ====================================================
        # QUESTION ANSWERING
        # ====================================================

        st.subheader(
            "💬 Ask Questions"
        )

        question = st.text_input(
            "Ask a question about the document:"
        )

        if st.button(
            "Get Answer",
            use_container_width=True
        ) and question.strip():

            with st.spinner(
                "Finding the answer..."
            ):

                answer = answer_question(
                    question,
                    extracted_text
                )

            st.markdown(
                "**Answer:**"
            )

            st.write(answer)

else:

    st.info(
        "Please upload an image to begin."
    )
