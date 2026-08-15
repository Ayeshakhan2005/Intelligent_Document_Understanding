import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from google import genai
from google.genai import types
import io


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Intelligent Document Q&A",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Intelligent Document Q&A")
st.caption(
    "Multilingual image understanding, text extraction and Q&A"
)


# ============================================================
# API
# ============================================================

if "GEMINI_API_KEY" not in st.secrets:
    st.error(
        "Gemini API key is missing. "
        "Add GEMINI_API_KEY in Streamlit Secrets."
    )
    st.stop()

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

MODEL = "gemini-3.6-flash"


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image):
    """
    Create a high-quality version of the image without
    destroying the original information.
    """

    image = image.convert("RGB")

    width, height = image.size

    # Avoid making extremely large images.
    maximum = 4000

    if max(width, height) > maximum:

        scale = maximum / max(width, height)

        image = image.resize(
            (
                int(width * scale),
                int(height * scale)
            ),
            Image.Resampling.LANCZOS
        )

    # Gentle enhancement for blurry/low-quality photos.
    enhanced = ImageEnhance.Contrast(
        image
    ).enhance(1.15)

    enhanced = ImageEnhance.Sharpness(
        enhanced
    ).enhance(1.25)

    return image, enhanced


# ============================================================
# IMAGE → BYTES
# ============================================================

def image_bytes(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95
    )

    return buffer.getvalue()


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_document(image):

    original, enhanced = prepare_image(image)

    prompt = """
You are a highly accurate document OCR and document-understanding
system.

Analyze the uploaded image itself. Do NOT rely on assumptions.

Extract all information that is actually visible.

GENERAL OCR:
- Read printed text.
- Read scanned documents.
- Try to read blurry or low-quality photographs.
- Handle different fonts and sizes.
- Handle rotated or skewed documents.
- Handle one-column and multi-column layouts.
- Preserve the natural reading order.

LANGUAGES:
- Detect the language/script automatically.
- Support English and other Latin-script languages.
- Support Urdu and Arabic script when visible.
- Support Hindi/Devanagari and other scripts when readable.
- NEVER translate the text unless requested.
- Preserve the original script.

MATHEMATICS:
- Recognize equations, numbers, symbols and formulas.
- Preserve mathematical expressions accurately.
- Use LaTeX when useful.

TABLES:
- Preserve rows and columns.
- Represent tables as Markdown tables when possible.

IMPORTANT:
- Do NOT invent missing words.
- Do NOT guess badly blurred text.
- If something cannot genuinely be read, write [UNREADABLE].
- Preserve names, dates, numbers, punctuation and symbols.
- Do not summarize.
- Do not explain your process.

Return ONLY the extracted document content.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(
                data=image_bytes(enhanced),
                mime_type="image/jpeg"
            )
        ]
    )

    if not response.text:
        return ""

    return response.text.strip()


# ============================================================
# QUESTION ANSWERING
# ============================================================

def answer_question(image, extracted_text, question):

    original, enhanced = prepare_image(image)

    prompt = f"""
You are an expert document question-answering system.

The user has uploaded a document image.

You have TWO sources:

1. The original document image.
2. OCR text extracted from the document.

OCR TEXT:
-------------------------
{extracted_text}
-------------------------

USER QUESTION:
{question}

RULES:

1. Carefully inspect the ORIGINAL IMAGE.
2. Use the OCR text as supporting information.
3. If OCR contains an error, use the image to correct it.
4. Answer only from information present in the document.
5. Understand the meaning of the question, not merely matching words.
6. Support questions about:
   - names
   - dates
   - places
   - numbers
   - paragraphs
   - tables
   - lists
   - headings
   - mathematical expressions
   - Urdu/Arabic text
   - other languages
7. If a calculation is required, calculate it carefully.
8. If the answer is explicitly present, give the direct answer.
9. If the document does not contain enough information, say:
   "The answer cannot be determined from this document."
10. Never invent information.
11. Keep the final answer clear and reasonably concise.

Return only the answer.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(
                data=image_bytes(original),
                mime_type="image/jpeg"
            )
        ]
    )

    if not response.text:
        return "I could not determine the answer."

    return response.text.strip()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a document/image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp"
    ]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # ========================================================
    # EXTRACTION
    # ========================================================

    if st.button(
        "🔍 Extract Text",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing the document..."
        ):

            try:

                text = extract_document(
                    image
                )

                if text:

                    st.session_state[
                        "extracted_text"
                    ] = text

                    st.success(
                        "Document extracted successfully!"
                    )

                else:

                    st.error(
                        "No readable content was detected."
                    )

            except Exception as e:

                st.error(
                    "Document processing failed."
                )

                st.code(
                    str(e)
                )


    # ========================================================
    # EXTRACTED TEXT
    # ========================================================

    if "extracted_text" in st.session_state:

        extracted_text = st.session_state[
            "extracted_text"
        ]

        st.subheader(
            "📝 Extracted Text"
        )

        st.text_area(
            "Document content",
            extracted_text,
            height=450
        )

        st.download_button(
            "⬇️ Download Text",
            extracted_text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )

        # ====================================================
        # Q&A
        # ====================================================

        st.subheader(
            "💬 Ask Questions"
        )

        question = st.text_input(
            "Ask anything about this document:"
        )

        if st.button(
            "🤖 Get Answer",
            use_container_width=True
        ) and question.strip():

            with st.spinner(
                "Reading the document and answering..."
            ):

                try:

                    answer = answer_question(
                        image,
                        extracted_text,
                        question
                    )

                    st.success(
                        answer
                    )

                except Exception as e:

                    st.error(
                        "Could not generate the answer."
                    )

                    st.code(
                        str(e)
                    )

else:

    st.info(
        "Upload an image to begin."
    )
