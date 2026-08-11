import streamlit as st
import subprocess
import sys

from pathlib import Path

from chatbot import search_document


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Intelligent Document Understanding",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "processed" not in st.session_state:

    st.session_state.processed = False


if "ocr_text" not in st.session_state:

    st.session_state.ocr_text = ""


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


if "uploaded_documents" not in st.session_state:

    st.session_state.uploaded_documents = {}


# ============================================================
# TITLE
# ============================================================

st.title(
    "📄 Intelligent Document Understanding System"
)

st.write(
    "Upload one or more document images, "
    "process them with OCR, and ask questions "
    "about their contents."
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "📁 Upload Document Image(s)",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    accept_multiple_files=True,
    key="document_uploader"
)


# ============================================================
# STORE UPLOADED FILES
# ============================================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        st.session_state.uploaded_documents[
            uploaded_file.name
        ] = uploaded_file.getvalue()


# ============================================================
# SHOW DOCUMENTS
# ============================================================

if st.session_state.uploaded_documents:

    st.subheader(
        "📁 Selected Documents"
    )

    st.write(
        f"**{len(st.session_state.uploaded_documents)} "
        f"document(s) selected.**"
    )

    for filename, file_data in (
        st.session_state.uploaded_documents.items()
    ):

        st.write(
            f"📄 **{filename}**"
        )

        st.image(
            file_data,
            caption=filename,
            use_container_width=True
        )


# ============================================================
# CLEAR DOCUMENTS
# ============================================================

if st.session_state.uploaded_documents:

    if st.button(
        "🗑️ Clear Selected Documents"
    ):

        st.session_state.uploaded_documents = {}

        st.session_state.processed = False

        st.session_state.ocr_text = ""

        st.session_state.chat_history = []

        st.rerun()


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if st.session_state.uploaded_documents:

    if st.button(
        "⚙️ Process Documents",
        type="primary",
        use_container_width=True
    ):

        documents_folder = Path(
            "documents"
        )

        documents_folder.mkdir(
            exist_ok=True
        )


        # ----------------------------------------------------
        # DELETE OLD DOCUMENTS
        # ----------------------------------------------------

        for old_file in documents_folder.iterdir():

            if old_file.is_file():

                try:

                    old_file.unlink()

                except Exception:

                    pass


        # ----------------------------------------------------
        # SAVE ALL UPLOADED DOCUMENTS
        # ----------------------------------------------------

        for filename, file_data in (
            st.session_state.uploaded_documents.items()
        ):

            file_path = (
                documents_folder /
                filename
            )

            with open(
                file_path,
                "wb"
            ) as file:

                file.write(
                    file_data
                )


        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Running OCR on all documents..."
        ):

            ocr_result = subprocess.run(
                [
                    sys.executable,
                    "ocr.py"
                ],
                capture_output=True,
                text=True
            )


        if ocr_result.returncode != 0:

            st.error(
                "❌ OCR failed."
            )

            if ocr_result.stderr:

                st.code(
                    ocr_result.stderr
                )

            st.stop()


        # ----------------------------------------------------
        # READ OCR TEXT
        # ----------------------------------------------------

        extracted_file = Path(
            "outputs/extracted_text.txt"
        )


        if not extracted_file.exists():

            st.error(
                "OCR finished but the extracted "
                "text file was not found."
            )

            st.stop()


        with open(
            extracted_file,
            "r",
            encoding="utf-8"
        ) as file:

            st.session_state.ocr_text = (
                file.read()
            )


        # ----------------------------------------------------
        # OBJECT DETECTION
        # ----------------------------------------------------

        with st.spinner(
            "🔍 Running object detection..."
        ):

            detection_result = subprocess.run(
                [
                    sys.executable,
                    "object_detection.py"
                ],
                capture_output=True,
                text=True
            )


        # ----------------------------------------------------
        # SAVE STATE
        # ----------------------------------------------------

        st.session_state.processed = True

        st.session_state.chat_history = []

        st.success(
            "✅ All documents processed successfully!"
        )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.processed:

    st.divider()

    st.header(
        "📝 Extracted Text"
    )

    st.text_area(
        "OCR Result",
        st.session_state.ocr_text,
        height=350
    )


    # ========================================================
    # QUESTION ANSWERING
    # ========================================================

    st.divider()

    st.header(
        "💬 Ask Questions About Your Documents"
    )

    st.write(
        "You can ask multiple questions. "
        "Previous questions and answers will remain visible."
    )


    # --------------------------------------------------------
    # QUESTION FORM
    # --------------------------------------------------------

    with st.form(
        "question_form",
        clear_on_submit=True
    ):

        question = st.text_input(
            "Enter your question",
            placeholder=(
                "Example: What is the main topic "
                "of the document?"
            )
        )

        ask_button = st.form_submit_button(
            "🔍 Ask Question",
            use_container_width=True
        )


    # --------------------------------------------------------
    # ANSWER QUESTION
    # --------------------------------------------------------

    if ask_button:

        if question.strip():

            with st.spinner(
                "🤔 Finding the answer..."
            ):

                answer = search_document(
                    question,
                    st.session_state.ocr_text
                )


            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

            st.rerun()


        else:

            st.warning(
                "Please enter a question."
            )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    if st.session_state.chat_history:

        st.subheader(
            "💭 Conversation"
        )


        for index, item in enumerate(
            st.session_state.chat_history,
            start=1
        ):

            st.markdown(
                f"### Question {index}"
            )

            st.write(
                item["question"]
            )

            st.markdown(
                "**Answer:**"
            )

            st.info(
                item["answer"]
            )

            st.divider()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "OCR + Object Detection + Question Answering"
)