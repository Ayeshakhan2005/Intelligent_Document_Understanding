import streamlit as st
import tempfile
import os
from pathlib import Path
from paddlex import create_pipeline

st.set_page_config(
    page_title="Intelligent Document Understanding",
    layout="wide"
)

st.title("📄 Intelligent Document Understanding")
st.write(
    "Upload a document image and extract text, tables, formulas, "
    "and structured information."
)


@st.cache_resource
def load_pipeline():
    return create_pipeline(
        pipeline="PP-StructureV3"
    )


try:
    pipeline = load_pipeline()
except Exception as e:
    st.error("OCR system could not start.")
    st.code(str(e))
    st.stop()


uploaded_file = st.file_uploader(
    "Upload a document",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
        "pdf"
    ]
)


if uploaded_file:

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        input_path = tmp.name

    st.image(
        uploaded_file,
        caption=uploaded_file.name,
        use_container_width=True
    )

    st.subheader("🔍 Processing")

    try:

        with st.spinner(
            "Analyzing document, layout, text, tables and formulas..."
        ):

            results = pipeline.predict(
                input=input_path,

                # Helps with rotated documents
                use_doc_orientation_classify=True,

                # Helps with distorted/warped document photos
                use_doc_unwarping=True,

                # Helps with rotated text lines
                use_textline_orientation=True
            )

            extracted_text = ""
            markdown_output = ""

            for result in results:

                # Get structured markdown when available
                try:
                    md = result.markdown

                    if isinstance(md, dict):
                        text = md.get("text", "")
                    else:
                        text = str(md)

                    if text:
                        markdown_output += text + "\n\n"

                except Exception:
                    pass

                # Try OCR result
                try:
                    ocr_result = result.json

                    if isinstance(ocr_result, dict):
                        overall = ocr_result.get(
                            "overall_ocr_res",
                            {}
                        )

                        rec_texts = overall.get(
                            "rec_texts",
                            []
                        )

                        if rec_texts:
                            extracted_text += (
                                "\n".join(rec_texts)
                                + "\n"
                            )

                except Exception:
                    pass

            # Prefer structured markdown
            if markdown_output.strip():
                final_text = markdown_output.strip()
            else:
                final_text = extracted_text.strip()

        st.success("Document processed successfully!")

        st.subheader("📝 Extracted Content")

        if final_text:

            st.text_area(
                "Result",
                final_text,
                height=500
            )

            st.download_button(
                "⬇️ Download Extracted Text",
                final_text,
                "extracted_text.txt",
                mime="text/plain"
            )

        else:

            st.warning(
                "The OCR pipeline did not return readable text."
            )

    except Exception as e:

        st.error("Error while processing the document.")

        st.code(str(e))

    finally:

        if os.path.exists(input_path):
            os.remove(input_path)

else:

    st.info(
        "Upload a document image to start."
    )
