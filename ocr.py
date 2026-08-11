from pathlib import Path
from paddleocr import PaddleOCR


ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False
)


def extract_text(image_path):

    print(f"\nProcessing: {image_path}")

    result = ocr.predict(str(image_path))

    extracted_text = []

    for item in result:

        # PaddleOCR result object
        if hasattr(item, "json"):
            data = item.json

            if callable(data):
                data = data()

        elif isinstance(item, dict):
            data = item

        else:
            data = {}

        # Some PaddleOCR versions return:
        # {"res": {...}}
        if "res" in data:
            data = data["res"]

        texts = data.get("rec_texts", [])

        for text in texts:

            if text and text.strip():
                extracted_text.append(text.strip())

    return "\n".join(extracted_text)


def main():

    documents_folder = Path("documents")
    output_folder = Path("outputs")

    output_folder.mkdir(exist_ok=True)

    image_files = sorted(
        {
            file
            for file in documents_folder.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        }
    )

    print(f"\nFound {len(image_files)} document(s).")

    if not image_files:

        print("No images found in the documents folder.")

        return

    all_documents = []

    for image_path in image_files:

        try:

            text = extract_text(image_path)

            print(f"\nExtracted text from {image_path.name}:")
            print(text)

            if not text:
                print("WARNING: No text was detected.")

            all_documents.append(
                f"===== {image_path.name} =====\n{text}"
            )

            individual_output = (
                output_folder /
                f"{image_path.stem}_extracted.txt"
            )

            with open(
                individual_output,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(text)

        except Exception as e:

            print(
                f"ERROR processing {image_path.name}: {e}"
            )

    combined_text = "\n\n".join(all_documents)

    combined_output = (
        output_folder / "extracted_text.txt"
    )

    with open(
        combined_output,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(combined_text)

    print("\n================================")
    print("OCR completed successfully.")
    print(
        f"Combined text saved to: {combined_output}"
    )
    print("================================")


if __name__ == "__main__":
    main()