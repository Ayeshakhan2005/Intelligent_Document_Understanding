from preprocessing import preprocess_image
from ocr import extract_text
from object_detection import detect_objects


def process_document(input_image):
    preprocessed_image = "outputs/preprocessed_test.jpg"

    # Step 1: Preprocess
    preprocess_image(input_image, preprocessed_image)

    # Step 2: OCR
    extracted_text = extract_text(preprocessed_image)

    # Step 3: Object Detection
    detection_results = detect_objects(input_image)

    print("\nDocument processing completed successfully.")

    return extracted_text, detection_results


if __name__ == "__main__":
    image_path = "documents/test.jpg"

    text, results = process_document(image_path)

    print("\n===== EXTRACTED TEXT =====")
    print(text)

    print("\n===== OBJECT DETECTION =====")
    print(f"Detection results: {len(results)}")