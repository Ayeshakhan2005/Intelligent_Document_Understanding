import cv2


def preprocess_image(input_path, output_path):
    image = cv2.imread(input_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {input_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    cv2.imwrite(output_path, processed)

    return output_path


if __name__ == "__main__":
    input_path = "documents/test.jpg"
    output_path = "outputs/preprocessed_test.jpg"

    preprocess_image(input_path, output_path)

    print("Preprocessing completed successfully.")
    print(f"Processed image saved to: {output_path}")