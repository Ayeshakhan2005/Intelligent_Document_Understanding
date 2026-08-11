from ultralytics import YOLO


model = YOLO("yolo11n.pt")


def detect_objects(image_path):
    results = model.predict(
        source=image_path,
        save=True
    )

    return results


if __name__ == "__main__":
    image_path = "documents/test.jpg"

    results = detect_objects(image_path)

    print("Object detection completed successfully.")
    print("Number of results:", len(results))