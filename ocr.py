import cv2
import pytesseract
from pathlib import Path

documents_folder = Path("documents")
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

all_text = ""

for img_path in documents_folder.glob("*.*"):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv_cOLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    all_text += f"\n--- {img_path.name} ---\n{text}\n"

with open(output_folder / "extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("OCR Done")
