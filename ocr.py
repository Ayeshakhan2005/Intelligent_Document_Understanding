import pytesseract
import cv2
from pathlib import Path

documents_folder = Path("documents")
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

all_text = ""

for image_file in documents_folder.iterdir():
    if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        # Read image
        img = cv2.imread(str(image_file))
        
        # Convert to grayscale <- FIXED THIS LINE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Improve contrast
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Run OCR
        text = pytesseract.image_to_string(gray)
        all_text += f"\n\n--- From {image_file.name} ---\n\n" + text

# Save to file
with open(output_folder / "extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("OCR completed successfully")
