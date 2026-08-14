import pytesseract
import cv2
import numpy as np
from pathlib import Path

documents_folder = Path("documents")
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

all_text = ""

for image_file in documents_folder.iterdir():
    if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        # 1. Read image
        img = cv2.imread(str(image_file))
        
        # 2. UPSCALE 2x - critical for blurry text
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 3. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 4. DENOISE - removes blur/noise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 5. INCREASE CONTRAST + THRESHOLD
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # 6. SAVE DEBUG IMAGE so we can check what OCR sees
        cv2.imwrite(str(output_folder / f"debug_{image_file.name}"), gray)
        
        # 7. Run OCR with LSTM neural net mode - BEST for blurry
        custom_config = r'--oem 1 --psm 3'
        text = pytesseract.image_to_string(gray, config=custom_config)
        
        all_text += f"\n\n--- From {image_file.name} ---\n\n" + text

# Save to file
with open(output_folder / "extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("OCR completed successfully")
