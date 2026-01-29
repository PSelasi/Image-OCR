import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

print("✓ OpenCV version:", cv2.__version__)
print("✓ NumPy version:", np.__version__)

try:
    print("✓ Tesseract version:", pytesseract.get_tesseract_version())
    print("\n✅ All dependencies installed successfully!")
except:
    print("❌ Tesseract not found. Please check installation.")