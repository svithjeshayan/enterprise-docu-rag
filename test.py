# test_ocr.py ← Run this file directly
from pdf2image import convert_from_path
import pytesseract
from pathlib import Path

# ←←← YOUR EXACT PATHS ←←←
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"E:\Downloads\Release-25.11.0-0\poppler-25.11.0\Library\bin"
PDF_PATH = r"E:\My Projects\Chatbot For excisting model\Server\epa_sample_letter_sent_to_commissioners_dated_february_29_2015.pdf"   # ← CHANGE TO YOUR REAL PDF NAME

# Test
print("Converting PDF to images...")
images = convert_from_path(PDF_PATH, dpi=300, poppler_path=POPPLER_PATH)
print(f"Success → {len(images)} pages")

print("\nRunning OCR on first page...")
text = pytesseract.image_to_string(images[0], lang='eng')
print("\n=== FIRST 1000 CHARACTERS EXTRACTED ===")
print(text[:1000])

if len(text) > 100:
    print(f"\nSUCCESS! OCR WORKS → {len(text)} characters from page 1")
else:
    print("\nFAIL → OCR returned almost nothing")