# resume_parser.py

from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import spacy
import re

# -----------------------------
# 1️⃣ Configure paths
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pdf_path = r"C:\Users\layel\Downloads\resum.pdf"
poppler_path = r"C:\Users\layel\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"

# -----------------------------
# 2️⃣ Convert PDF pages to images
# -----------------------------
print("Converting PDF to images...")
pages = convert_from_path(pdf_path, poppler_path=poppler_path)

# -----------------------------
# 3️⃣ Extract text from each page
# -----------------------------
print("Extracting text using OCR...")
full_text = ""
for page_number, page in enumerate(pages, start=1):
    text = pytesseract.image_to_string(page)
    full_text += text + "\n"  # separate pages

# -----------------------------
# 4️⃣ Clean extracted text
# -----------------------------
def clean_text(t):
    t = t.replace("\n", " ")
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

cleaned_text = clean_text(full_text)
print("\n=== Extracted Resume Text ===")
print(cleaned_text)

# Save cleaned text to a file
with open(r"C:\Users\layel\Desktop\gomycode\resume_text.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

# -----------------------------
# 5️⃣ Load SpaCy model for NER
# -----------------------------
print("\nLoading SpaCy NER model...")
nlp = spacy.load("en_core_web_sm")
doc = nlp(cleaned_text)

# -----------------------------
# 6️⃣ Extract named entities
# -----------------------------
print("\n=== Named Entities Found ===")
for ent in doc.ents:
    print(f"{ent.text} | {ent.label_}")

# Optional: categorize entities manually
categories = {"NAME": [], "ORG": [], "GPE": [], "SKILL": []}
for ent in doc.ents:
    if ent.label_ == "PERSON":
        categories["NAME"].append(ent.text)
    elif ent.label_ == "ORG":
        categories["ORG"].append(ent.text)
    elif ent.label_ == "GPE":
        categories["GPE"].append(ent.text)

print("\n=== Categorized Entities ===")
for k, v in categories.items():
    print(f"{k}: {v}")