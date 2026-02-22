# Phase 2: Resume Text Cleaning and Normalization

import re
import spacy
from dateutil import parser

#  Load the extracted text from Phase 1
with open(r"C:/Users/layel/Desktop/gomycode/resume_text.txt", "r", encoding="utf-8") as f:
    extracted_text = f.read()

#  Remove noise and irrelevant characters
def remove_noise(text):
    # Remove special characters except essential ones
    text = re.sub(r'[^a-zA-Z0-9\s,.-]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove common OCR artifacts like page numbers
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    return text.strip()

clean_text = remove_noise(extracted_text)

# Standardize dates to YYYY-MM-DD format
def standardize_dates(text):
    # Regex to capture common date patterns
    date_patterns = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
    for d in date_patterns:
        try:
            std_date = parser.parse(d, dayfirst=True).strftime('%Y-%m-%d')
            text = text.replace(d, std_date)
        except:
            continue
    return text

clean_text = standardize_dates(clean_text)

#  Normalize text using SpaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp(clean_text)

# Tokenization and Lemmatization
tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

#  Save cleaned and normalized text
with open(r"C:/Users/layel/Desktop/gomycode/resume_cleaned.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(lemmas))

# Print results
print("----- Tokens -----")
print(tokens[:50], "...")  # first 50 tokens for preview

print("\n----- Lemmas (cleaned) -----")
print(lemmas[:50], "...")  # first 50 lemmas for preview

print("\n Resume text cleaned, normalized, and saved to resume_cleaned.txt")