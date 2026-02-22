# Phase 3: Named Entity Recognition (NER) for Resume Parsing

import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding
import random
import re

# -------------------------------
# Load cleaned text
# -------------------------------
with open(r"C:/Users/layel/Desktop/gomycode/resume_cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------------
#  Define labeled training data
# -------------------------------
# Format: (text, {"entities": [(start_char, end_char, "LABEL")]})
# Example entities: NAME, EMAIL, PHONE, SKILL, EDUCATION, EXPERIENCE
TRAIN_DATA = [
    ("Layeli El Ghofrane Moumni is a student at ISG Sousse.", 
     {"entities": [(0, 26, "NAME"), (39, 49, "EDUCATION")]}),
    
    ("Email: layalimoumni@gmail.com, Phone: 56107271", 
     {"entities": [(7, 29, "EMAIL"), (38, 46, "PHONE")]}),
    
    ("Skills: Python, SQL, Power BI, Tableau", 
     {"entities": [(8, 43, "SKILL")]}),
    
    ("Work experience includes internships at AF Bodybuilding Center and GymBox.", 
     {"entities": [(35, 60, "EXPERIENCE"), (65, 71, "EXPERIENCE")]}),
]

# -------------------------------
#  Create a blank SpaCy model
# -------------------------------
nlp = spacy.blank("en")  # blank English model
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Add entity labels to NER
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# -------------------------------
#  Train the NER model
# -------------------------------
# Disable other pipes for faster training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for itn in range(50):  # number of iterations
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(2.0, 16.0, 1.5))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.3, losses=losses)
        print(f"Iteration {itn+1}, Losses: {losses}")

# -------------------------------
#  Test the trained NER model
# -------------------------------
doc = nlp(text)
print("\n------ Extracted Entities ------")
for ent in doc.ents:
    print(ent.text, ":", ent.label_)

# -------------------------------
#  Optional: Save the model
# -------------------------------
output_dir = r"C:/Users/layel/Desktop/gomycode/ner_model"
nlp.to_disk(output_dir)
print(f"\n✅ NER model saved to {output_dir}")