# Phase 4: Relationship Extraction from Resume Entities

import spacy

# -------------------------------
#  Load the trained NER model from Phase 3
# -------------------------------
nlp = spacy.load(r"C:/Users/layel/Desktop/gomycode/ner_model")

# -------------------------------
#  Load cleaned resume text
# -------------------------------
with open(r"C:/Users/layel/Desktop/gomycode/resume_cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()

doc = nlp(text)

# -------------------------------
#  Extract entities
# -------------------------------
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Entities Found:")
for ent in entities:
    print(ent)

# -------------------------------
# Define simple relationship rules
# -------------------------------
# Example relationships: "Person -> Education", "Person -> Experience", "Person -> Skill"

relationships = []

for sent in doc.sents:
    sent_doc = nlp(sent.text)
    person_entities = [ent for ent in sent_doc.ents if ent.label_ == "NAME"]
    education_entities = [ent for ent in sent_doc.ents if ent.label_ == "EDUCATION"]
    skill_entities = [ent for ent in sent_doc.ents if ent.label_ == "SKILL"]
    experience_entities = [ent for ent in sent_doc.ents if ent.label_ == "EXPERIENCE"]

    # Link Person -> Education
    for person in person_entities:
        for edu in education_entities:
            relationships.append((person.text, "has_education", edu.text))

    # Link Person -> Skills
    for person in person_entities:
        for skill in skill_entities:
            relationships.append((person.text, "has_skill", skill.text))

    # Link Person -> Experience
    for person in person_entities:
        for exp in experience_entities:
            relationships.append((person.text, "has_experience", exp.text))

# -------------------------------
#  Print extracted relationships
# -------------------------------
print("\n------ Extracted Relationships ------")
for rel in relationships:
    print(f"{rel[0]} --{rel[1]}--> {rel[2]}")

# -------------------------------
# Optional: Save relationships to a file
# -------------------------------
with open(r"C:/Users/layel/Desktop/gomycode/resume_relationships.txt", "w", encoding="utf-8") as f:
    for rel in relationships:
        f.write(f"{rel[0]} --{rel[1]}--> {rel[2]}\n")