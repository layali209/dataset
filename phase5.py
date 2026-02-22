# Phase 5: Store Parsed Resume Data in a Structured Database

import sqlite3

# -------------------------------
#  Connect to SQLite database (or create one)
# -------------------------------
conn = sqlite3.connect(r"C:/Users/layel/Desktop/gomycode/resume_data.db")
cursor = conn.cursor()

# -------------------------------
#  Create tables for entities and relationships
# -------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS Persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Education (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    degree TEXT,
    FOREIGN KEY (person_id) REFERENCES Persons(id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    skill TEXT,
    FOREIGN KEY (person_id) REFERENCES Persons(id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Experience (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    experience TEXT,
    FOREIGN KEY (person_id) REFERENCES Persons(id)
)
""")

conn.commit()

# -------------------------------
#  Load relationships extracted in Phase 4
# -------------------------------
relationships_file = r"C:/Users/layel/Desktop/gomycode/resume_relationships.txt"
with open(relationships_file, "r", encoding="utf-8") as f:
    relationships = f.readlines()

# -------------------------------
# Insert data into database
# -------------------------------
person_ids = {}  # to keep track of person name -> id

for line in relationships:
    line = line.strip()
    if not line:
        continue
    person, relation, value = line.split(" --")
    relation = relation.replace(">", "").strip()
    value = value.strip()

    # Insert person if not already in database
    if person not in person_ids:
        cursor.execute("INSERT INTO Persons (name) VALUES (?)", (person,))
        person_id = cursor.lastrowid
        person_ids[person] = person_id
    else:
        person_id = person_ids[person]

    # Insert related data
    if relation == "has_education":
        cursor.execute("INSERT INTO Education (person_id, degree) VALUES (?, ?)", (person_id, value))
    elif relation == "has_skill":
        cursor.execute("INSERT INTO Skills (person_id, skill) VALUES (?, ?)", (person_id, value))
    elif relation == "has_experience":
        cursor.execute("INSERT INTO Experience (person_id, experience) VALUES (?, ?)", (person_id, value))

conn.commit()

# -------------------------------
#  Example queries
# -------------------------------
print("All Persons:")
for row in cursor.execute("SELECT * FROM Persons"):
    print(row)

print("\nSkills of all Persons:")
for row in cursor.execute("""
SELECT p.name, s.skill
FROM Persons p
JOIN Skills s ON p.id = s.person_id
"""):
    print(row)

conn.close()