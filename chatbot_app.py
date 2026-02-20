import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --------------------------
# 1️⃣ Text embedded directly
# --------------------------
text = """
Artificial Intelligence (AI) is the simulation of human intelligence in machines.
It is used in many fields including healthcare, finance, and robotics.
Machine Learning is a subset of AI that allows machines to learn from data.
Deep Learning is a subset of Machine Learning that uses neural networks.
AI is transforming the way we live and work.
"""

# --------------------------
# 2️⃣ Split text into sentences
# --------------------------
sentences = []
for sep in ['.', '!', '?']:
    parts = text.split(sep)
    sentences += [p.strip() for p in parts if p.strip() != '']

# --------------------------
# 3️⃣ Preprocess sentences
# --------------------------
def preprocess(sentence):
    return sentence.lower().translate(str.maketrans('', '', string.punctuation))

processed_sentences = [preprocess(s) for s in sentences]

# --------------------------
# 4️⃣ TF-IDF vectorization
# --------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_sentences)

# --------------------------
# 5️⃣ Get most relevant sentence
# --------------------------
def get_most_relevant_sentence(user_input):
    user_vec = vectorizer.transform([preprocess(user_input)])
    similarities = cosine_similarity(user_vec, X)
    idx = np.argmax(similarities)
    if similarities[0][idx] < 0.1:
        return "Sorry, I don't know the answer to that."
    return sentences[idx]

# --------------------------
# 6️⃣ Chatbot function
# --------------------------
def chatbot(user_input):
    return get_most_relevant_sentence(user_input)

# --------------------------
# 7️⃣ Streamlit interface
# --------------------------
st.title("🤖 AI Text Chatbot (Embedded Text)")
st.write("Ask me anything about AI!")

user_input = st.text_input("You: ")
if user_input:
    response = chatbot(user_input)
    st.text_area("Chatbot:", value=response, height=150)