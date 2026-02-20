import streamlit as st
import nltk
import random
import string
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    with open('chatbot.txt', 'r', errors='ignore') as file:
        raw_text = file.read().lower()
except FileNotFoundError:
    st.error("chatbot.txt not found! Please create the file in the same folder.")
    st.stop()

sent_tokens = nltk.sent_tokenize(raw_text)
word_tokens = nltk.word_tokenize(raw_text)


lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def chatbot_response(user_input):
    sent_tokens.append(user_input)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        response = "I am sorry! I don't understand you."
    else:
        response = sent_tokens[idx]
    sent_tokens.pop()
    return response


def transcribe_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(" Speak now...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return " Sorry, I could not understand the audio."
    except sr.RequestError:
        return "⚠️ API unavailable."

def main():
    st.title("🎤🤖 Speech-Enabled Chatbot")
    st.write("Choose your input method:")

    input_method = st.radio("Input Type:", ("Text", "Speech"))

    if input_method == "Text":
        user_input = st.text_input("Type your message:")

        if st.button("Send"):
            if user_input:
                response = chatbot_response(user_input)
                st.success("Bot: " + response)

    elif input_method == "Speech":
        if st.button("Start Recording"):
            speech_text = transcribe_speech()
            st.write("You said:", speech_text)
            if speech_text:
                response = chatbot_response(speech_text)
                st.success("Bot: " + response)

if __name__ == "__main__":
    main()