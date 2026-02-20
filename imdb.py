import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('IMDB Dataset.csv')


print(df.head())
print(df.info())
print(df['sentiment'].value_counts())


sns.countplot(x='sentiment', data=df)
plt.title("Distribution of Sentiments")
plt.show()

df['review_length'] = df['review'].apply(lambda x: len(x.split()))
plt.hist(df['review_length'], bins=50)
plt.title("Distribution of Review Lengths")
plt.xlabel("Number of words")
plt.ylabel("Frequency")
plt.show()
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  
    return ' '.join(tokens)

df['clean_review'] = df['review'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['sentiment'], test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

input_dim = X_train_tfidf.shape[1] 

model = Sequential([
    Dense(128, input_dim=input_dim, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()
history = model.fit(
    X_train_tfidf.toarray(), 
    y_train.map({'positive': 1, 'negative': 0}).values,
    validation_split=0.2,
    epochs=5,  
    batch_size=64,
    verbose=1
)

loss, accuracy = model.evaluate(
    X_test_tfidf.toarray(), 
    y_test.map({'positive': 1, 'negative': 0}).values
)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()