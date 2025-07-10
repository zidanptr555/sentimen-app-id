import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# --- Load dataset tsv
df = pd.read_csv("id-movie-review-sentimentanalysis.tsv", sep="\t", names=["no","review","sentiment"], header=None)
df = df.drop(columns="no")

# --- Data Preprocessing sederhana
df['review'] = df['review'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)

# --- TF-IDF dan label
vectorizer = TfidfVectorizer(stop_words=None)
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# --- Split dan train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# --- Aplikasi Streamlit
st.set_page_config(page_title="Sentimen Ulasan Film 🇮🇩", layout="centered")
st.title("🎬 Sentimen Ulasan Film (ID)")
st.sidebar.metric("Akurasi", f"{accuracy*100:.2f}%")
user_input = st.text_area("Tulis ulasan film kamu di sini", height=200)

if st.button("Prediksi"):
    if not user_input.strip():
        st.warning("Silakan tulis ulasan sebelum prediksi.")
    else:
        vec = vectorizer.transform([user_input.lower()])
        pred = model.predict(vec)[0]
        if pred == "positive":
            st.success("👍 Sentimen: Positif")
        else:
            st.error("👎 Sentimen: Negatif")

st.caption("Dataset: 500 tweet ulasan film (ID)")
