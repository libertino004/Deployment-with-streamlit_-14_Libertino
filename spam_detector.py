import joblib
import string
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def load_model():
    return joblib.load("spam_classifier_model.pkl"), joblib.load("vectorizer.pkl")

def predict_spam(text, model, vectorizer):
    cleaned = clean_text(text)
    vect_text = vectorizer.transform([cleaned])
    return model.predict(vect_text)[0]
