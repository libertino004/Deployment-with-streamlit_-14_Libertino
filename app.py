import streamlit as st
from spam_detector import load_model, predict_spam

st.title("📩 Spam Message Detection")

user_input = st.text_area("Enter a message to classify:")

if st.button("Predict"):
    model, vectorizer = load_model()
    prediction = predict_spam(user_input, model, vectorizer)
    if prediction == 'spam':
        st.error("🚫 This is a SPAM message!")
    else:
        st.success("✅ This is a HAM (not spam) message.")
