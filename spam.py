import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit app
st.title('SMS Spam Detection')
st.subheader('Check if an SMS is spam:')

# User input
user_input = st.text_area("Enter SMS message:", 
                         "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005...")

if st.button('Detect'):
    # Vectorize the input
    X = vectorizer.transform([user_input])
    
    # Make prediction
    prediction = model.predict(X)
    proba = model.predict_proba(X)
    
    # Display results
    st.subheader("Detection Result:")
    if prediction[0] == 'spam':
        st.write("**Result:** 💷 Spam")
    else:
        st.write("**Result:** 🆗 Ham")
    
    st.write(f"Spam Probability: {proba[0][1]:.2f}")