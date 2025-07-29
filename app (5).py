
import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('/content/lstm_sentiment_model (1).keras')
vectorizer = joblib.load('/content/tokenizer.pkl')

st.title('Movie Review Sentiment Analysis')
review = st.text_area('Enter a movie review:')

if st.button('Predict'):
    if review.strip():
        transformed_review = vectorizer.transform([review])
        prediction = model.predict(transformed_review)[0]
        sentiment = 'Positive' if prediction == 'positive' else 'Negative'
        st.write(f'The sentiment is: **{sentiment}**')
    else:
        st.write('Please enter a valid review.')
