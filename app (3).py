import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the model
model = tf.keras.models.load_model("lstm_sentiment_model.h5")

# Constants
MAX_LEN = 100
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a review and get sentiment prediction using your LSTM model.")

user_input = st.text_area("Enter your review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        pred = model.predict(padded)
        label = label_map[np.argmax(pred)]

        st.success(f"Predicted Sentiment: **{label}**")
