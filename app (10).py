import streamlit as st
import joblib
import base64
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="🎬 Interactive Movie Review Sentiment Analyzer", layout="wide")

# --- Set Background Image ---
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        backdrop-filter: blur(2px);
    }}
    .title-text {{
        background-color: rgba(0,0,0,0.6);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
    }}
    .result {{
        background-color: rgba(255,255,255,0.9);
        padding: 0.8rem;
        margin-top: 1rem;
        border-radius: 8px;
        font-size: 1.3rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("background_image.jpg")

# --- Load Model and Vectorizer ---
model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Sidebar ---
st.sidebar.header("⚙️ App Controls")
st.sidebar.info("Enter your review below and click *Analyze Sentiment*.")
show_probs = st.sidebar.checkbox("Show Prediction Probabilities", value=True)

# --- Title ---
st.markdown('<div class="title-text">🎬 Interactive Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)

# --- Text Input ---
review = st.text_area("✍️ Write your movie review here:", height=150)

# --- Session State for History ---
if "history" not in st.session_state:
    st.session_state["history"] = []

# --- Predict Button ---
if st.button("🔍 Analyze Sentiment"):
    if review.strip():
        transformed_review = vectorizer.transform([review])
        prediction = model.predict(transformed_review)[0].capitalize()
        st.session_state["history"].append((review, prediction))

        # Display Result
        st.markdown(f'<div class="result">🧠 **Predicted Sentiment:** {prediction}</div>', unsafe_allow_html=True)

        # Show Probability Chart
        if show_probs and hasattr(model, "predict_proba"):
            probs = model.predict_proba(transformed_review)[0]
            labels = model.classes_
            fig, ax = plt.subplots()
            ax.bar(labels, probs, color=["#4CAF50", "#FFC107", "#F44336"][:len(labels)])
            ax.set_title("Prediction Probabilities")
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please enter a valid review.")

# --- History ---
if st.session_state["history"]:
    st.subheader("📜 Recent Predictions")
    for i, (rev, pred) in enumerate(reversed(st.session_state["history"][-5:]), 1):
        st.markdown(f"**{i}.** *{rev[:80]}...* → 🎯 {pred}")

        
