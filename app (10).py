import streamlit as st
import joblib
import base64
import pandas as pd
import random

# --- Page Config ---
st.set_page_config(page_title="🎬 SENTIMENTAL ANALYSIS ON IMDB MOVIE REVIEWS", layout="wide")

# --- Background ---
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
        background-color: rgba(0,0,0,0.7);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("background_image.jpg")

# --- Load Model & Vectorizer ---
model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Sidebar Navigation ---
st.sidebar.title("")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📝 Analyze Review", "📂 Batch Analysis", "ℹ️ About"])

# --- Home ---
if page == "🏠 Home":
    st.markdown('<div class="title-text">🎬 SENTIMENTAL ANALYSIS ON IMDB MOVIE REVIEWS</div>', unsafe_allow_html=True)
    st.write("""
    Welcome! This app analyzes movie reviews and predicts whether they are **Positive, Negative**.
    
    ### 🚀 Features:
    - Single review analysis  
    - Batch file analysis (CSV/TXT upload)  
    - Sentiment probability charts  
    - Recent history tracking  
    - Downloadable results  
    
    
    """)

# --- Analyze Single Review ---
elif page == "📝 Analyze Review":
    st.markdown('<div class="title-text">📝 Analyze a Single Review</div>', unsafe_allow_html=True)

    # Preloaded random reviews
    samples = {
        "Positive": "The movie was absolutely fantastic! I loved every moment of it.",
        "Negative": "This was the worst film I’ve ever seen. A complete waste of time.",
        "Neutral": "The movie was okay, nothing too special but not terrible either."
    }

    col1, col2 = st.columns([2,1])

    with col1:
        review = st.text_area("✍️ Write your movie review here:", height=150)



    if st.button("🔍 Analyze"):
        if review.strip():
            transformed = vectorizer.transform([review])
            prediction = model.predict(transformed)[0].capitalize()
            probs = model.predict_proba(transformed)[0]

            st.success(f"🧠 Predicted Sentiment: **{prediction}**")

            # Probability chart
            prob_dict = {label: prob for label, prob in zip(model.classes_, probs)}
            import altair as alt

# Convert prob_dict to DataFrame
prob_df = pd.DataFrame({
    "Sentiment": list(prob_dict.keys()),
    "Probability": list(prob_dict.values())
})

# Create custom bar chart
chart = (
    alt.Chart(prob_df)
    .mark_bar(cornerRadius=8, size=60)  # Rounded, thicker bars
    .encode(
        x=alt.X("Sentiment", sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Sentiment", scale=alt.Scale(scheme="set2")),
        tooltip=["Sentiment", "Probability"]
    )
    .properties(width=500, height=400, title="Sentiment Prediction Probabilities")
)

st.altair_chart(chart, use_container_width=True)



# --- Batch Analysis ---
elif page == "📂 Batch Analysis":
    st.markdown('<div class="title-text">📂 Batch Review Analysis</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV/TXT file with reviews", type=["csv", "txt"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            if "review" not in df.columns:
                st.error("CSV must have a column named 'review'")
            else:
                reviews = df["review"].astype(str).tolist()
        else:
            reviews = uploaded_file.read().decode("utf-8").splitlines()

        if reviews:
            transformed = vectorizer.transform(reviews)
            preds = model.predict(transformed)
            probs = model.predict_proba(transformed)

            results_df = pd.DataFrame({
                "Review": reviews,
                "Prediction": preds
            })

            st.write("✅ Batch Analysis Complete:")
            st.dataframe(results_df.head(10))

            # Sentiment distribution
            st.subheader("📊 Sentiment Distribution")
            st.bar_chart(results_df["Prediction"].value_counts())

            # Download option
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results", csv, "sentiment_results.csv", "text/csv")

# --- About ---
elif page == "ℹ️ About":
    st.markdown('<div class="title-text">ℹ️ About this App</div>', unsafe_allow_html=True)
    st.write("""
    This interactive web app is built with **Streamlit** and uses a **Logistic Regression model** trained on IMDB reviews.  
    
    - Author: *Your Name*  
    - Tools: Python, Streamlit, Scikit-learn  
    - Dataset: IMDB Movie Reviews  
    
    💡 Future improvements:  
    - Add more ML models (LSTM, Transformers)  
    - Support multilingual reviews  
    - Deploy with a database for storing user feedback  
    """)


        
