import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Load model
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Sidebar
st.sidebar.title("📌 Project Information")

st.sidebar.write("""
Fake News Detection System

Model: Logistic Regression  
Vectorization: TF-IDF  
Dataset Size: 44,000+ news articles
""")

st.sidebar.write("Developed using Python, Scikit-learn, and Streamlit")

# Title
st.title("📰 Fake News Detection System")

st.markdown("""
This application uses **Machine Learning and Natural Language Processing (NLP)**  
to classify news articles as **REAL** or **FAKE**.

You can:

• Enter news text manually  
• Upload a CSV file for batch prediction
""")

st.markdown("---")

# ===============================
# Single News Detection
# ===============================

st.header("🔎 Single News Detection")

news_text = st.text_area("Enter News Text")

if st.button("Check News"):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")

    else:

        news_vector = vectorizer.transform([news_text])

        prediction = model.predict(news_vector)
        prob = model.predict_proba(news_vector)[0]

        fake_prob = prob[0] * 100
        real_prob = prob[1] * 100

        st.markdown("### 🤖 AI Analysis Result")

        if prediction[0] == 0:
            st.error("⚠️ FAKE NEWS DETECTED")
        else:
            st.success("✅ REAL NEWS DETECTED")

        # Metrics
        col1, col2 = st.columns(2)

        col1.metric("Fake Probability", f"{fake_prob:.2f}%")
        col2.metric("Real Probability", f"{real_prob:.2f}%")

        # Progress bar
        confidence = max(fake_prob, real_prob)
        st.write(f"Confidence Score: {confidence:.2f}%")
        st.progress(int(confidence))

        # Probability Chart
        labels = ["Fake", "Real"]
        values = [fake_prob, real_prob]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("Probability (%)")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

st.markdown("---")

# ===============================
# Batch News Detection
# ===============================

st.header("📂 Batch News Detection (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload CSV file containing a 'text' column",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head())

    if "text" in df.columns:

        vectors = vectorizer.transform(df["text"])

        predictions = model.predict(vectors)

        df["Prediction"] = predictions
        df["Prediction"] = df["Prediction"].map({0: "Fake", 1: "Real"})

        st.subheader("📊 Prediction Results")
        st.dataframe(df)

        # Download results
        csv = df.to_csv(index=False)

        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

    else:
        st.error("CSV file must contain a column named 'text'.")

st.markdown("---")

