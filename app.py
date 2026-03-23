import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE SETUP ---
st.set_page_config(page_title="SmartReview AI", layout="wide")

st.title("PRODUCT-ANALYSIS-AND-REVIEW-USING-LDA-AND-NAIVE-BAYES
")
st.markdown("**Impact Pillar: Digital Inclusion** | AI-powered recommendation engine mapping consumer sentiment.")
st.divider()

# --- ML PIPELINE (Cached so it runs lightning fast!) ---
@st.cache_resource
def run_ml_pipeline():
    # 1. LOAD DATA
    df = pd.read_csv("jumia_reviews_dataset.csv")

    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    naija_stops = ['na', 'dey', 'go', 'una', 'abeg', 'seff', 'sha', 'make', 'don', 'abi', 'wetin', 'wey']
    stop_words.update(naija_stops)

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = [w for w in text.split() if w not in stop_words]
        return " ".join(words)

    df['Cleaned_Content'] = df['Review_Content'].apply(clean_text)

    # 2. NAIVE BAYES SENTIMENT
    def get_sentiment_label(text):
        score = TextBlob(text).sentiment.polarity
        if score > 0: return 'Positive'
        elif score < 0: return 'Negative'
        else: return 'Neutral'

    df['Sentiment_Label'] = df['Cleaned_Content'].apply(get_sentiment_label)
    training_data = df[df['Sentiment_Label'] != 'Neutral']

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(training_data['Cleaned_Content'])
    
    nb_model = MultinomialNB()
    nb_model.fit(X_vec, training_data['Sentiment_Label'])

    all_vec = vectorizer.transform(df['Cleaned_Content'])
    df['NB_Predicted_Sentiment'] = nb_model.predict(all_vec)

    # 3. LDA TOPIC MODELING
    lda_model = LatentDirichletAllocation(n_components=3, random_state=42, learning_method='batch')
    lda_output = lda_model.fit_transform(all_vec)
    df['Topic_ID'] = lda_output.argmax(axis=1)

    topic_map = {
        0: "Delivery & Order Experience",
        1: "General Satisfaction",
        2: "Device Performance (Battery/Camera)"
    }
    df['Aspect'] = df['Topic_ID'].map(topic_map).fillna("Other")

    # 4. FINAL SCORING
    df['Sentiment_Score'] = df['NB_Predicted_Sentiment'].map({'Positive': 1, 'Negative': 0})
    df_scoring = df.dropna(subset=['Sentiment_Score']).copy()

    final_table = (
        df_scoring
        .groupby(['Product_Name', 'Aspect'])['Sentiment_Score']
        .mean()
        .unstack()
        .fillna(0) * 100
    ).round(1)

    return df, final_table

# --- UI EXECUTION ---
try:
    with st.spinner("Crunching Jumia reviews and training AI models... please wait!"):
        df, final_table = run_ml_pipeline()
    
    st.success("Analysis Complete!")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Brand Comparison Heatmap")
        st.write("Percentage of positive sentiment across extracted features.")
        
        # Draw the Seaborn Heatmap in Streamlit
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(final_table, annot=True, cmap="RdYlGn", fmt=".1f", linewidths=0.5, ax=ax)
        plt.ylabel("Phone Model")
        plt.xlabel("Feature Aspect")
        st.pyplot(fig)

    with col2:
        st.subheader("💡 System Recommendation")
        target_aspect = "Device Performance (Battery/Camera)"
        
        if target_aspect in final_table.columns:
            best_phone = final_table[target_aspect].idxmax()
            best_score = final_table[target_aspect].max()
            
            st.info("**User Persona Request:**\n*\"I want a phone with good battery and camera.\"*")
            st.success(f"### 👉 WINNER: {best_phone}")
            st.write(f"**Reason:** Achieved the highest satisfaction score (**{best_score}%**) for {target_aspect} out of {len(df)} total reviews analyzed.")

    st.divider()
    st.subheader("Raw Review Data Preview")
    st.dataframe(df[['Product_Name', 'Review_Content', 'NB_Predicted_Sentiment', 'Aspect']].sample(50))

except FileNotFoundError:
    st.error("🚨 Missing file! Please ensure 'jumia_reviews_dataset.csv' is uploaded exactly with this name to your GitHub repository.")
