import streamlit as st
import pandas as pd
import torch
import openai
from openai import OpenAI
import os
import numpy as np
import re
from dotenv import load_dotenv, find_dotenv
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertModel
import joblib

# ====== Setup ======
st.set_page_config(page_title="🛍️ Automated Customer Reviews", layout="wide")

st.markdown("""
    <style>
    .big-title {
        font-size: 36px !important;
        font-weight: bold;
        color: #2e86c1;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% {color: #2e86c1;}
        50% {color: #45b39d;}
        100% {color: #2e86c1;}
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🛒 Automated Customer Reviews Dashboard</div>', unsafe_allow_html=True)
st.write("Upload a review dataset or type a review directly to get started!")

# ====== Load Models & Tokenizer ======
def load_sentiment_model():
    # Load the tokenizer and model for sentiment classification
    tokenizer = AutoTokenizer.from_pretrained("best_sentiment_model")
    model = AutoModelForSequenceClassification.from_pretrained("best_sentiment_model")

    return tokenizer, model

@st.cache_resource
def load_bert_embedding_model():
    model = DistilBertModel.from_pretrained("distilbert-base-uncased") 
    return model

tokenizer, classification_model = load_sentiment_model()
bert_model = load_bert_embedding_model()
kmeans = joblib.load('kmeans_model.pkl')

cluster_names = {
    0: "Pet & Home Essentials",
    1: "E-Readers & Kids Tablets",
    2: "Smart Home Devices & Accessories",
    3: "Fire Tablets",
    4: "Accessories"
}


# ====== Utilities ======
def get_embeddings(text, tokenizer, model):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Generate embeddings using the model
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Get the average of the last hidden state
    
    return embeddings.squeeze().numpy()  # Convert to a numpy array

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def simple_sent_tokenize(text):
    return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

def analyze_multi_product_review(review):
    sentences = simple_sent_tokenize(review)
    results = []
    for sent in sentences:
        sentiment = classify_sentiment(sent)
        cluster_id, cluster_name = get_cluster(sent, tokenizer, bert_model, kmeans, cluster_names)
        results.append((sent, sentiment, cluster_name))
    return results
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = classification_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
 
    return label_map[pred_class]

def get_cluster(user_review, tokenizer, model, kmeans, cluster_names):
    emb = get_embeddings(user_review, tokenizer, model)
    emb = np.array([emb])
    label = kmeans.predict(emb)[0]
    return label, cluster_names[label]


# Load your key from .env
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Use the client 
client = OpenAI(api_key=OPENAI_API_KEY)


def summarize_category(df_cat, category_name):
    top_df = df_cat[df_cat['sentiment'] == 'Positive']
    top_products = top_df.groupby('name')['reviews.rating'].mean().sort_values(ascending=False).head(3).reset_index()

    neg_reviews = df_cat[df_cat['sentiment'] == 'Negative']
    neg_reviews = neg_reviews[neg_reviews['cleaned_review'].str.strip().astype(bool)]

    if not neg_reviews.empty:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
        X = vectorizer.fit_transform(neg_reviews['cleaned_review'])
        word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))
        complaints = [w for w, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]]
    else:
        complaints = []

    worst_product = neg_reviews.groupby('name')['reviews.rating'].count().sort_values(ascending=False).head(1).reset_index()


    prompt = f"""
    Here is a list of the top 3 products in the {category_name} category, including their ratings and features:

    {top_products[['name', 'reviews.rating']].to_string(index=False)}

    **Top Complaints for the products:**
    - {', '.join([c.split(':')[0] for c in complaints])}

    **Worst Product:**
    - {worst_product[['name', 'reviews.rating']].to_string(index=False)}

    Please generate a blog post that:
    1. Summarizes the top 3 products in the {category_name} category, highlighting their key features and why they stand out based on user reviews.
    2. Mentions the common complaints users have for these top products, and provides context for these complaints based on the reviews.
    3. Highlights the worst product in this category, explaining why it underperformed compared to the others, based on user feedback.
    4. Make the blog post engaging, clear, and informative, ensuring that readers can easily understand the key points about both the best and worst products in this category.
    """

    # Call OpenAI API for summarization
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def summarize_single_review(review, category_name):
    prompt = f"""
    The user wrote the following product review:  
    "{review}"  
    This product belongs to the "{category_name}" category.  

    Please write a short summary that:
    1. Highlights what the review is about.
    2. Reflects the reviewer's sentiment (positive, neutral, or negative).
    3. Describes key points mentioned in the review.
    Make it clear and helpful for others who might be considering this category.
    """

    # Call OpenAI API for summarization
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content



# ====== Sidebar ======
with st.sidebar:
    st.header("🗂️ Options")
    input_mode = st.radio("Choose Input Method:", ["Upload File", "Type Review"])
    if input_mode == "Upload File":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
    else:
        #product_name = st.text_input("Enter the product name:")
        user_review = st.text_area("Write your review:")


# ====== Main App ======
if input_mode == "Upload File" and uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")        
    if 'cleaned_review' not in df.columns:
        st.warning("🧹 'cleaned_review' column not found. Cleaning reviews.text now...")

        # Clean text
        with st.spinner("🧼 Cleaning text..."):
            df['cleaned_review'] = df['reviews.text'].apply(clean_text)
        
        st.success("✅ Text cleaned and 'cleaned_review' column created.")

        # Sentiment classification
        with st.spinner("🔍 Classifying sentiment..."):
            df["sentiment"] = df["cleaned_review"].apply(classify_sentiment)

        st.success("✅ Sentiment classified.")

        # Clustering
        with st.spinner("🔄 Clustering reviews into categories..."):
            df["cluster"] = df["name"].apply(lambda x: get_cluster(x)[0])
            df["cluster_name"] = df["cluster"].map(cluster_names)

        st.success("✅ Reviews grouped into product categories.")

        # Select category to display summary
        selected_category = st.selectbox("📂 Select a Product Category:", df["cluster_name"].unique())
        df_cat = df[df["cluster_name"] == selected_category]

        # Generate blog post
        st.subheader(f"📝 Review Summary for {selected_category}")
        with st.spinner("🧠 Generating blog post summary..."):
            summary = summarize_category(df_cat, selected_category)
            st.markdown(summary)

        st.subheader("📊 Sentiment Distribution")
        st.bar_chart(df_cat["sentiment"].value_counts())

        st.subheader("📌 Sample Reviews")
        st.dataframe(df_cat[["name", "reviews.text", "sentiment"]].head(5))


    else:
        df["sentiment"] = df["cleaned_review"].apply(classify_sentiment)
        df["cluster"] = df["name"].apply(lambda x: get_cluster(x)[0])
        df["cluster_name"] = df["cluster"].map(cluster_names)

        st.success("✅ File processed successfully!")
        selected_category = st.selectbox("Select a Product Category:", df["cluster_name"].unique())
        df_cat = df[df["cluster_name"] == selected_category]

        st.subheader(f"📝 Review Summary for {selected_category}")
        summary = summarize_category(df_cat, selected_category)
        st.markdown(summary)

        st.subheader("📊 Review Distribution")
        st.bar_chart(df_cat["sentiment"].value_counts())

        st.subheader("📌 Sample Reviews")
        st.dataframe(df_cat[["name", "reviews.text", "sentiment"]].head(5))

elif input_mode == "Type Review" and user_review.strip():
    st.success("🧠 Review Analyzed!")
    with st.spinner("Analyzing each part of your review..."):
        results = analyze_multi_product_review(user_review)

    for sent, sentiment, category in results:
        st.markdown(f"**Sentence:** {sent}")
        st.markdown(f"- **Sentiment:** {sentiment}")
        st.markdown(f"- **Predicted Category:** {category}")
        with st.spinner("Generating summary..."):
            summary = summarize_single_review(sent, category)
            st.markdown(f"**📝 Summary:**\n{summary}")
        st.markdown("---")
