import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = {
    'product_id': [1, 2, 3, 4, 5],
    'product_name': [
        'Wireless Mouse',
        'Gaming Keyboard',
        'USB-C Adapter',
        'Laptop Stand',
        'Noise Cancelling Headphones'
    ],
    'description': [
        'A sleek wireless mouse with ergonomic design.',
        'Mechanical keyboard with RGB lighting for gaming.',
        'Adapter to convert USB-C to HDMI and USB.',
        'Adjustable laptop stand for desk setup.',
        'Headphones with noise cancellation and long battery life.'
    ]
}

df = pd.DataFrame(data)


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_products(product_name, top_n=3):
    try:
        idx = df[df['product_name'].str.lower() == product_name.lower()].index[0]
    except IndexError:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    recommended = [df.iloc[i[0]]['product_name'] for i in sim_scores]
    return recommended


st.title("🛍️ E-Commerce Product Recommender")

product_list = df['product_name'].tolist()
selected_product = st.selectbox("Select a product:", product_list)

if st.button("Recommend Similar Products"):
    recommendations = recommend_products(selected_product)
    st.subheader("You may also like:")
    for rec in recommendations:
        st.write(f"🔸 {rec}")
