'''
Author: Getachew Getu
Email: getachewgetu2010@gmail.com
Date: 2025-May-07
'''

import streamlit as st
import pickle
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# Define KGAT model class (same as in kgat_recommendation_system.py)
class KGAT(nn.Module):
    def __init__(self, n_entities, n_relations, embed_dim):
        super(KGAT, self).__init__()
        self.embed_dim = embed_dim
        self.entity_emb = nn.Embedding(n_entities, embed_dim)
        self.relation_emb = nn.Embedding(n_relations, embed_dim)
        self.attention = nn.Linear(embed_dim * 2, 1)
        self.W = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, edge_index, edge_type):
        head = self.entity_emb(edge_index[0])
        tail = self.entity_emb(edge_index[1])
        rel = self.relation_emb(edge_type)
        head_rel = torch.cat([head, rel], dim=-1)
        att_scores = torch.sigmoid(self.attention(head_rel))
        neighbor_emb = att_scores * tail
        aggr_emb = torch.zeros(self.entity_emb.num_embeddings, self.embed_dim).to(head.device)
        for i in range(edge_index.shape[1]):
            head_idx = edge_index[0, i]
            aggr_emb[head_idx] += neighbor_emb[i]
        entity_emb = torch.tanh(self.W(aggr_emb))
        return entity_emb

    def predict(self, user_ids, item_ids):
        user_emb = self.entity_emb(user_ids)
        item_emb = self.entity_emb(item_ids)
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores

# Load artifacts
st.header('KGAT Based Book Recommender System ')
try:
    model_state = pickle.load(open('artifacts/kgat_model.pkl', 'rb'))
    user_encoder = pickle.load(open('artifacts/user_encoder.pkl', 'rb'))
    book_encoder = pickle.load(open('artifacts/book_encoder.pkl', 'rb'))
    author_encoder = pickle.load(open('artifacts/author_encoder.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading artifacts: {e}. Please ensure all .pkl files are in the artifacts/ directory.")
    st.stop()

# Load books dataset to access image URLs
books_dtypes = {
    'ISBN': str,
    'Book-Title': str,
    'Book-Author': str,
    'Year-Of-Publication': str,
    'Publisher': str,
    'Image-URL-S': str,
    'Image-URL-M': str,
    'Image-URL-L': str
}
try:
    books = pd.read_csv('data/BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip', dtype=books_dtypes)
except FileNotFoundError as e:
    st.error(f"Error loading dataset: {e}. Please ensure BX-Books.csv is in the data/ directory.")
    st.stop()

books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']]
books.rename(columns={'Book-Title': 'title', 'Book-Author': 'author', 'Year-Of-Publication': 'year', 'Publisher': 'publisher', 'Image-URL-L': 'image_url'}, inplace=True)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_entities = len(user_encoder.classes_) + len(book_encoder.classes_) + len(author_encoder.classes_)
n_relations = 2  # 'rated' and 'written_by'
embed_dim = 64
model = KGAT(n_entities, n_relations, embed_dim).to(device)
try:
    model.load_state_dict(model_state)
except Exception as e:
    st.error(f"Error loading model state: {e}. Ensure kgat_model.pkl matches the KGAT model architecture.")
    st.stop()
model.eval()

# Create final_rating DataFrame for poster URLs
try:
    ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
except FileNotFoundError as e:
    st.error(f"Error loading dataset: {e}. Please ensure BX-Book-Ratings.csv is in the data/ directory.")
    st.stop()

ratings.rename(columns={'User-ID': 'user_id', 'Book-Rating': 'rating'}, inplace=True)
user_counts = ratings['user_id'].value_counts()
active_users = user_counts[user_counts > 200].index
ratings = ratings[ratings['user_id'].isin(active_users)]
ratings_with_books = ratings.merge(books, on='ISBN')
number_rating = ratings_with_books.groupby('title')['rating'].count().reset_index()
number_rating.rename(columns={'rating': 'num_of_rating'}, inplace=True)
final_rating = ratings_with_books.merge(number_rating, on='title')
final_rating = final_rating[final_rating['num_of_rating'] >= 50]
final_rating.drop_duplicates(['user_id', 'title'], inplace=True)
final_rating['book_id_encoded'] = book_encoder.transform(final_rating['title'])

def fetch_poster(book_titles):
    poster_url = []
    for name in book_titles:
        try:
            idx = np.where(final_rating['title'] == name)[0][0]
            url = final_rating.iloc[idx]['image_url']
            poster_url.append(url)
        except IndexError:
            poster_url.append('https://via.placeholder.com/150')  # Fallback image
    return poster_url

def recommend_books(user_id, model, book_encoder, n_users, top_k=5):
    with torch.no_grad():
        try:
            user_id_encoded = user_encoder.transform([user_id])[0]
        except ValueError:
            return [], []
        user_tensor = torch.tensor([user_id_encoded], dtype=torch.long).to(device)
        book_ids = torch.arange(n_users, n_users + len(book_encoder.classes_), dtype=torch.long).to(device)
        scores = model.predict(user_tensor.expand(len(book_ids)), book_ids)
        _, top_indices = scores.topk(top_k)
        top_book_ids = top_indices.cpu().numpy()
        recommended_books = book_encoder.inverse_transform(top_book_ids)
        poster_urls = fetch_poster(recommended_books)
        return recommended_books, poster_urls

# Streamlit UI
user_ids = user_encoder.classes_.astype(str).tolist()
selected_user = st.selectbox("Select a user ID", user_ids)

if st.button('Show Recommendations'):
    recommended_books, poster_urls = recommend_books(int(selected_user), model, book_encoder, len(user_encoder.classes_))
    if len(recommended_books) == 0:
        st.error(f"No recommendations available for user {selected_user}. Please select another user.")
    else:
        st.success(f"Recommended books for user {selected_user}:")
        cols = st.columns(5)
        for i, (book, url) in enumerate(zip(recommended_books[:5], poster_urls[:5])):
            with cols[i]:
                st.text(book)
                st.image(url, use_column_width=True)