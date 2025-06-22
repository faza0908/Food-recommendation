# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

# Load the data
@st.cache_data
def load_data():
    rating = pd.read_csv('ratings.csv')
    food = pd.read_csv('1662574418893344.csv')
    rating.dropna(inplace=True)
    rating[['User_ID' ,'Food_ID']] =rating[['User_ID' ,'Food_ID']].astype('int64')
    food_rating = pd.merge(food ,rating ,on='Food_ID')
    user_item_matrix = food_rating.pivot_table(index='User_ID', columns='Food_ID', values='Rating').fillna(0)
    return food, user_item_matrix

food, user_item_matrix = load_data()

# Item-based Collaborative Filtering Recommendation Function
@st.cache_data
def recommend_items_cf(user_id, user_item_matrix, n_recommendations=5):
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    user_ratings = user_item_matrix.loc[user_id]
    user_unrated_items = user_ratings[user_ratings == 0].index.tolist()

    item_scores = {}
    for item in user_unrated_items:
        similar_items = item_similarity_df[item]
        user_scores = user_ratings[user_ratings > 0]
        item_scores[item] = sum(user_scores * similar_items[user_scores.index]) / (sum(similar_items[user_scores.index]) + 1e-9)

    recommended_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    return [item for item, score in recommended_items]

# Matrix Factorization Recommendation Function
@st.cache_data
def recommend_mf(userid, user_item_matrix, n=5):
    user_item_matrix_csr = csr_matrix(user_item_matrix.values)
    u, sigma, vt = svds(user_item_matrix_csr, k=min(user_item_matrix_csr.shape)-1)
    sigma_diag = np.diag(sigma)
    predicted_rating = np.dot(np.dot(u, sigma_diag), vt)
    predicted_rating_df = pd.DataFrame(predicted_rating, index=user_item_matrix.index, columns=user_item_matrix.columns)

    user_items = predicted_rating_df.iloc[userid - 1, :] # Adjust index for 0-based
    r = user_items.nlargest(n).index
    return sorted(r)

# Streamlit App
st.title('Food Recommendation System')

user_id = st.number_input('Enter User ID:', min_value=1, max_value=100, value=1)
n_recommendations = st.slider('Number of recommendations:', min_value=1, max_value=20, value=5)
recommendation_method = st.selectbox('Select Recommendation Method:', ('Item-based Collaborative Filtering', 'Matrix Factorization'))

if st.button('Get Recommendations'):
    if recommendation_method == 'Item-based Collaborative Filtering':
        recommended_food_ids = recommend_items_cf(user_id, user_item_matrix, n_recommendations)
        st.subheader('Recommended Food Items (Item-based CF):')
    else:
        recommended_food_ids = recommend_mf(user_id, user_item_matrix, n_recommendations)
        st.subheader('Recommended Food Items (Matrix Factorization):')

    recommended_foods = food[food['Food_ID'].isin(recommended_food_ids)]
    st.table(recommended_foods)
