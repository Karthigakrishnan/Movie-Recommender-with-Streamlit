import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_data
def load_data():
    movies = pd.read_csv("C:\\PROJECTS GITHUB\\Movie Recommendation Model\\data\\movies.csv")
    ratings = pd.read_csv("C:\\PROJECTS GITHUB\\Movie Recommendation Model\\data\\ratings.csv")
    df = pd.merge(ratings, movies, on='movieId')
    return df, movies

# Build a movie-user matrix (each row: movie, columns: users)
def create_movie_user_matrix(df):
    return df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

def build_similarity_matrix(movie_user_matrix):
    return cosine_similarity(movie_user_matrix)

def get_similar_movies(movie_name, similarity_matrix, movie_titles):
    if movie_name not in movie_titles:
        return []

    index = movie_titles.index(movie_name)
    sim_scores = list(enumerate(similarity_matrix[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_similar = [movie_titles[i] for i, _ in sim_scores[1:11]]  # Skip self
    return top_similar

# App
st.title("🎬 Movie Recommendation System")
df, movies_df = load_data()

st.sidebar.title("Select a Movie")
movie_user_matrix = create_movie_user_matrix(df)
movie_titles = movie_user_matrix.index.tolist()
selected_movie = st.sidebar.selectbox("Choose a movie you like:", sorted(movie_titles))

similarity_matrix = build_similarity_matrix(movie_user_matrix)

if selected_movie:
    recommendations = get_similar_movies(selected_movie, similarity_matrix, movie_titles)

    st.subheader(f"✨ Movies similar to: **{selected_movie}**")
    for i, title in enumerate(recommendations, 1):
        st.write(f"{i}. 🎥 **{title}**")
