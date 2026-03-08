import streamlit as st
import joblib
import pandas as pd

cosine_sim = joblib.load("similarity.joblib")
movies_dataset = joblib.load("movies.joblib")
indices = pd.Series(
    movies_dataset.index,
    index=movies_dataset['title']
).drop_duplicates()



def recommend(movie_title, n=5):

    idx = indices[movie_title]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:n+1]
    movie_indices = [i[0] for i in similarity_scores]

    return movies_dataset['title'].iloc[movie_indices]


st.title("🎬 Movie Recommendation System")

st.write("Select a movie to get recommendations.")

movie_list = movies_dataset['title'].values

selected_movie = st.selectbox(
    "Choose a movie",
    movie_list
)

if st.button("Recommend"):

    recommendations = recommend(selected_movie)
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write(movie)