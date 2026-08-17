"""
app.py
-------
Streamlit demo for the hybrid movie recommender. Wires together the
existing modules (data_preprocessing, content_based, collaborative_filtering,
hybrid, popularity) exactly as they are -- no logic is duplicated here,
this file is purely UI + caching.

Run with:
    streamlit run app.py
"""

import streamlit as st

from data_preprocessing import load_and_preprocess
from content_based import build_similarity_matrix, get_similar_movies
from collaborative_filtering import load_ratings, train_svd
from hybrid import hybrid_recommend
from popularity import build_popularity_ranking, popularity_recommend


# ---------- cached loaders ----------
# st.cache_resource is for objects that shouldn't be copied/serialized (models,
# matrices used by reference). st.cache_data is for plain data (dataframes).
# Without these, Streamlit would rebuild the similarity matrix and retrain SVD
# on every single widget interaction, which would make the app unusably slow.

@st.cache_data(show_spinner="Loading and preprocessing movie data...")
def get_content_data():
    return load_and_preprocess(
        movies_path="data/tmdb_5000_movies.csv",
        credits_path="data/tmdb_5000_credits.csv",
    )


@st.cache_resource(show_spinner="Building content similarity matrix...")
def get_similarity(_new_df):
    # leading underscore tells Streamlit not to try hashing this dataframe arg
    return build_similarity_matrix(_new_df)


@st.cache_resource(show_spinner="Loading ratings and training SVD model...")
def get_cf_model(_valid_tmdb_ids):
    ratings = load_ratings(
        ratings_path="data/ratings.csv",
        links_path="data/links.csv",
        valid_tmdb_ids=_valid_tmdb_ids,
    )
    svd, trainset, testset, train_ratings_df = train_svd(ratings, verbose=False)
    return svd, trainset, testset, train_ratings_df


@st.cache_resource(show_spinner="Ranking popular movies...")
def get_popularity_ranking(_train_ratings_df):
    return build_popularity_ranking(_train_ratings_df)


# ---------- app ----------

st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Hybrid Movie Recommender")
st.caption("Content-based filtering + collaborative filtering (SVD), blended.")

new_df = get_content_data()
similarity = get_similarity(new_df)
svd, trainset, testset, train_ratings_df = get_cf_model(new_df['movie_id'])
popularity_ranking = get_popularity_ranking(train_ratings_df)

with st.sidebar:
    st.header("Settings")
    user_id = st.number_input("User ID", min_value=1, value=1, step=1,
                               help="Try a user ID from the MovieLens ratings.csv, or an unused ID (e.g. 999999) to see the cold-start fallback.")
    alpha = st.slider("Alpha (content vs. collaborative blend)", 0.0, 1.0, 0.5, 0.05,
                       help="1.0 = pure content-based, 0.0 = pure collaborative filtering")
    top_n = st.slider("Number of recommendations", 3, 15, 5)
    st.caption(f"alpha={alpha:.2f} → {alpha*100:.0f}% content similarity, {(1-alpha)*100:.0f}% predicted rating")

movie_title = st.selectbox("Pick a movie you like:", sorted(new_df['title'].tolist()))

if st.button("Get recommendations", type="primary"):
    tab1, tab2, tab3 = st.tabs(["Hybrid", "Content-based only", "Popular movies"])

    with tab1:
        results, is_cold_start = hybrid_recommend(
            user_id=user_id, movie_title=movie_title,
            new_df=new_df, similarity=similarity, svd=svd, trainset=trainset,
            top_n=top_n, alpha=alpha,
        )
        if is_cold_start:
            st.info(f"User {user_id} has no rating history — showing pure content-based results instead of a real hybrid blend.")
        for title, score in results:
            st.write(f"**{title}** — score: {score:.3f}")

    with tab2:
        st.caption("For comparison: what content-based alone (no personalization) would recommend.")
        for title, score in get_similar_movies(movie_title, new_df, similarity, top_n=top_n):
            st.write(f"**{title}** — similarity: {score:.3f}")

    with tab3:
        st.caption("Globally popular movies (IMDB-style weighted rating) — used when there's no seed movie and no user history at all.")
        for title, score in popularity_recommend(new_df, popularity_ranking, top_n=top_n):
            st.write(f"**{title}** — weighted rating: {score:.3f}")
else:
    st.info("Pick a movie and click **Get recommendations** to see the hybrid system in action.")