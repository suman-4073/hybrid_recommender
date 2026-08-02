"""
collaborative_filtering.py
----------------------------
Loads MovieLens ratings, links them to the TMDB movie ids used by the
content-based dataset, and trains an SVD model that predicts how a
specific user would rate a specific movie.
"""

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate


def load_ratings(ratings_path="data/ratings.csv",
                  links_path="data/links.csv",
                  valid_tmdb_ids=None):
    """
    Loads MovieLens ratings and maps MovieLens movieId -> TMDB movie_id
    using links.csv, so ratings can be joined against the content-based
    dataset (which is keyed by TMDB id).

    valid_tmdb_ids: optional iterable restricting ratings to movies that
                    also exist in the content-based dataset (new_df['movie_id']).
    """
    ratings = pd.read_csv(ratings_path)   # userId, movieId, rating, timestamp
    links = pd.read_csv(links_path)       # movieId, imdbId, tmdbId

    ratings = ratings.merge(links, on='movieId')
    ratings = ratings.dropna(subset=['tmdbId'])
    ratings['tmdbId'] = ratings['tmdbId'].astype(int)

    if valid_tmdb_ids is not None:
        ratings = ratings[ratings['tmdbId'].isin(valid_tmdb_ids)]

    return ratings


def train_svd(ratings, rating_scale=(0.5, 5.0), n_factors=50, test_size=0.2, random_state=42, verbose=True):
    """
    Trains an SVD matrix-factorization model on (userId, tmdbId, rating) triples.

    Returns:
        svd: trained model, use svd.predict(user_id, tmdb_id).est to get a predicted rating
        trainset, testset: for further evaluation if needed
    """
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(ratings[['userId', 'tmdbId', 'rating']], reader)

    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)

    svd = SVD(n_factors=n_factors, random_state=random_state)
    svd.fit(trainset)

    if verbose:
        cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    return svd, trainset, testset
