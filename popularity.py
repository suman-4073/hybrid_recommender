"""
popularity.py
--------------
Computes a popularity ranking from the ratings data itself, using an
IMDB-style weighted rating so movies with very few ratings don't
outrank well-established ones just from a lucky high score.

This is the cold-start fallback for when there's no seed movie AND
no user history to work with (e.g. a brand-new user's homepage) --
distinct from hybrid.py's cold-start fallback, which still has a seed
movie to stay relevant to.
"""

import pandas as pd


def build_popularity_ranking(ratings_df, m=None, m_percentile=0.90):
    """
    Args:
        ratings_df: dataframe with at least ['tmdbId', 'rating'] columns
        m: minimum rating count threshold. If None, computed automatically
           as the `m_percentile` percentile of rating counts (default: only
           movies at or above the 90th percentile of rating counts are
           treated as "fully trusted" -- fewer ratings get pulled toward C).

    Returns:
        DataFrame with columns [tmdbId, rating_count, avg_rating, weighted_rating],
        sorted by weighted_rating descending.
    """
    stats = ratings_df.groupby('tmdbId')['rating'].agg(['count', 'mean']).reset_index()
    stats.columns = ['tmdbId', 'rating_count', 'avg_rating']

    C = ratings_df['rating'].mean()
    if m is None:
        m = stats['rating_count'].quantile(m_percentile)

    v = stats['rating_count']
    R = stats['avg_rating']
    stats['weighted_rating'] = (v / (v + m)) * R + (m / (v + m)) * C

    return stats.sort_values('weighted_rating', ascending=False).reset_index(drop=True)


def popularity_recommend(new_df, popularity_ranking, top_n=5):
    """
    Returns the top_n most popular movies (by weighted rating), joined
    back to their titles via new_df. Used when there's no seed movie
    and no user history to personalize with at all.
    """
    top = popularity_ranking.head(top_n)
    merged = top.merge(new_df[['movie_id', 'title']], left_on='tmdbId', right_on='movie_id')
    return list(zip(merged['title'], merged['weighted_rating']))
