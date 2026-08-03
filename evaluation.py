"""
evaluation.py
--------------
Compares three approaches on the SAME held-out test ratings, using RMSE
and MAE, so you can show a concrete accuracy improvement from combining
content-based and collaborative filtering rather than just asserting it.

Approach for making content-based filtering produce a *predicted rating*
(needed to be comparable to CF/hybrid on RMSE/MAE):
For a given (user, movie) pair, look at the movies THIS user has already
rated, weight those ratings by how content-similar each one is to the
target movie, and take the weighted average. This is a standard way to
turn a similarity matrix into a rating predictor.
"""

import numpy as np


def build_movie_id_index(new_df):
    """Maps TMDB movie_id -> row position in new_df/similarity matrix. Build once, reuse for every prediction."""
    return {mid: idx for idx, mid in zip(new_df.index, new_df['movie_id'])}


def content_based_predict_rating(user_id, tmdb_id, new_df, similarity,
                                  user_rated_lookup, id_to_index, k=10, global_mean=3.0):
    """
    Predicts what `user_id` would rate `tmdb_id`, using a similarity-weighted
    average of ratings this user already gave to other movies.

    user_rated_lookup: dict of {user_id: [(tmdb_id, rating), ...]} built once
                        from the training ratings (see build_user_rated_lookup).
    """
    target_idx = id_to_index.get(tmdb_id)
    user_history = user_rated_lookup.get(user_id)

    if target_idx is None or not user_history:
        return global_mean

    sims = similarity[target_idx]

    scored = []
    for rated_tmdb_id, rating in user_history:
        idx = id_to_index.get(rated_tmdb_id)
        if idx is not None:
            scored.append((sims[idx], rating))

    if not scored:
        return global_mean

    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = scored[:k]

    sim_sum = sum(sim for sim, _ in top_k)
    if sim_sum == 0:
        return np.mean([r for _, r in top_k])

    return sum(sim * r for sim, r in top_k) / sim_sum


def build_user_rated_lookup(train_ratings_df):
    """
    Builds {user_id: [(tmdb_id, rating), ...]} from the TRAINING split only —
    using test-set ratings here would leak the answer into the prediction.
    """
    lookup = {}
    for row in train_ratings_df.itertuples(index=False):
        lookup.setdefault(row.userId, []).append((row.tmdbId, row.rating))
    return lookup


def _rmse_mae(preds, actuals):
    preds = np.array(preds)
    actuals = np.array(actuals)
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mae = float(np.mean(np.abs(preds - actuals)))
    return rmse, mae


def evaluate_all(svd, testset, train_ratings_df, new_df, similarity, alpha=0.5, k=10):
    """
    Runs content-only, CF-only, and hybrid prediction over the same testset
    and returns a dict of {method: (rmse, mae)}.

    testset: the surprise testset from train_test_split() — list of (userId, tmdbId, actual_rating) tuples.
    train_ratings_df: the ratings dataframe used to FIT the model (not testset), needed
                       to build each user's rating history for content-based prediction.
    """
    global_mean = train_ratings_df['rating'].mean()
    id_to_index = build_movie_id_index(new_df)
    user_rated_lookup = build_user_rated_lookup(train_ratings_df)

    cb_preds, cf_preds, hybrid_preds, actuals = [], [], [], []

    for user_id, tmdb_id, actual in testset:
        cb_pred = content_based_predict_rating(
            user_id, tmdb_id, new_df, similarity, user_rated_lookup, id_to_index, k=k, global_mean=global_mean
        )
        cf_pred = svd.predict(user_id, tmdb_id).est
        hybrid_pred = alpha * cb_pred + (1 - alpha) * cf_pred

        cb_preds.append(cb_pred)
        cf_preds.append(cf_pred)
        hybrid_preds.append(hybrid_pred)
        actuals.append(actual)

    return {
        "content-based only": _rmse_mae(cb_preds, actuals),
        "collaborative only": _rmse_mae(cf_preds, actuals),
        f"hybrid (alpha={alpha})": _rmse_mae(hybrid_preds, actuals),
    }


def print_comparison_table(results):
    print(f"{'Method':<25}{'RMSE':>10}{'MAE':>10}")
    print("-" * 45)
    for method, (rmse, mae) in results.items():
        print(f"{method:<25}{rmse:>10.4f}{mae:>10.4f}")