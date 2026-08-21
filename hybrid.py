"""
hybrid.py
----------
Combines content-based similarity and collaborative-filtering predicted
ratings into a single ranked recommendation list.

Strategy: cascade + weighted blend.
  1. Cascade: content-based similarity shortlists candidate movies (fast, cheap).
  2. Weighted: each candidate is re-scored using both its content similarity
     AND the user's predicted rating from SVD, blended by alpha.
"""


def hybrid_recommend(user_id, movie_title, new_df, similarity, svd,trainset=None,
                      shortlist_size=25, top_n=5, alpha=0.2):
    """
       Args:
           user_id: the user to personalize recommendations for
           movie_title: the "seed" movie the user is currently viewing/liked
           new_df: content-based dataframe (movie_id, title, tags)
           similarity: precomputed cosine similarity matrix from content_based.py
           svd: trained SVD model from collaborative_filtering.py
           trainset: the surprise trainset returned by train_svd(). Optional, but
                     needed to detect cold-start users. If omitted, cold-start
                     detection is skipped and CF is always used.
           shortlist_size: how many content-similar candidates to consider before re-ranking
           top_n: how many final recommendations to return
           alpha: weight given to content similarity vs. collaborative score (0-1).
                  alpha=1.0 -> pure content-based, alpha=0.0 -> pure collaborative
   
       Returns:
           (results, is_cold_start)
           results: list of (title, final_score) tuples, sorted best-first.
           is_cold_start: True if user_id had no ratings in the training data,
                           meaning this call fell back to pure content-based
                           ranking instead of a real hybrid blend.
       """
    movie_index = new_df[new_df['title'] == movie_title].index[0]
    content_scores = similarity[movie_index]

    # Step 1: cascade — shortlist candidates using content similarity only
    candidates = sorted(
        list(enumerate(content_scores)), reverse=True, key=lambda x: x[1]
    )[1:shortlist_size + 1]  # skip index 0, it's the movie itself
     # Cold-start check: if this user has no ratings in the training data,
    # svd.predict() doesn't fail — it silently returns the global mean
    # rating for everyone, which isn't personalization at all. Detecting
    # this explicitly and falling back to pure content-based similarity
    # is more honest than pretending a fake CF score is meaningful.
    is_cold_start = trainset is not None and not trainset.knows_user(user_id)

    # Step 2: weighted re-rank using collaborative signal
    results = []
    for idx, content_score in candidates:
        tmdb_id = new_df.iloc[idx]['movie_id']
        if is_cold_start:
                    final_score = content_score
        else:
            pred_rating = svd.predict(user_id, tmdb_id).est   # predicted rating on a 0.5-5 scale
            norm_cf = pred_rating / 5.0                        # normalize to 0-1 to match content_score's range

            final_score = alpha * content_score + (1 - alpha) * norm_cf
        results.append((new_df.iloc[idx]['title'], final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n], is_cold_start
