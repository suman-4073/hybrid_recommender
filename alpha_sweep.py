"""
alpha_sweep.py
----------------
Runs evaluate_all() across a range of alpha values to see whether ANY
blend of content-based + collaborative beats pure collaborative filtering
on RMSE/MAE, and if so, at what alpha.

Run with:
    python alpha_sweep.py
"""

from data_preprocessing import load_and_preprocess
from content_based import build_similarity_matrix
from collaborative_filtering import load_ratings, train_svd
from evaluation import evaluate_all, print_comparison_table


def main():
    print("Loading and preprocessing content-based data...")
    new_df = load_and_preprocess(
        movies_path="data/tmdb_5000_movies.csv",
        credits_path="data/tmdb_5000_credits.csv",
    )

    print("Building content similarity matrix...")
    similarity = build_similarity_matrix(new_df)

    print("Loading MovieLens ratings and training SVD...")
    ratings = load_ratings(
        ratings_path="data/ratings.csv",
        links_path="data/links.csv",
        valid_tmdb_ids=new_df['movie_id'],
    )
    svd, trainset, testset, train_ratings_df = train_svd(ratings, verbose=False)

    print("\nSweeping alpha values...\n")
    print(f"{'alpha':<8}{'RMSE':>10}{'MAE':>10}")
    print("-" * 28)

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        results = evaluate_all(svd, testset, train_ratings_df, new_df, similarity, alpha=alpha)
        rmse, mae = results[f"hybrid (alpha={alpha})"]
        print(f"{alpha:<8}{rmse:>10.4f}{mae:>10.4f}")

    print("\nFor reference:")
    print_comparison_table({
        "content-based only": evaluate_all(svd, testset, train_ratings_df, new_df, similarity, alpha=1.0)[f"hybrid (alpha=1.0)"],
        "collaborative only": evaluate_all(svd, testset, train_ratings_df, new_df, similarity, alpha=0.0)[f"hybrid (alpha=0.0)"],
    })


if __name__ == "__main__":
    main()