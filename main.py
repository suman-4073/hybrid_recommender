
from data_preprocessing import load_and_preprocess
from content_based import build_similarity_matrix, get_similar_movies
from collaborative_filtering import load_ratings, train_svd
from hybrid import hybrid_recommend


def main():
    print("Loading and preprocessing content-based data...")
    new_df = load_and_preprocess(
        movies_path="data/tmdb_5000_movies.csv",
        credits_path="data/tmdb_5000_credits.csv",
    )

    print("Building content similarity matrix...")
    similarity = build_similarity_matrix(new_df)

    print("\nPure content-based recommendations for 'Avatar':")
    for title, score in get_similar_movies("Avatar", new_df, similarity):
        print(f"  {title}  (similarity: {score:.3f})")

    print("\nLoading MovieLens ratings and training SVD...")
    ratings = load_ratings(
        ratings_path="data/ratings.csv",
        links_path="data/links.csv",
        valid_tmdb_ids=new_df['movie_id'],
    )
    svd, trainset, testset = train_svd(ratings)

    print("\nHybrid recommendations for user 1, seeded on 'Avatar':")
    for title, score in hybrid_recommend(
        user_id=1, movie_title="Avatar",
        new_df=new_df, similarity=similarity, svd=svd, alpha=0.5,
    ):
        print(f"  {title}  (hybrid score: {score:.3f})")


if __name__ == "__main__":
    main()