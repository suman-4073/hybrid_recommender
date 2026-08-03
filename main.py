
from data_preprocessing import load_and_preprocess
from content_based import build_similarity_matrix, get_similar_movies
from collaborative_filtering import load_ratings, train_svd
from hybrid import hybrid_recommend
from evaluation import evaluate_all, print_comparison_table
from popularity import build_popularity_ranking, popularity_recommend

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
    svd, trainset, testset ,train_ratings_df= train_svd(ratings)

    print("\nHybrid recommendations for user 1, seeded on 'Avatar':")
    results, is_cold_start = hybrid_recommend(
            user_id=1, movie_title="Avatar",
            new_df=new_df, similarity=similarity, svd=svd, trainset=trainset, alpha=0.5,
        )
    if is_cold_start:
            print("  (user 1 has no rating history -> showing pure content-based results)")
    for title, score in results:
            print(f"  {title}  (score: {score:.3f})")
    
    # demonstrate cold-start fallback explicitly with a user id guaranteed not
    # to exist in the training data
    print("\nHybrid recommendations for a brand-new user (id=999999), seeded on 'Avatar':")
    results, is_cold_start = hybrid_recommend(
            user_id=999999, movie_title="Avatar",
            new_df=new_df, similarity=similarity, svd=svd, trainset=trainset, alpha=0.5,
        )
    if is_cold_start:
            print("  (cold start detected -> falling back to pure content-based results)")
    for title, score in results:
            print(f"  {title}  (score: {score:.3f})")
    
    print("\nEvaluating content-based-only vs collaborative-only vs hybrid on held-out test ratings...")
    comparison = evaluate_all(svd, testset,train_ratings_df, new_df, similarity, alpha=0.5)
    print_comparison_table(comparison)


     # true cold-start case: brand-new user, no seed movie either (e.g. a homepage,
    # not a "more like this" screen) -- content-based has nothing to anchor on here,
    # so popularity is the right fallback instead
    print("\nPopularity fallback for a brand-new user with no seed movie (e.g. homepage):")
    popularity_ranking = build_popularity_ranking(train_ratings_df)
    for title, score in popularity_recommend(new_df, popularity_ranking):
        print(f"  {title}  (weighted rating: {score:.3f})")
    


if __name__ == "__main__":
    main()