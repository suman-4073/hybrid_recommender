
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_matrix(new_df, max_features=5000):
    """
    Vectorizes the 'tags' column and computes pairwise cosine similarity
    between every movie in new_df.

    Returns:
        similarity: (n_movies, n_movies) numpy array,
                    similarity[i][j] = similarity score between movie i and movie j
    """
    cv = CountVectorizer(max_features=max_features, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity


def get_similar_movies(movie_title, new_df, similarity, top_n=5):
    """
    Pure content-based recommendation: most similar movies by tag overlap,
    with no personalization for any specific user.
    """
    movie_index = new_df[new_df['title'] == movie_title].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n + 1]

    return [(new_df.iloc[idx].title, score) for idx, score in movies_list]
