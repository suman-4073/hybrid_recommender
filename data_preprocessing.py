import ast
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def convert(obj):
    """Turn a stringified list-of-dicts column (e.g. genres, keywords) into a list of names."""
    L = []
    if isinstance(obj, str):
        try:
            obj = ast.literal_eval(obj)
        except Exception:
            return []
    if isinstance(obj, list):
        for i in obj:
            if isinstance(i, dict):
                L.append(i['name'])
            else:
                L.append(i)
    return L


def convert_cast(obj, top_n=3):
    """Extract the top N cast member names."""
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != top_n:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


def fetch_director(obj):
    """Extract the director's name from the crew column."""
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


def stem(text):
    """Reduce each word in a string to its root form."""
    return " ".join(ps.stem(word) for word in text.split())


def load_and_preprocess(movies_path="data/tmdb_5000_movies.csv",
                         credits_path="data/tmdb_5000_credits.csv"):
    """
    Loads the raw TMDB csvs and returns new_df with columns:
    movie_id, title, tags (cleaned, stemmed, lowercase string)
    """
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    movies = movies.merge(credits, on='title')

    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    # remove spaces so "Sam Worthington" -> "SamWorthington" (keeps multi-word names as one token)
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    new_df['tags'] = new_df['tags'].apply(stem)

    return new_df