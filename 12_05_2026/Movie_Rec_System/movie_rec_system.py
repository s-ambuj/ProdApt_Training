import requests
from ast import literal_eval
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()

DATA_PATH = r"C:\Users\Administrator\Desktop\ProdApt_Training\12_05_2026\Movie_Rec_System\tmdb_5000_movies.csv"
CREDITS_PATH = r"C:\Users\Administrator\Desktop\ProdApt_Training\12_05_2026\Movie_Rec_System\tmdb_5000_credits.csv"


def parse_features(column):
    return column.astype(str).apply(literal_eval).apply(
        lambda items: ' '.join([item['name'].replace(' ', '') for item in items])
    )


def parse_cast(cast_series):
    def extract_names(cast_str):
        try:
            items = literal_eval(cast_str)
        except Exception:
            return ''
        names = [item.get('name', '').replace(' ', '') for item in items[:6] if item.get('name')]
        return ' '.join(names)
    return cast_series.astype(str).apply(extract_names)


def parse_crew(crew_series):
    important_jobs = {'Director', 'Screenplay', 'Writer', 'Producer'}
    def extract_names(crew_str):
        try:
            items = literal_eval(crew_str)
        except Exception:
            return ''
        names = [item.get('name', '').replace(' ', '') for item in items if item.get('job') in important_jobs and item.get('name')]
        return ' '.join(names[:6])
    return crew_series.astype(str).apply(extract_names)


def load_movies(path, credits_path):
    movies = pd.read_csv(path)
    credits = pd.read_csv(credits_path)
    credits = credits[['movie_id', 'cast', 'crew']].copy()
    credits['cast'] = parse_cast(credits['cast'])
    credits['crew'] = parse_crew(credits['crew'])
    movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
    movies = movies[['id', 'title', 'genres', 'keywords', 'overview', 'release_date', 'vote_average', 'cast', 'crew']].copy()
    movies['genres'] = parse_features(movies['genres'])
    movies['keywords'] = parse_features(movies['keywords'])
    movies['overview'] = movies['overview'].fillna('')
    movies['cast'] = movies['cast'].fillna('')
    movies['crew'] = movies['crew'].fillna('')
    movies['soup'] = (
        movies['genres'] + ' ' + movies['keywords'] + ' ' + movies['overview'] + ' ' + movies['cast'] + ' ' + movies['crew']
    )
    return movies


def build_similarity_matrix(movies):
    vectorizer = CountVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(movies['soup'])
    return cosine_similarity(matrix, matrix)


def get_recommendations(title, movies, similarity_matrix, count=5):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    if title not in indices:
        return pd.DataFrame([], columns=movies.columns)
    idx = indices[title]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: count + 1]
    recommended_indices = [i[0] for i in sim_scores]
    return movies.iloc[recommended_indices]


def get_movie_details(movie_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    response = requests.get(url, params={"api_key": api_key, "language": "en-US"})
    return response.json() if response.ok else {}


def make_poster_url(poster_path):
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return "https://via.placeholder.com/500x750?text=No+Poster"


@st.cache_data
def cached_load_movies(path, credits_path):
    return load_movies(path, credits_path)


@st.cache_data
def cached_build_similarity_matrix(movies):
    return build_similarity_matrix(movies)


def main():
    st.title("Simple Movie Recommendation System")
    st.write("Pick a movie and view the top 5 recommended titles with poster cards and details fetched from TMDB.")

    api_key = os.getenv("TMDB_API_KEY", "")
    if not api_key:
        st.sidebar.warning("Enter your TMDB API key to fetch posters and movie details.")
        st.stop()

    movies = cached_load_movies(DATA_PATH, CREDITS_PATH)
    similarity_matrix = cached_build_similarity_matrix(movies)
    movie_query = st.text_input("Enter a movie title to get recommendations")
    num_recs = st.sidebar.slider("Number of recommendations", 1, 10, 5)

    if movie_query:
        titles = movies['title'].str.lower()
        query_lower = movie_query.strip().lower()
        exact_matches = movies[titles == query_lower]
        if not exact_matches.empty:
            selected_title = exact_matches.iloc[0]['title']
        else:
            partial_matches = movies[titles.str.contains(query_lower, na=False)]
            if not partial_matches.empty:
                selected_title = partial_matches.iloc[0]['title']
                st.info(f"Showing recommendations for closest match: {selected_title}")
            else:
                st.warning("Movie not found. Try a different title or spelling.")
                return

        recommendations = get_recommendations(selected_title, movies, similarity_matrix, num_recs)
        if recommendations.empty:
            st.warning("No recommendations found for this movie. Try another title.")
        else:
            st.write(f"Recommendations for **{selected_title}**")
            for _, movie in recommendations.iterrows():
                details = get_movie_details(movie['id'], api_key)
                poster_url = make_poster_url(details.get('poster_path'))
                cols = st.columns([1, 2])
                with cols[0]:
                    st.image(poster_url, width=250)
                with cols[1]:
                    st.subheader(movie['title'])
                    st.write(f"**Release date:** {movie.get('release_date', 'N/A')}")
                    st.write(f"**TMDB rating:** {details.get('vote_average', movie.get('vote_average', 'N/A'))}")
                    st.write(f"**Genres:** {movie['genres']}")
                    st.write(f"**Cast:** {movie['cast']}")
                    st.write(f"**Crew:** {movie['crew']}")
                    st.write(details.get('overview', movie['overview']))
                    if details.get('homepage'):
                        st.write(f"[Official page]({details['homepage']})")


if __name__ == "__main__":
    main()
