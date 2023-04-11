import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import json
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import contractions
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.stem import PorterStemmer
import re
import spacy
from heapq import nlargest
import openai
import os
from dotenv import load_dotenv
load_dotenv()
import random
import ast

# Set the seed for Python's built-in random module
random.seed(42)

# Set the seed for NumPy's random number generator
np.random.seed(42)

api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key
tmdb_api_key = os.environ.get("TMDB_API_KEY")

global df
df = None

# Define stopwords
stop_words = set(stopwords.words('english'))

# Load the small English NER model
nlp = spacy.load("en_core_web_sm")

# Load the embeddings from the binary file
embeddings = np.load('ada_embeddings_movie_40000.npy', allow_pickle=True)

# Load the rest of the DataFrame from the CSV file
ada_40000_min_df = pd.read_csv('ada_40000_min_streamlit.csv')

# Add the embeddings back to the DataFrame
ada_40000_min_df['ada_embeddings'] = pd.Series(embeddings)

# Assign the DataFrame to the global variable
df = ada_40000_min_df

# Define the function to preprocess text and remove people names
def preprocess_text(text):
    
    # Expand contractions
    text = contractions.fix(text)
    
    # Remove people names
    doc = nlp(text)
    no_name_text = [token.text for token in doc if not token.ent_type_ == 'PERSON']
    
    # Join the text
    no_name_text = ' '.join(no_name_text)
    
    # Remove non-alphabetical characters
    no_name_text = re.sub(r'[^a-zA-Z\s]', '', no_name_text)
    
    # Lowercase the text
    no_name_text = no_name_text.lower()
    
    # Split the text into words
    words = no_name_text.split()
    
    # Remove stopwords
    no_stopword_text = [w for w in words if not w in stop_words]
    
    # Apply stemming to each word
    stemmed_text = [PorterStemmer().stem(word) for word in no_stopword_text]
    
    return ' '.join(stemmed_text)

def ada_embeddings(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text,
    )
    return np.array(response['data'][0]['embedding'])

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    if len(union) == 0:
        return 0
    else:
        return len(intersection) / len(union)

def most_similar_movies(user_input, df, n=5, cosine_weight=0.70, jaccard_weight=0.05, genre_weight=0.25):
    input_embeddings = ada_embeddings(user_input)

    # Preprocess the user input
    user_input_preprocessed = preprocess_text(user_input)
    user_keywords = set(user_input_preprocessed.split())

    # Unique genres in the dataset
    unique_genres = set(sum([ast.literal_eval(genres) for genres in df['genres'].tolist()], []))

    # Find the matched genres
    matched_genres = unique_genres.intersection(user_keywords)

    similarities = []

    for index, row in df.iterrows():
        cur_embeddings = row['ada_embeddings']
        cosine_sim = cosine_similarity(input_embeddings.reshape(1, -1), cur_embeddings.reshape(1, -1))[0, 0]

        # Use preprocessed movie description
        movie_keywords = set(row['clean_overview'].split())

        jaccard_sim_keywords = jaccard_similarity(user_keywords, movie_keywords)

        # Calculate Jaccard similarity for genres
        movie_genres = set(row['genres'])
        jaccard_sim_genres = jaccard_similarity(matched_genres, movie_genres)

        # Calculate the weighted average of cosine similarity, Jaccard similarity (keywords), and Jaccard similarity (genres)
        similarity = cosine_weight * cosine_sim + jaccard_weight * jaccard_sim_keywords + genre_weight * jaccard_sim_genres
        similarities.append((similarity, index))

    top_n_similarities = nlargest(n, similarities)

    top_n_indices = [index for similarity, index in top_n_similarities]

    return df.loc[top_n_indices]

def fetch_poster(imdb_id, tmdb_api_key):
    # Use the TMDB API to get the movie details by IMDb ID
    tmdb_url = f"https://api.themoviedb.org/3/movie/{imdb_id}?api_key={tmdb_api_key}"
    response = requests.get(tmdb_url)
    if response.status_code == 200:
        data = json.loads(response.text)
        # Extract the poster path from the TMDB API response
        poster_path = data.get("poster_path")
        if poster_path:
            # Use the TMDB image base URL to construct the full poster URL
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

def main():
    custom_css = """
    <style>
        .title {
            color: deepskyblue;
            font-size: 2.5em;
            font-weight: bold;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.markdown("<div class='title'>ReelWhisperer</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        user_input = st.text_area("Tell me about the movie plot you're in the mood for!", "")

    with col2:
        st.image("streamlit.png", width=400)

    if st.button("Discover Movies"):
        if user_input:
            recommendations = most_similar_movies(user_input, df, n=5)
            if recommendations.empty:
                st.write("No movies found. Please try a different description.")
            else:
                st.write("Top 5 Movie Recommendations:")
                for index, movie in recommendations.iterrows():
                    st.write(f"{movie['title']} - {movie['overview']}")
                    poster_url = fetch_poster(movie['imdb_id'], os.environ.get("TMDB_API_KEY"))
                    if poster_url:
                        st.image(poster_url, width=200)
                    else:
                        st.write("Poster not available.")
        else:
            st.write("Please enter a few sentences to get movie recommendations.")

if __name__ == "__main__":
    main()
