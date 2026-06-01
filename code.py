import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from google.colab import files
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN,LSTM, Dense,Dropout
from tensorflow.keras.optimizers import Adam,SGD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

uploaded=files.upload()

df=pd.read_csv("all_movies_dataset.csv")

df.head()

df=df[["Title","Genres","Language"]]

df.index

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

df.index

df.head()

len(df['Language'].unique())

#count of each language
print(df['Language'].value_counts())

#remove rows of language where value count less than 100
language_counts = df['Language'].value_counts()
languages_to_keep = language_counts[language_counts >= 100].index
df = df[df['Language'].isin(languages_to_keep)]


print(df['Language'].value_counts())


lang_map = {
    "en": "English Hollywood",
    "fr": "French European",
    "es": "Spanish Latin",
    "tr": "Turkish Drama",
    "ja": "Japanese Anime"
}

df["Language"] = df["Language"].map(lang_map)

df["Genres"] = df["Genres"].str.replace(",", " ")

df["features"] = df["Genres"] + " " + df["Language"]

df["features"] = df["features"] + " " + df["Language"]

tfidf = TfidfVectorizer()
matrix = tfidf.fit_transform(df["features"])

similarity = cosine_similarity(matrix)

def recommend(movie_name, top_n=5):
    # check if movie exists
    if movie_name not in df['Title'].values:
        return "Movie not found"

    # get index of movie
    idx = df[df['Title'] == movie_name].index[0]

    # get similarity scores
    scores = list(enumerate(similarity[idx]))

    # sort by similarity
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # skip first (itself) and take top_n
    top_movies = scores[1:top_n+1]

    results = []

    for i in top_movies:
        movie_title = df.iloc[i[0]]['Title']
        score = round(i[1], 3)
        results.append((movie_title))

    return results

recommend("G20")
