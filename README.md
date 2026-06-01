# Movie Recommendation System

A content-based movie recommendation system that suggests similar movies using Natural Language Processing (NLP) techniques. The system analyzes movie genres and language information, converts them into numerical representations using TF-IDF vectorization, and recommends movies based on cosine similarity.

---

## Overview

Recommendation systems are widely used by platforms such as Netflix, Amazon, Spotify, and YouTube to provide personalized suggestions.

This project implements a content-based recommendation approach that recommends movies with similar characteristics based on:

- Movie Genres
- Movie Language

The system computes similarity scores between movies and returns the most relevant recommendations.

---

## Features

- Data Cleaning and Preprocessing
- Genre-Based Recommendation
- Language-Based Recommendation
- TF-IDF Feature Extraction
- Cosine Similarity Calculation
- Top-N Movie Recommendations
- Content-Based Filtering

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## Dataset

The dataset contains information about movies including:

- Movie Title
- Genres
- Language

### Sample Attributes

| Feature | Description |
|----------|-------------|
| Title | Name of the movie |
| Genres | Categories of the movie |
| Language | Primary language |

---

## Workflow

### 1. Data Preprocessing

- Remove missing values
- Remove duplicate records
- Filter languages with insufficient samples
- Standardize language categories

### 2. Feature Engineering

Genres and language information are combined into a single feature representation.

Example:

Action Adventure English Hollywood

---

### 3. TF-IDF Vectorization

Text features are converted into numerical vectors using TF-IDF.

```python
TfidfVectorizer()
```

---

### 4. Similarity Computation

Cosine similarity is calculated between movie vectors.

```python
cosine_similarity()
```

---

### 5. Recommendation Generation

Given a movie title:

```python
recommend("G20")
```

The system returns the most similar movies.

---

## Machine Learning Concepts Used

- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Cosine Similarity
- Information Retrieval
- Recommendation Systems

---

## Project Structure

```text
movie-recommendation-system/
│
├── movie_recommendation.ipynb
├── all_movies_dataset.csv
└── README.md
```

---

## Future Improvements

- Collaborative Filtering
- Hybrid Recommendation Systems
- User Rating Integration
- Deep Learning-Based Recommendations
- Streamlit Web Interface
- Personalized User Profiles

---

## Learning Outcomes

Through this project, I learned:

- Recommendation System Design
- Feature Engineering for Text Data
- Similarity-Based Retrieval
- Content-Based Filtering
- TF-IDF Representation
- Cosine Similarity Analysis

---

## Author

Dhananjay Divekar
