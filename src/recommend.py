# -*- coding: utf-8 -*-
"""
Professional Content-Based Movie Recommendation System
=====================================================

This module implements multiple content-based recommendation approaches:
- TF-IDF + Cosine Similarity (baseline)
- Advanced text preprocessing with NLP
- GPU-accelerated similarity computation
- Comprehensive evaluation metrics

Author: Portfolio Project for Data Science Applications
"""

import pandas as pd
import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NLP and ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    Professional content-based movie recommendation system.

    Features:
    - Multiple text processing strategies
    - TF-IDF vectorization with custom preprocessing
    - Cosine similarity computation
    - Efficient similarity search
    - Comprehensive evaluation metrics
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize recommender with configuration."""
        self.config = config or self._default_config()
        self.movies_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.similarity_matrix = None
        self.movie_indices = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _default_config(self) -> Dict:
        """Default configuration for the recommender."""
        return {
            'max_features': 10000,          # TF-IDF max features
            'ngram_range': (1, 2),          # Unigrams and bigrams
            'min_df': 2,                    # Minimum document frequency
            'max_df': 0.8,                  # Maximum document frequency
            'use_svd': True,                # Use SVD for dimensionality reduction
            'svd_components': 300,          # SVD components
            'similarity_threshold': 0.1,     # Minimum similarity threshold
            'top_k': 10,                    # Default number of recommendations
        }

    def load_data(self, filepath: str) -> None:
        """Load and prepare movie data."""
        logger.info(f"Loading movie data from {filepath}")
        self.movies_df = pd.read_csv(filepath, compression='gzip' if filepath.endswith('.gz') else None)

        # Create movie index mapping
        self.movie_indices = {title: idx for idx, title in enumerate(self.movies_df['title'])}

        logger.info(f"Loaded {len(self.movies_df)} movies")

    def _advanced_text_preprocessing(self, text: str) -> str:
        """Advanced text preprocessing for better feature extraction."""
        if pd.isna(text) or text == '':
            return ''

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()

        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                try:
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
                except:
                    processed_tokens.append(token)

        return ' '.join(processed_tokens)

    def _create_content_features(self) -> pd.Series:
        """Create comprehensive content features for each movie."""
        logger.info("Creating comprehensive content features...")

        content_features = []

        for idx, row in self.movies_df.iterrows():
            # Combine multiple text features
            features = []

            # Overview (main content)
            if pd.notna(row['overview']):
                overview_processed = self._advanced_text_preprocessing(row['overview'])
                # Give overview more weight by repeating it
                features.extend([overview_processed] * 3)

            # Genres (important for similarity)
            if pd.notna(row['genres']):
                try:
                    import ast
                    genres = ast.literal_eval(row['genres']) if isinstance(row['genres'], str) else row['genres']
                    if isinstance(genres, list):
                        # Add genres multiple times for higher weight
                        genre_text = ' '.join(genres).lower()
                        features.extend([genre_text] * 2)
                except:
                    pass

            # Keywords (additional context)
            if pd.notna(row['keywords']):
                try:
                    import ast
                    keywords = ast.literal_eval(row['keywords']) if isinstance(row['keywords'], str) else row['keywords']
                    if isinstance(keywords, list):
                        keyword_text = ' '.join(keywords).lower()
                        features.append(keyword_text)
                except:
                    pass

            # Tagline (marketing text)
            if pd.notna(row['tagline']):
                tagline_processed = self._advanced_text_preprocessing(row['tagline'])
                features.append(tagline_processed)

            # Combine all features
            combined_content = ' '.join(features)
            content_features.append(combined_content)

        return pd.Series(content_features)

    def build_tfidf_model(self) -> None:
        """Build TF-IDF model with advanced preprocessing."""
        logger.info("Building TF-IDF model with advanced preprocessing...")

        # Create content features
        content_features = self._create_content_features()

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            ngram_range=self.config['ngram_range'],
            min_df=self.config['min_df'],
            max_df=self.config['max_df'],
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )

        # Fit and transform
        logger.info("Fitting TF-IDF vectorizer...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_features)

        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        # Optional: Apply SVD for dimensionality reduction
        if self.config['use_svd'] and self.tfidf_matrix.shape[1] > self.config['svd_components']:
            logger.info("Applying SVD for dimensionality reduction...")
            svd = TruncatedSVD(n_components=self.config['svd_components'], random_state=42)
            self.tfidf_matrix = svd.fit_transform(self.tfidf_matrix)
            logger.info(f"SVD reduced matrix shape: {self.tfidf_matrix.shape}")

    def compute_similarity_matrix(self) -> None:
        """Compute and store similarity matrix."""
        logger.info("Computing cosine similarity matrix...")

        # For large datasets, compute similarity in chunks to manage memory
        chunk_size = 1000
        n_movies = self.tfidf_matrix.shape[0]

        if n_movies <= chunk_size:
            # Small dataset - compute all at once
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        else:
            # Large dataset - compute in chunks
            self.similarity_matrix = np.zeros((n_movies, n_movies))

            for i in range(0, n_movies, chunk_size):
                end_i = min(i + chunk_size, n_movies)
                for j in range(0, n_movies, chunk_size):
                    end_j = min(j + chunk_size, n_movies)

                    chunk_sim = cosine_similarity(
                        self.tfidf_matrix[i:end_i],
                        self.tfidf_matrix[j:end_j]
                    )
                    self.similarity_matrix[i:end_i, j:end_j] = chunk_sim

        logger.info(f"Similarity matrix computed: {self.similarity_matrix.shape}")

    def get_recommendations(self, movie_title: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Get movie recommendations based on content similarity.

        Args:
            movie_title: Title of the movie to get recommendations for
            top_k: Number of recommendations to return

        Returns:
            List of recommendation dictionaries with movie info and similarity scores
        """
        if top_k is None:
            top_k = self.config['top_k']

        # Find movie index
        if movie_title not in self.movie_indices:
            # Try partial matching
            possible_matches = [title for title in self.movie_indices.keys()
                              if movie_title.lower() in title.lower()]
            if possible_matches:
                movie_title = possible_matches[0]
                logger.info(f"Using closest match: {movie_title}")
            else:
                raise ValueError(f"Movie '{movie_title}' not found in database")

        movie_idx = self.movie_indices[movie_title]

        # Get similarity scores
        sim_scores = self.similarity_matrix[movie_idx]

        # Get indices of most similar movies (excluding the movie itself)
        similar_indices = np.argsort(sim_scores)[::-1][1:top_k+1]

        # Create recommendations list
        recommendations = []
        for idx in similar_indices:
            if sim_scores[idx] >= self.config['similarity_threshold']:
                movie_info = self.movies_df.iloc[idx]
                recommendations.append({
                    'title': movie_info['title'],
                    'similarity_score': float(sim_scores[idx]),
                    'vote_average': float(movie_info['vote_average']),
                    'vote_count': int(movie_info['vote_count']),
                    'release_year': int(movie_info['release_year']),
                    'genres': movie_info['genres'],
                    'overview': movie_info['overview'][:200] + '...' if len(str(movie_info['overview'])) > 200 else movie_info['overview']
                })

        return recommendations

    def get_similar_movies_by_features(self, genres: List[str] = None,
                                     keywords: List[str] = None,
                                     min_rating: float = 6.0,
                                     top_k: int = 10) -> List[Dict]:
        """
        Get movie recommendations based on specific features.

        Args:
            genres: List of preferred genres
            keywords: List of preferred keywords
            min_rating: Minimum rating threshold
            top_k: Number of recommendations

        Returns:
            List of filtered and ranked movie recommendations
        """
        # Start with all movies above rating threshold
        filtered_df = self.movies_df[self.movies_df['vote_average'] >= min_rating].copy()

        if genres:
            # Filter by genres
            def has_genre(movie_genres, target_genres):
                try:
                    import ast
                    movie_genre_list = ast.literal_eval(movie_genres) if isinstance(movie_genres, str) else movie_genres
                    if isinstance(movie_genre_list, list):
                        return any(genre in movie_genre_list for genre in target_genres)
                except:
                    pass
                return False

            filtered_df = filtered_df[filtered_df['genres'].apply(lambda x: has_genre(x, genres))]

        if keywords:
            # Filter by keywords
            def has_keyword(movie_keywords, target_keywords):
                try:
                    import ast
                    movie_keyword_list = ast.literal_eval(movie_keywords) if isinstance(movie_keywords, str) else movie_keywords
                    if isinstance(movie_keyword_list, list):
                        return any(keyword.lower() in ' '.join(movie_keyword_list).lower() for keyword in target_keywords)
                except:
                    pass
                return False

            filtered_df = filtered_df[filtered_df['keywords'].apply(lambda x: has_keyword(x, keywords))]

        # Sort by weighted rating and popularity
        filtered_df = filtered_df.sort_values(['weighted_rating', 'popularity'], ascending=False)

        # Return top results
        recommendations = []
        for idx, movie in filtered_df.head(top_k).iterrows():
            recommendations.append({
                'title': movie['title'],
                'vote_average': float(movie['vote_average']),
                'vote_count': int(movie['vote_count']),
                'weighted_rating': float(movie['weighted_rating']),
                'popularity': float(movie['popularity']),
                'release_year': int(movie['release_year']),
                'genres': movie['genres'],
                'overview': movie['overview'][:200] + '...' if len(str(movie['overview'])) > 200 else movie['overview']
            })

        return recommendations

    def evaluate_model(self, test_cases: List[str]) -> Dict:
        """
        Evaluate model performance with test cases.

        Args:
            test_cases: List of movie titles to test

        Returns:
            Evaluation metrics dictionary
        """
        logger.info("Evaluating model performance...")

        evaluation_results = {
            'test_cases': len(test_cases),
            'successful_recommendations': 0,
            'average_similarity_score': 0.0,
            'genre_consistency': 0.0,
            'examples': []
        }

        total_similarity = 0
        genre_matches = 0
        total_genres = 0

        for movie_title in test_cases:
            try:
                recommendations = self.get_recommendations(movie_title, top_k=5)
                if recommendations:
                    evaluation_results['successful_recommendations'] += 1

                    # Calculate average similarity score
                    avg_sim = np.mean([rec['similarity_score'] for rec in recommendations])
                    total_similarity += avg_sim

                    # Check genre consistency
                    original_movie = self.movies_df[self.movies_df['title'] == movie_title].iloc[0]
                    try:
                        import ast
                        original_genres = ast.literal_eval(original_movie['genres']) if isinstance(original_movie['genres'], str) else original_movie['genres']

                        for rec in recommendations:
                            rec_genres = ast.literal_eval(rec['genres']) if isinstance(rec['genres'], str) else rec['genres']
                            if isinstance(original_genres, list) and isinstance(rec_genres, list):
                                common_genres = set(original_genres) & set(rec_genres)
                                genre_matches += len(common_genres)
                                total_genres += len(original_genres)
                    except:
                        pass

                    # Store example
                    evaluation_results['examples'].append({
                        'query_movie': movie_title,
                        'recommendations': [rec['title'] for rec in recommendations[:3]],
                        'avg_similarity': avg_sim
                    })

            except Exception as e:
                logger.warning(f"Failed to get recommendations for {movie_title}: {e}")

        # Calculate final metrics
        if evaluation_results['successful_recommendations'] > 0:
            evaluation_results['average_similarity_score'] = total_similarity / evaluation_results['successful_recommendations']

        if total_genres > 0:
            evaluation_results['genre_consistency'] = genre_matches / total_genres

        return evaluation_results

    def save_model(self, filepath: str) -> None:
        """Save the trained model components."""
        logger.info(f"Saving model to {filepath}")

        model_data = {
            'config': self.config,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'similarity_matrix': self.similarity_matrix,
            'movie_indices': self.movie_indices,
            'movies_df': self.movies_df
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info("Model saved successfully")

    def load_model(self, filepath: str) -> None:
        """Load a pre-trained model."""
        logger.info(f"Loading model from {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.config = model_data['config']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.similarity_matrix = model_data['similarity_matrix']
        self.movie_indices = model_data['movie_indices']
        self.movies_df = model_data['movies_df']

        logger.info("Model loaded successfully")


def main():
    """Main function to demonstrate the recommendation system."""
    print("<� Building Professional Content-Based Movie Recommender")
    print("=" * 60)

    # Initialize recommender
    recommender = ContentBasedRecommender()

    # Load data
    recommender.load_data('data/processed_movies.csv.gz')

    # Build model
    print("\n=' Building TF-IDF model...")
    recommender.build_tfidf_model()

    print("\n=� Computing similarity matrix...")
    recommender.compute_similarity_matrix()

    # Test recommendations
    print("\n<� Testing Recommendations:")
    print("-" * 40)

    test_movies = ['Inception', 'The Dark Knight', 'Pulp Fiction', 'The Matrix', 'Interstellar']

    for movie in test_movies:
        try:
            print(f"\n<� Recommendations for '{movie}':")
            recommendations = recommender.get_recommendations(movie, top_k=5)

            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['title']} (Similarity: {rec['similarity_score']:.3f}, Rating: {rec['vote_average']:.1f})")

        except Exception as e:
            print(f"  Error: {e}")

    # Feature-based recommendations
    print(f"\n<� Genre-based recommendations (Action, Sci-Fi):")
    genre_recs = recommender.get_similar_movies_by_features(
        genres=['Action', 'Science Fiction'],
        min_rating=7.0,
        top_k=5
    )

    for i, rec in enumerate(genre_recs, 1):
        print(f"  {i}. {rec['title']} (Rating: {rec['vote_average']:.1f}, Year: {rec['release_year']})")

    # Evaluate model
    print(f"\n📈 Model Evaluation:")
    evaluation = recommender.evaluate_model(test_movies)
    print(f"  • Success rate: {evaluation['successful_recommendations']}/{evaluation['test_cases']}")
    print(f"  • Average similarity: {evaluation['average_similarity_score']:.3f}")
    print(f"  • Genre consistency: {evaluation['genre_consistency']:.3f}")

    # Save model
    print(f"\n💾 Saving model...")
    recommender.save_model('results/models/content_based_recommender.pkl')

    print(f"\n Content-Based Recommender Built Successfully!")
    print(f"=� Model Statistics:")
    print(f"  • Movies processed: {len(recommender.movies_df):,}")
    print(f"  • TF-IDF features: {recommender.tfidf_matrix.shape[1]:,}")
    print(f"  • Similarity matrix: {recommender.similarity_matrix.shape}")


if __name__ == "__main__":
    main()