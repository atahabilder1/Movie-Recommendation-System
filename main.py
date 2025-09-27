#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Movie Recommendation System - Main Entry Point
===============================================

Unified interface for both content-based and collaborative filtering
recommendation approaches. Demonstrates the complete recommendation pipeline
from data loading to model training and recommendation generation.

Author: Portfolio Project for Data Science Applications
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import argparse
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import TMDBDataProcessor
from src.recommend import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('recommendation_system.log')
    ]
)
logger = logging.getLogger(__name__)


class MovieRecommendationSystem:
    """
    Unified movie recommendation system combining multiple approaches.

    Features:
    - Content-based recommendations using TF-IDF and cosine similarity
    - Collaborative filtering using SVD and Neural CF
    - Unified interface for both recommendation types
    - Performance comparison and evaluation
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the recommendation system."""
        self.config = config or self._default_config()
        self.content_recommender = None
        self.collaborative_recommender = None
        self.data_processor = None
        self.movies_df = None

    def _default_config(self) -> Dict:
        """Default configuration for the recommendation system."""
        return {
            # Data paths
            'raw_data_path': 'data/TMDB_movie_dataset_v11.csv',
            'processed_data_path': 'data/processed_movies.csv.gz',

            # Content-based config
            'content_based': {
                'max_features': 10000,
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.8,
                'use_svd': True,
                'svd_components': 300,
                'similarity_threshold': 0.1,
                'top_k': 10
            },

            # Collaborative filtering config
            'collaborative': {
                'svd_components': 50,
                'embedding_dim': 64,
                'hidden_dims': [128, 64],
                'learning_rate': 0.001,
                'batch_size': 1024,
                'epochs': 50,
                'min_user_ratings': 5,
                'min_movie_ratings': 3
            },

            # Demo settings
            'demo_mode': True,
            'demo_movies': [
                'The Dark Knight',
                'Inception',
                'Pulp Fiction',
                'The Matrix',
                'Forrest Gump'
            ]
        }

    def initialize_system(self) -> None:
        """Initialize all components of the recommendation system."""
        logger.info("Initializing Movie Recommendation System")
        logger.info("=" * 50)

        # Check if processed data exists
        if not os.path.exists(self.config['processed_data_path']):
            logger.info("Processed data not found. Processing raw data...")
            self._process_data()

        # Load processed data
        logger.info("Loading processed movie data...")
        self.movies_df = pd.read_csv(self.config['processed_data_path'], compression='gzip')
        logger.info(f"Loaded {len(self.movies_df)} movies")

        # Initialize content-based recommender
        logger.info("Initializing Content-Based Recommender...")
        self.content_recommender = ContentBasedRecommender(self.config['content_based'])
        self.content_recommender.load_data(self.config['processed_data_path'])

        # Initialize collaborative filtering recommender
        logger.info("Initializing Collaborative Filtering Recommender...")
        self.collaborative_recommender = CollaborativeFilteringRecommender(self.config['collaborative'])
        self.collaborative_recommender.load_data(self.config['processed_data_path'])

        logger.info("System initialized successfully!")

    def _process_data(self) -> None:
        """Process raw data if needed."""
        self.data_processor = TMDBDataProcessor()
        self.data_processor.load_data(self.config['raw_data_path'])
        self.data_processor.clean_data()
        self.data_processor.extract_features()
        self.data_processor.save_processed_data(self.config['processed_data_path'])

    def train_models(self, quick_demo: bool = False) -> None:
        """Train all recommendation models."""
        logger.info("Training Recommendation Models")
        logger.info("=" * 40)

        # Train content-based model
        logger.info("Training Content-Based Model...")
        self.content_recommender.build_tfidf_model()
        self.content_recommender.compute_similarity_matrix()
        logger.info("Content-based model trained!")

        # Train collaborative filtering models
        if not quick_demo:
            logger.info("Training Collaborative Filtering Models...")
            X_train, X_test, y_train, y_test = self.collaborative_recommender.prepare_collaborative_data()

            # Train SVD
            logger.info("Training SVD model...")
            self.collaborative_recommender.train_svd_model(X_train, y_train)

            # Train Neural CF
            logger.info("Training Neural CF model...")
            self.collaborative_recommender.train_ncf_model(X_train, X_test, y_train, y_test)

            logger.info("Collaborative filtering models trained!")
        else:
            logger.info("Quick demo mode - skipping collaborative training")

    def get_content_recommendations(self, movie_title: str, top_k: int = 5) -> List[Dict]:
        """Get content-based recommendations for a movie."""
        if not self.content_recommender:
            raise ValueError("Content recommender not initialized")

        try:
            recommendations = self.content_recommender.get_recommendations(movie_title, top_k)
            return recommendations
        except Exception as e:
            logger.error(f"Error getting content recommendations: {e}")
            return []

    def get_collaborative_recommendations(self, user_id: int = 0, top_k: int = 5, model_type: str = 'svd') -> List[Dict]:
        """Get collaborative filtering recommendations for a user."""
        if not self.collaborative_recommender:
            raise ValueError("Collaborative recommender not initialized")

        try:
            recommendations = self.collaborative_recommender.get_user_recommendations(user_id, top_k, model_type)
            return recommendations
        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {e}")
            return []

    def demonstrate_recommendations(self) -> None:
        """Demonstrate both recommendation approaches."""
        logger.info("Demonstration: Movie Recommendations")
        logger.info("=" * 40)

        # Content-based demonstrations
        logger.info("Content-Based Recommendations:")
        for movie in self.config['demo_movies']:
            try:
                recommendations = self.get_content_recommendations(movie, top_k=3)
                if recommendations:
                    logger.info(f"\nSimilar to '{movie}':")
                    for i, rec in enumerate(recommendations, 1):
                        similarity_score = rec.get('similarity_score', 0)
                        logger.info(f"  {i}. {rec['title']} (Score: {similarity_score:.3f})")
                else:
                    logger.info(f"\nNo recommendations found for '{movie}'")
            except Exception as e:
                logger.error(f"Error with movie '{movie}': {e}")

        # Collaborative filtering demonstration (if available)
        if hasattr(self.collaborative_recommender, 'svd_model') and self.collaborative_recommender.svd_model:
            logger.info("\nCollaborative Filtering Recommendations:")
            try:
                user_recs = self.get_collaborative_recommendations(user_id=0, top_k=5, model_type='svd')
                if user_recs:
                    logger.info(f"\nTop recommendations for User 0:")
                    for i, rec in enumerate(user_recs, 1):
                        pred_rating = rec.get('predicted_rating', 0)
                        logger.info(f"  {i}. {rec['title']} (Predicted Rating: {pred_rating:.1f})")
            except Exception as e:
                logger.error(f"Error with collaborative recommendations: {e}")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the recommendation system."""
        stats = {
            'total_movies': len(self.movies_df) if self.movies_df is not None else 0,
            'content_based_ready': self.content_recommender is not None,
            'collaborative_ready': self.collaborative_recommender is not None,
        }

        if self.movies_df is not None:
            stats.update({
                'unique_genres': len(self.movies_df['genres'].str.split('|').explode().unique()) if 'genres' in self.movies_df.columns else 0,
                'movies_with_overview': self.movies_df['overview'].notna().sum() if 'overview' in self.movies_df.columns else 0,
                'average_rating': self.movies_df['vote_average'].mean() if 'vote_average' in self.movies_df.columns else 0
            })

        return stats


def main():
    """Main entry point for the recommendation system."""
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('--quick-demo', action='store_true',
                       help='Run quick demo without training collaborative models')
    parser.add_argument('--movie', type=str,
                       help='Get recommendations for a specific movie')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of recommendations to return')

    args = parser.parse_args()

    try:
        # Initialize system
        rec_system = MovieRecommendationSystem()
        rec_system.initialize_system()

        # Train models
        rec_system.train_models(quick_demo=args.quick_demo)

        # Get system statistics
        stats = rec_system.get_system_stats()
        logger.info(f"\nSystem Statistics:")
        logger.info(f"Total Movies: {stats['total_movies']}")
        logger.info(f"Unique Genres: {stats.get('unique_genres', 'N/A')}")
        logger.info(f"Average Rating: {stats.get('average_rating', 0):.1f}")

        # Handle specific movie request
        if args.movie:
            logger.info(f"\nGetting recommendations for: {args.movie}")
            recommendations = rec_system.get_content_recommendations(args.movie, args.top_k)
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    similarity_score = rec.get('similarity_score', 0)
                    logger.info(f"{i}. {rec['title']} (Score: {similarity_score:.3f})")
            else:
                logger.info("No recommendations found for this movie.")
        else:
            # Run demonstration
            rec_system.demonstrate_recommendations()

        logger.info("\nRecommendation system completed successfully!")

    except Exception as e:
        logger.error(f"Error in recommendation system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()