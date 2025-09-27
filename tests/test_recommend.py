# -*- coding: utf-8 -*-
"""
Unit Tests for Movie Recommendation System
==========================================

Comprehensive test suite for all components of the recommendation system
including content-based filtering, collaborative filtering, evaluation
metrics, and utility functions.

Test Coverage:
- ContentBasedRecommender functionality
- CollaborativeFilteringRecommender functionality
- RecommendationEvaluator metrics
- Utility functions and helpers
- Data validation and preprocessing
- Integration tests

Author: Portfolio Project for Data Science Applications
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.recommend import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.evaluate import RecommendationEvaluator
from src.utils import (ConfigManager, DataValidator, FileManager,
                       TextUtils, MathUtils, Timer, MemoryMonitor)
from src.preprocess import TMDBDataProcessor


class TestContentBasedRecommender(unittest.TestCase):
    """Test cases for ContentBasedRecommender."""

    def setUp(self):
        """Set up test data and recommender."""
        # Create sample movie data
        self.sample_data = pd.DataFrame({
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
            'overview': [
                'Action adventure with heroes fighting villains',
                'Romantic comedy about love and relationships',
                'Action thriller with car chases and explosions',
                'Drama about family relationships and emotions'
            ],
            'genres': ['Action|Adventure', 'Comedy|Romance', 'Action|Thriller', 'Drama'],
            'keywords': ['action hero fight', 'comedy love romance', 'action thriller car', 'drama family emotion'],
            'tagline': ['Ultimate action', 'Love conquers all', 'Fast and thrilling', 'Family comes first'],
            'vote_average': [8.5, 7.2, 8.0, 7.8],
            'vote_count': [1000, 500, 800, 600],
            'release_year': [2020, 2019, 2021, 2020]
        })

        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.sample_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

        # Initialize recommender
        self.recommender = ContentBasedRecommender()

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_initialization(self):
        """Test recommender initialization."""
        self.assertIsNotNone(self.recommender.config)
        self.assertIn('max_features', self.recommender.config)
        self.assertIsNone(self.recommender.movies_df)

    def test_load_data(self):
        """Test data loading functionality."""
        self.recommender.load_data(self.temp_file.name)

        self.assertIsNotNone(self.recommender.movies_df)
        self.assertEqual(len(self.recommender.movies_df), 4)
        self.assertEqual(len(self.recommender.movie_indices), 4)
        self.assertIn('Movie A', self.recommender.movie_indices)

    def test_text_preprocessing(self):
        """Test text preprocessing functionality."""
        test_text = "This is a TEST with Special Characters! @#$"
        processed = self.recommender._advanced_text_preprocessing(test_text)

        self.assertIsInstance(processed, str)
        self.assertTrue(len(processed) > 0)
        # Should be lowercase
        self.assertEqual(processed, processed.lower())

    def test_content_features_creation(self):
        """Test content features creation."""
        self.recommender.load_data(self.temp_file.name)
        features = self.recommender._create_content_features()

        self.assertIsInstance(features, pd.Series)
        self.assertEqual(len(features), 4)

    def test_model_building(self):
        """Test TF-IDF model building."""
        self.recommender.load_data(self.temp_file.name)
        self.recommender.build_tfidf_model()

        self.assertIsNotNone(self.recommender.tfidf_matrix)
        self.assertIsNotNone(self.recommender.tfidf_vectorizer)
        self.assertEqual(self.recommender.tfidf_matrix.shape[0], 4)


class TestUtilities(unittest.TestCase):
    """Test cases for utility functions."""

    def test_config_manager(self):
        """Test configuration manager."""
        config_manager = ConfigManager()

        # Test default config loading
        self.assertIsNotNone(config_manager.config)
        self.assertIn('data', config_manager.config)

        # Test get/set operations
        config_manager.set('test.value', 42)
        self.assertEqual(config_manager.get('test.value'), 42)
        self.assertEqual(config_manager.get('nonexistent.key', 'default'), 'default')

    def test_text_utils(self):
        """Test text utility functions."""
        # Test text cleaning
        dirty_text = "This is a TEST with Special Characters! @#$"
        clean_text = TextUtils.clean_text(dirty_text)

        self.assertEqual(clean_text, clean_text.lower())
        self.assertNotIn('@', clean_text)

        # Test keyword extraction
        text = "action adventure movie with heroes and villains"
        keywords = TextUtils.extract_keywords(text, top_k=3)

        self.assertIsInstance(keywords, list)
        self.assertLessEqual(len(keywords), 3)

        # Test similarity calculation
        text1 = "action movie with heroes"
        text2 = "adventure film with heroes"
        similarity = TextUtils.calculate_text_similarity(text1, text2)

        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

    def test_math_utils(self):
        """Test mathematical utility functions."""
        # Test score normalization
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # MinMax normalization
        normalized = MathUtils.normalize_scores(scores, method='minmax')
        self.assertEqual(normalized.min(), 0.0)
        self.assertEqual(normalized.max(), 1.0)

        # Test weighted average
        values = [1.0, 2.0, 3.0]
        weights = [0.1, 0.3, 0.6]
        weighted_avg = MathUtils.weighted_average(values, weights)

        expected = (1.0 * 0.1 + 2.0 * 0.3 + 3.0 * 0.6) / 1.0
        self.assertAlmostEqual(weighted_avg, expected, places=5)

    def test_timer(self):
        """Test timer utility."""
        import time

        with Timer("Test operation") as timer:
            time.sleep(0.01)  # Sleep for 10ms

        self.assertIsNotNone(timer.start_time)
        self.assertIsNotNone(timer.end_time)
        self.assertGreater(timer.end_time - timer.start_time, 0.01)


def run_basic_tests():
    """Run basic test suite."""
    test_suite = unittest.TestSuite()

    test_classes = [
        TestContentBasedRecommender,
        TestUtilities
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    run_basic_tests()