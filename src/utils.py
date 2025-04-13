# -*- coding: utf-8 -*-
"""
Utility Functions for Movie Recommendation System
================================================

This module provides common utility functions, helpers, and shared
functionality used across the recommendation system components.

Functions Include:
- Data validation and preprocessing helpers
- Performance monitoring and profiling
- Configuration management
- File I/O operations
- Logging utilities
- Mathematical and statistical helpers

Author: Portfolio Project for Data Science Applications
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from functools import wraps
from datetime import datetime, timedelta
import psutil
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code execution."""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Started: {self.description}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"Completed: {self.description} in {duration:.2f} seconds")


class MemoryMonitor:
    """Monitor memory usage during operations."""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None

    def start_monitoring(self) -> float:
        """Start monitoring memory usage."""
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self.initial_memory

    def get_current_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_memory_delta(self) -> float:
        """Get memory usage increase since monitoring started."""
        if self.initial_memory is None:
            return 0.0
        return self.get_current_usage() - self.initial_memory


def timing_decorator(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def memory_profile(func: Callable) -> Callable:
    """Decorator to profile memory usage of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        initial_memory = monitor.start_monitoring()
        result = func(*args, **kwargs)
        final_memory = monitor.get_current_usage()
        memory_delta = monitor.get_memory_delta()

        logger.info(f"{func.__name__} memory usage: "
                   f"Initial: {initial_memory:.1f}MB, "
                   f"Final: {final_memory:.1f}MB, "
                   f"Delta: {memory_delta:.1f}MB")
        return result
    return wrapper


class ConfigManager:
    """Manage configuration settings for the recommendation system."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self.config = self._load_default_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "data": {
                "raw_data_path": "data/TMDB_movie_dataset_v11.csv",
                "processed_data_path": "data/processed_movies.csv.gz",
                "chunk_size": 10000,
                "encoding": "utf-8"
            },
            "content_based": {
                "max_features": 10000,
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_df": 0.8,
                "use_svd": True,
                "svd_components": 300,
                "similarity_threshold": 0.1
            },
            "collaborative": {
                "svd_components": 50,
                "embedding_dim": 64,
                "hidden_dims": [128, 64],
                "learning_rate": 0.001,
                "batch_size": 1024,
                "epochs": 50
            },
            "evaluation": {
                "k_values": [1, 3, 5, 10, 20],
                "relevance_threshold": 4.0,
                "test_size": 0.2,
                "random_state": 42
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(levelname)s - %(message)s",
                "file": "recommendation_system.log"
            }
        }

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file."""
        path = config_path or self.config_path
        if os.path.exists(path):
            with open(path, 'r') as f:
                file_config = json.load(f)
            # Merge with default config
            self.config = self._deep_merge(self.config, file_config)
            logger.info(f"Configuration loaded from {path}")
        else:
            logger.info("Configuration file not found, using defaults")
        return self.config

    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        path = config_path or self.config_path
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)
        logger.info(f"Configuration saved to {path}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'data.chunk_size')."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


class DataValidator:
    """Validate data quality and format for recommendation system."""

    @staticmethod
    def validate_movie_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate movie dataset."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # Required columns
        required_columns = ['title', 'overview']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
            validation_results['is_valid'] = False

        # Check for duplicates
        if df.duplicated().sum() > 0:
            validation_results['warnings'].append(f"Found {df.duplicated().sum()} duplicate rows")

        # Check for missing values
        missing_stats = df.isnull().sum()
        if missing_stats.sum() > 0:
            validation_results['warnings'].append(f"Missing values found: {missing_stats.to_dict()}")

        # Basic statistics
        validation_results['statistics'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        }

        return validation_results

    @staticmethod
    def validate_ratings_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate ratings dataset."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # Required columns for ratings
        required_columns = ['user_id', 'movie_id', 'rating']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
            validation_results['is_valid'] = False

        if validation_results['is_valid']:
            # Check rating range
            if 'rating' in df.columns:
                min_rating = df['rating'].min()
                max_rating = df['rating'].max()
                if min_rating < 0 or max_rating > 10:
                    validation_results['warnings'].append(
                        f"Rating values outside expected range [0-10]: [{min_rating}, {max_rating}]")

            # Statistics
            validation_results['statistics'] = {
                'total_ratings': len(df),
                'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else 0,
                'unique_movies': df['movie_id'].nunique() if 'movie_id' in df.columns else 0,
                'rating_distribution': df['rating'].value_counts().to_dict() if 'rating' in df.columns else {},
                'sparsity': 1 - (len(df) / (df['user_id'].nunique() * df['movie_id'].nunique()))
                           if all(col in df.columns for col in ['user_id', 'movie_id']) else 0
            }

        return validation_results


class FileManager:
    """Manage file operations for the recommendation system."""

    @staticmethod
    def safe_save_pickle(obj: Any, filepath: str, backup: bool = True) -> None:
        """Safely save object to pickle file with optional backup."""
        if backup and os.path.exists(filepath):
            backup_path = f"{filepath}.backup.{int(time.time())}"
            os.rename(filepath, backup_path)
            logger.info(f"Created backup: {backup_path}")

        temp_path = f"{filepath}.tmp"
        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(obj, f)
            os.rename(temp_path, filepath)
            logger.info(f"Successfully saved to {filepath}")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"Failed to save {filepath}: {e}")
            raise

    @staticmethod
    def safe_load_pickle(filepath: str) -> Any:
        """Safely load object from pickle file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            logger.info(f"Successfully loaded from {filepath}")
            return obj
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            raise

    @staticmethod
    def ensure_directory(filepath: str) -> None:
        """Ensure directory exists for the given filepath."""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

    @staticmethod
    def get_file_hash(filepath: str) -> str:
        """Get MD5 hash of file for integrity checking."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def clean_old_files(directory: str, pattern: str, days_old: int = 7) -> List[str]:
        """Clean old files matching pattern from directory."""
        if not os.path.exists(directory):
            return []

        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        cleaned_files = []

        for filename in os.listdir(directory):
            if pattern in filename:
                filepath = os.path.join(directory, filename)
                if os.path.getctime(filepath) < cutoff_time:
                    try:
                        os.remove(filepath)
                        cleaned_files.append(filepath)
                        logger.info(f"Cleaned old file: {filepath}")
                    except Exception as e:
                        logger.error(f"Failed to clean {filepath}: {e}")

        return cleaned_files


class TextUtils:
    """Text processing utilities for recommendation system."""

    @staticmethod
    def clean_text(text: str, remove_special_chars: bool = True) -> str:
        """Clean and normalize text."""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower().strip()

        # Remove special characters if requested
        if remove_special_chars:
            import re
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """Extract top keywords from text using simple frequency analysis."""
        if not text:
            return []

        # Simple tokenization and frequency counting
        words = text.lower().split()
        word_freq = {}

        # Common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}

        for word in words:
            if len(word) > 2 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]

    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """Calculate simple Jaccard similarity between two texts."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class MathUtils:
    """Mathematical utilities for recommendation system."""

    @staticmethod
    def normalize_scores(scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize scores using different methods."""
        if len(scores) == 0:
            return scores

        if method == 'minmax':
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score == min_score:
                return np.ones_like(scores)
            return (scores - min_score) / (max_score - min_score)

        elif method == 'zscore':
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            if std_score == 0:
                return np.zeros_like(scores)
            return (scores - mean_score) / std_score

        elif method == 'sigmoid':
            return 1 / (1 + np.exp(-scores))

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def calculate_diversity_score(items: List[Any],
                                similarity_func: Callable[[Any, Any], float]) -> float:
        """Calculate diversity score for a list of items."""
        if len(items) < 2:
            return 0.0

        total_similarity = 0.0
        pair_count = 0

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                total_similarity += similarity_func(items[i], items[j])
                pair_count += 1

        average_similarity = total_similarity / pair_count if pair_count > 0 else 0.0
        return 1.0 - average_similarity  # Diversity is inverse of similarity

    @staticmethod
    def weighted_average(values: List[float], weights: List[float]) -> float:
        """Calculate weighted average."""
        if len(values) != len(weights) or len(values) == 0:
            return 0.0

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(v * w for v, w in zip(values, weights))
        return weighted_sum / total_weight


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration for the recommendation system."""
    log_config = config.get('logging', {})

    level = getattr(logging, log_config.get('level', 'INFO').upper())
    format_str = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', 'recommendation_system.log')

    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

    logger.info("Logging configured successfully")


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging and monitoring."""
    return {
        'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'disk_usage_gb': psutil.disk_usage('/').total / (1024**3),
        'timestamp': datetime.now().isoformat()
    }


def main():
    """Example usage of utility functions."""
    print("Recommendation System Utilities")
    print("==============================")

    # Test configuration manager
    config_manager = ConfigManager()
    print(f"Default config loaded with {len(config_manager.config)} sections")

    # Test system info
    system_info = get_system_info()
    print(f"System has {system_info['cpu_count']} CPUs and {system_info['memory_total_gb']:.1f}GB RAM")

    # Test timer
    with Timer("Example operation"):
        time.sleep(0.1)

    print("All utility functions loaded successfully!")


if __name__ == "__main__":
    main()