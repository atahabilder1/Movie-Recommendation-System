# -*- coding: utf-8 -*-
"""
Professional Data Preprocessing Pipeline for Movie Recommendation System
========================================================================

This module implements comprehensive data preprocessing with:
- Data quality assessment and validation
- Missing value handling strategies
- Feature engineering for recommendation models
- Data type optimization and memory efficiency
- Scalable preprocessing for large datasets

Author: Portfolio Project for Data Science Applications
"""

import pandas as pd
import numpy as np
import json
import re
import logging
import os
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging for professional debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TMDBDataProcessor:
    """
    Professional data preprocessing pipeline for TMDB movie dataset.

    Features:
    - Comprehensive data quality assessment
    - Memory-optimized processing for large datasets
    - Advanced feature engineering for recommendation systems
    - Robust error handling and validation
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize processor with configuration."""
        self.config = config or self._default_config()
        self.data_quality_report = {}
        self.preprocessing_stats = {}

    def _default_config(self) -> Dict:
        """Default preprocessing configuration."""
        return {
            'min_vote_count': 10,           # Minimum votes for reliability
            'min_year': 1950,               # Filter very old movies
            'max_year': 2024,               # Current year limit
            'min_runtime': 60,              # Minimum movie duration
            'text_min_length': 10,          # Minimum overview length
            'memory_optimization': True,     # Optimize data types
            'chunk_size': 50000,            # For large dataset processing
        }

    def load_and_assess_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data with comprehensive quality assessment.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with initial assessment completed
        """
        logger.info(f"Loading dataset from {filepath}")

        # Memory-efficient loading for large datasets
        try:
            df = pd.read_csv(filepath, low_memory=False)
            logger.info(f"Successfully loaded {len(df):,} records")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

        # Immediate data quality assessment
        self._assess_data_quality(df)

        return df

    def _assess_data_quality(self, df: pd.DataFrame) -> None:
        """Comprehensive data quality assessment."""
        logger.info("Conducting data quality assessment...")

        quality_report = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': {},
            'data_types': {},
            'duplicates': df.duplicated().sum(),
            'unique_values': {}
        }

        # Missing value analysis
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            quality_report['missing_values'][col] = {
                'count': missing_count,
                'percentage': round(missing_pct, 2)
            }

        # Data type analysis
        for col in df.columns:
            quality_report['data_types'][col] = str(df[col].dtype)
            quality_report['unique_values'][col] = df[col].nunique()

        self.data_quality_report = quality_report

        # Log critical issues
        high_missing = [col for col, info in quality_report['missing_values'].items()
                       if info['percentage'] > 50]
        if high_missing:
            logger.warning(f"High missing values in: {high_missing}")

        logger.info(f"Data quality assessment completed. {quality_report['duplicates']} duplicates found.")

    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data with professional standards.

        Args:
            df: Raw dataframe

        Returns:
            Cleaned and validated dataframe
        """
        logger.info("Starting data cleaning and validation...")
        initial_count = len(df)

        # Create a copy to avoid modifying original
        df_clean = df.copy()

        # 1. Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=['id'], keep='first')
        logger.info(f"Removed {initial_count - len(df_clean)} duplicate records")

        # 2. Handle missing critical features
        critical_features = ['id', 'title', 'overview']
        df_clean = df_clean.dropna(subset=critical_features)
        logger.info(f"Removed records with missing critical features")

        # 3. Data type optimization and validation
        df_clean = self._optimize_data_types(df_clean)

        # 4. Content validation
        df_clean = self._validate_content(df_clean)

        # 5. Feature-specific cleaning
        df_clean = self._clean_features(df_clean)

        # Update preprocessing stats
        self.preprocessing_stats = {
            'initial_records': initial_count,
            'final_records': len(df_clean),
            'records_removed': initial_count - len(df_clean),
            'removal_percentage': round(((initial_count - len(df_clean)) / initial_count) * 100, 2)
        }

        logger.info(f"Data cleaning completed. Retained {len(df_clean):,} records")

        return df_clean

    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        if not self.config['memory_optimization']:
            return df

        logger.info("Optimizing data types for memory efficiency...")

        # Numeric optimizations
        numeric_cols = ['vote_average', 'vote_count', 'revenue', 'budget', 'runtime', 'popularity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Boolean conversion
        if 'adult' in df.columns:
            df['adult'] = df['adult'].map({'False': False, 'True': True, False: False, True: True})

        # Date conversion
        if 'release_date' in df.columns:
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

        return df

    def _validate_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate content according to business rules."""
        logger.info("Validating content according to business rules...")

        initial_count = len(df)

        # Filter by vote count (reliability threshold)
        if 'vote_count' in df.columns:
            df = df[df['vote_count'] >= self.config['min_vote_count']]

        # Filter by release year
        if 'release_date' in df.columns:
            df = df[df['release_date'].dt.year.between(
                self.config['min_year'],
                self.config['max_year']
            )]

        # Filter by runtime
        if 'runtime' in df.columns:
            df = df[df['runtime'] >= self.config['min_runtime']]

        # Filter by overview length
        if 'overview' in df.columns:
            df = df[df['overview'].str.len() >= self.config['text_min_length']]

        # Remove adult content for general recommendations
        if 'adult' in df.columns:
            df = df[df['adult'] == False]

        logger.info(f"Content validation removed {initial_count - len(df)} records")
        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean individual features with advanced techniques."""
        logger.info("Cleaning individual features...")

        # Clean text features
        text_features = ['overview', 'tagline', 'title']
        for col in text_features:
            if col in df.columns:
                df[col] = self._clean_text(df[col])

        # Parse JSON-like features
        json_features = ['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']
        for col in json_features:
            if col in df.columns:
                df[col] = self._parse_json_feature(df[col])

        # Handle missing values strategically
        df = self._handle_missing_values(df)

        return df

    def _clean_text(self, series: pd.Series) -> pd.Series:
        """Advanced text cleaning for NLP processing."""
        # Remove extra whitespace and normalize
        cleaned = series.astype(str).str.strip()
        cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)

        # Replace 'nan' string with actual NaN
        cleaned = cleaned.replace(['nan', 'None', ''], np.nan)

        return cleaned

    def _parse_json_feature(self, series: pd.Series) -> pd.Series:
        """Parse JSON-like string features into clean lists."""
        def safe_parse(x):
            if pd.isna(x) or x == '':
                return []
            try:
                # Simple comma-based parsing for now
                return [item.strip() for item in str(x).split(',') if item.strip()]
            except:
                return []

        return series.apply(safe_parse)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strategic missing value handling based on feature importance."""
        logger.info("Handling missing values strategically...")

        # Numeric features: median imputation for recommendation-relevant features
        numeric_features = ['vote_average', 'popularity', 'runtime']
        for col in numeric_features:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled {col} missing values with median: {median_val}")

        # Text features: fill with empty string for processing
        text_features = ['tagline', 'homepage']
        for col in text_features:
            if col in df.columns:
                df[col] = df[col].fillna('')

        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for recommendation systems.
        """
        logger.info("Starting advanced feature engineering...")

        df_engineered = df.copy()

        # 1. Temporal features
        df_engineered = self._create_temporal_features(df_engineered)

        # 2. Content-based features
        df_engineered = self._create_content_features(df_engineered)

        # 3. Popularity and quality features
        df_engineered = self._create_popularity_features(df_engineered)

        logger.info("Feature engineering completed")
        return df_engineered

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if 'release_date' in df.columns:
            df['release_year'] = df['release_date'].dt.year
            df['release_month'] = df['release_date'].dt.month
            df['release_decade'] = (df['release_year'] // 10) * 10
            df['movie_age'] = 2024 - df['release_year']

        return df

    def _create_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create content-based features for recommendation."""
        # Text length features
        if 'overview' in df.columns:
            df['overview_length'] = df['overview'].str.len()
            df['overview_word_count'] = df['overview'].str.split().str.len()

        # Genre features
        if 'genres' in df.columns:
            df['genre_count'] = df['genres'].apply(len)

        # Keywords features
        if 'keywords' in df.columns:
            df['keyword_count'] = df['keywords'].apply(len)

        return df

    def _create_popularity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create popularity and quality indicators."""
        # Rating features
        if 'vote_average' in df.columns and 'vote_count' in df.columns:
            # Weighted rating (IMDB formula)
            C = df['vote_average'].mean()
            m = df['vote_count'].quantile(0.9)

            df['weighted_rating'] = (df['vote_count'] / (df['vote_count'] + m) * df['vote_average'] +
                                   m / (df['vote_count'] + m) * C)

            # Rating categories
            df['rating_category'] = pd.cut(df['vote_average'],
                                         bins=[0, 5, 6.5, 8, 10],
                                         labels=['Poor', 'Average', 'Good', 'Excellent'])

        # Budget/Revenue features
        if 'budget' in df.columns and 'revenue' in df.columns:
            df['profit'] = df['revenue'] - df['budget']
            df['roi'] = np.where(df['budget'] > 0, df['profit'] / df['budget'], 0)

        return df

    def get_data_summary(self) -> Dict:
        """Get comprehensive data summary for reporting."""
        return {
            'data_quality_report': self.data_quality_report,
            'preprocessing_stats': self.preprocessing_stats,
            'config': self.config
        }

    def save_processed_data(self, df: pd.DataFrame, filepath: str) -> None:
        """Save processed data with compression."""
        logger.info(f"Saving processed data to {filepath}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        df.to_csv(filepath, index=False, compression='gzip')
        logger.info(f"Saved {len(df):,} records successfully")


def main():
    """Main preprocessing pipeline execution."""
    # Create results directory if it doesn't exist
    os.makedirs('results/metrics', exist_ok=True)

    # Initialize processor with professional configuration
    processor = TMDBDataProcessor()

    # Load and process data
    raw_data = processor.load_and_assess_data('data/TMDB_movie_dataset_v11.csv')
    cleaned_data = processor.clean_and_validate(raw_data)
    processed_data = processor.feature_engineering(cleaned_data)

    # Save processed data
    processor.save_processed_data(processed_data, 'data/processed_movies.csv.gz')

    # Generate summary report
    summary = processor.get_data_summary()
    with open('results/metrics/preprocessing_report.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("✅ Professional data preprocessing completed!")
    print(f"📊 Processed {len(processed_data):,} movies")
    print(f"📈 Data quality report saved to results/metrics/")


if __name__ == "__main__":
    main()