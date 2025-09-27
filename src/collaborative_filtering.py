# -*- coding: utf-8 -*-
"""
Professional Collaborative Filtering Movie Recommendation System
================================================================

This module implements multiple collaborative filtering approaches:
- Matrix Factorization using SVD
- Neural Collaborative Filtering (NCF)
- User-Item interaction modeling
- Advanced evaluation metrics for recommendation systems

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

# ML and Deep Learning imports
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MovieRatingsDataset(Dataset):
    """PyTorch Dataset for movie ratings data."""

    def __init__(self, user_ids, movie_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.movie_ids = torch.LongTensor(movie_ids)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering model for movie recommendations.

    Combines matrix factorization with neural networks for improved performance.
    """

    def __init__(self, num_users, num_movies, embedding_dim=50, hidden_dims=[128, 64]):
        super(NeuralCollaborativeFiltering, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # Neural MF path
        self.user_embedding_mf = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding_mf = nn.Embedding(num_movies, embedding_dim)

        # MLP layers
        input_dim = embedding_dim * 2
        self.mlp_layers = nn.ModuleList()

        for hidden_dim in hidden_dims:
            self.mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        # Final prediction layers
        self.mlp_output = nn.Linear(hidden_dims[-1], 1)
        self.mf_output = nn.Linear(embedding_dim, 1)

        # Combine both paths
        self.final_layer = nn.Linear(2, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for layer in self.modules():
            if isinstance(layer, nn.Embedding):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_ids, movie_ids):
        # MLP path
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        mlp_input = torch.cat([user_emb, movie_emb], dim=1)

        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
        mlp_pred = self.mlp_output(mlp_output)

        # Matrix Factorization path
        user_emb_mf = self.user_embedding_mf(user_ids)
        movie_emb_mf = self.movie_embedding_mf(movie_ids)
        mf_output = user_emb_mf * movie_emb_mf
        mf_pred = self.mf_output(mf_output)

        # Combine predictions
        combined = torch.cat([mlp_pred, mf_pred], dim=1)
        final_pred = self.final_layer(combined)

        return final_pred.squeeze()


class CollaborativeFilteringRecommender:
    """
    Professional collaborative filtering recommendation system.

    Features:
    - Matrix Factorization using SVD
    - Neural Collaborative Filtering
    - User-item interaction modeling
    - Comprehensive evaluation metrics
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize collaborative filtering recommender."""
        self.config = config or self._default_config()
        self.movies_df = None
        self.ratings_matrix = None
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.svd_model = None
        self.ncf_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _default_config(self) -> Dict:
        """Default configuration for collaborative filtering."""
        return {
            # SVD parameters
            'svd_components': 50,
            'svd_random_state': 42,

            # Neural CF parameters
            'embedding_dim': 50,
            'hidden_dims': [128, 64],
            'learning_rate': 0.001,
            'batch_size': 1024,
            'epochs': 50,
            'early_stopping': 5,

            # Data parameters
            'min_user_ratings': 5,     # Minimum ratings per user
            'min_movie_ratings': 3,    # Minimum ratings per movie
            'test_size': 0.2,          # Train-test split ratio
            'implicit_threshold': 3.5, # Threshold for implicit feedback
        }

    def load_data(self, filepath: str) -> None:
        """Load and prepare movie data for collaborative filtering."""
        logger.info(f"Loading movie data from {filepath}")
        self.movies_df = pd.read_csv(filepath, compression='gzip' if filepath.endswith('.gz') else None)
        logger.info(f"Loaded {len(self.movies_df)} movies")

    def _create_user_item_matrix(self) -> pd.DataFrame:
        """
        Create user-item interaction matrix from movie ratings.

        Since we don't have explicit user ratings, we'll create synthetic users
        based on movie characteristics and voting patterns.
        """
        logger.info("Creating synthetic user-item interaction matrix...")

        # Strategy: Create user profiles based on genre preferences and rating patterns
        np.random.seed(42)

        # Extract genres and create user profiles
        all_genres = set()
        for genres_str in self.movies_df['genres'].dropna():
            try:
                import ast
                genres = ast.literal_eval(genres_str) if isinstance(genres_str, str) else genres_str
                if isinstance(genres, list):
                    all_genres.update(genres)
            except:
                pass

        all_genres = list(all_genres)
        logger.info(f"Found {len(all_genres)} unique genres")

        # Create synthetic users with genre preferences
        num_users = min(500, len(self.movies_df) // 50)  # Reasonable number of synthetic users
        user_data = []

        for user_id in range(num_users):
            # Create user profile: preferred genres, rating bias, selectivity
            num_preferred_genres = np.random.randint(1, min(4, len(all_genres)))
            preferred_genres = np.random.choice(all_genres, num_preferred_genres, replace=False)
            rating_bias = np.random.normal(6.0, 0.8)  # Personal rating tendency
            selectivity = 0.03  # Fixed selectivity to ensure enough ratings

            # Generate ratings for movies
            user_ratings = []
            num_ratings = max(self.config['min_user_ratings'] * 2, int(len(self.movies_df) * selectivity))
            rated_movies = np.random.choice(len(self.movies_df),
                                          size=min(num_ratings, len(self.movies_df)),
                                          replace=False)

            for movie_idx in rated_movies:
                movie = self.movies_df.iloc[movie_idx]

                # Calculate rating based on genre match and movie quality
                base_rating = movie['vote_average']

                # Bonus for preferred genres
                genre_bonus = 0
                try:
                    import ast
                    movie_genres = ast.literal_eval(movie['genres']) if isinstance(movie['genres'], str) else movie['genres']
                    if isinstance(movie_genres, list):
                        genre_matches = len(set(movie_genres) & set(preferred_genres))
                        genre_bonus = genre_matches * 0.5
                except:
                    pass

                # Add user bias and some noise
                final_rating = base_rating + genre_bonus + (rating_bias - 6.0) + np.random.normal(0, 0.5)
                final_rating = np.clip(final_rating, 1.0, 10.0)

                user_ratings.append({
                    'user_id': user_id,
                    'movie_id': movie_idx,
                    'title': movie['title'],
                    'rating': final_rating
                })

            user_data.extend(user_ratings)

        # Create ratings DataFrame
        ratings_df = pd.DataFrame(user_data)

        # Filter based on minimum requirements
        user_counts = ratings_df['user_id'].value_counts()
        movie_counts = ratings_df['movie_id'].value_counts()

        valid_users = user_counts[user_counts >= self.config['min_user_ratings']].index
        valid_movies = movie_counts[movie_counts >= self.config['min_movie_ratings']].index

        filtered_ratings = ratings_df[
            (ratings_df['user_id'].isin(valid_users)) &
            (ratings_df['movie_id'].isin(valid_movies))
        ]

        logger.info(f"Created {len(filtered_ratings)} ratings from {len(valid_users)} users for {len(valid_movies)} movies")

        return filtered_ratings

    def prepare_collaborative_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for collaborative filtering models."""
        # Create synthetic user-item interactions
        ratings_df = self._create_user_item_matrix()

        # Encode users and movies
        ratings_df['user_encoded'] = self.user_encoder.fit_transform(ratings_df['user_id'])
        ratings_df['movie_encoded'] = self.movie_encoder.fit_transform(ratings_df['movie_id'])

        # Split into train and test sets
        train_data, test_data = train_test_split(
            ratings_df,
            test_size=self.config['test_size'],
            random_state=42,
            stratify=ratings_df['user_encoded']
        )

        # Extract features and targets
        X_train = train_data[['user_encoded', 'movie_encoded']].values
        y_train = train_data['rating'].values
        X_test = test_data[['user_encoded', 'movie_encoded']].values
        y_test = test_data['rating'].values

        # Store for later use
        self.train_data = train_data
        self.test_data = test_data
        self.num_users = len(self.user_encoder.classes_)
        self.num_movies = len(self.movie_encoder.classes_)

        logger.info(f"Prepared data: {len(X_train)} train samples, {len(X_test)} test samples")
        logger.info(f"Users: {self.num_users}, Movies: {self.num_movies}")

        return X_train, X_test, y_train, y_test

    def train_svd_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train matrix factorization model using SVD."""
        logger.info("Training SVD-based matrix factorization model...")

        # Create user-item matrix
        user_item_matrix = np.zeros((self.num_users, self.num_movies))

        for (user_id, movie_id), rating in zip(X_train, y_train):
            user_item_matrix[user_id, movie_id] = rating

        # Apply SVD
        self.svd_model = TruncatedSVD(
            n_components=self.config['svd_components'],
            random_state=self.config['svd_random_state']
        )

        # Fit SVD on user-item matrix
        self.user_factors = self.svd_model.fit_transform(user_item_matrix)
        self.movie_factors = self.svd_model.components_.T

        # Store the mean rating for bias correction
        self.global_mean = y_train.mean()

        logger.info(f"SVD model trained with {self.config['svd_components']} components")
        logger.info(f"Explained variance ratio: {self.svd_model.explained_variance_ratio_.sum():.3f}")

    def train_ncf_model(self, X_train: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_test: np.ndarray) -> None:
        """Train Neural Collaborative Filtering model."""
        logger.info(f"Training Neural Collaborative Filtering model on {self.device}")

        # Create datasets
        train_dataset = MovieRatingsDataset(X_train[:, 0], X_train[:, 1], y_train)
        test_dataset = MovieRatingsDataset(X_test[:, 0], X_test[:, 1], y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)

        # Initialize model
        self.ncf_model = NeuralCollaborativeFiltering(
            num_users=self.num_users,
            num_movies=self.num_movies,
            embedding_dim=self.config['embedding_dim'],
            hidden_dims=self.config['hidden_dims']
        ).to(self.device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.ncf_model.parameters(), lr=self.config['learning_rate'])

        # Training loop
        best_test_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            # Training phase
            self.ncf_model.train()
            train_loss = 0.0

            for user_ids, movie_ids, ratings in train_loader:
                user_ids, movie_ids, ratings = user_ids.to(self.device), movie_ids.to(self.device), ratings.to(self.device)

                optimizer.zero_grad()
                predictions = self.ncf_model(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            self.ncf_model.eval()
            test_loss = 0.0

            with torch.no_grad():
                for user_ids, movie_ids, ratings in test_loader:
                    user_ids, movie_ids, ratings = user_ids.to(self.device), movie_ids.to(self.device), ratings.to(self.device)
                    predictions = self.ncf_model(user_ids, movie_ids)
                    loss = criterion(predictions, ratings)
                    test_loss += loss.item()

            train_loss /= len(train_loader)
            test_loss /= len(test_loader)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                # Save best model
                torch.save(self.ncf_model.state_dict(), 'results/models/ncf_best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Load best model
        self.ncf_model.load_state_dict(torch.load('results/models/ncf_best_model.pth'))
        logger.info(f"NCF model training completed. Best test loss: {best_test_loss:.4f}")

    def predict_svd(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Make predictions using SVD model."""
        predictions = []

        for user_id, movie_id in zip(user_ids, movie_ids):
            if user_id < len(self.user_factors) and movie_id < len(self.movie_factors):
                pred = np.dot(self.user_factors[user_id], self.movie_factors[movie_id]) + self.global_mean
                predictions.append(np.clip(pred, 1.0, 10.0))
            else:
                predictions.append(self.global_mean)  # Fallback to global mean

        return np.array(predictions)

    def predict_ncf(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Make predictions using NCF model."""
        self.ncf_model.eval()
        predictions = []

        with torch.no_grad():
            for user_id, movie_id in zip(user_ids, movie_ids):
                user_tensor = torch.LongTensor([user_id]).to(self.device)
                movie_tensor = torch.LongTensor([movie_id]).to(self.device)
                pred = self.ncf_model(user_tensor, movie_tensor).cpu().item()
                predictions.append(np.clip(pred, 1.0, 10.0))

        return np.array(predictions)

    def get_user_recommendations(self, user_id: int, top_k: int = 10, model_type: str = 'ncf') -> List[Dict]:
        """
        Get movie recommendations for a specific user.

        Args:
            user_id: User ID for recommendations
            top_k: Number of recommendations
            model_type: 'svd' or 'ncf'

        Returns:
            List of recommended movies with scores
        """
        if user_id >= self.num_users:
            raise ValueError(f"User ID {user_id} not found. Max user ID: {self.num_users - 1}")

        # Get movies the user hasn't rated
        user_rated_movies = set(self.train_data[self.train_data['user_encoded'] == user_id]['movie_encoded'])
        unrated_movies = [i for i in range(self.num_movies) if i not in user_rated_movies]

        if not unrated_movies:
            return []

        # Make predictions for unrated movies
        user_ids = [user_id] * len(unrated_movies)
        movie_ids = unrated_movies

        if model_type == 'svd':
            predictions = self.predict_svd(np.array(user_ids), np.array(movie_ids))
        else:
            predictions = self.predict_ncf(np.array(user_ids), np.array(movie_ids))

        # Get top recommendations
        movie_scores = list(zip(unrated_movies, predictions))
        movie_scores.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        for movie_encoded, score in movie_scores[:top_k]:
            # Decode movie ID back to original
            original_movie_id = self.movie_encoder.inverse_transform([movie_encoded])[0]
            movie_info = self.movies_df.iloc[original_movie_id]

            recommendations.append({
                'title': movie_info['title'],
                'predicted_rating': float(score),
                'actual_rating': float(movie_info['vote_average']),
                'release_year': int(movie_info['release_year']),
                'genres': movie_info['genres'],
                'overview': movie_info['overview'][:200] + '...' if len(str(movie_info['overview'])) > 200 else movie_info['overview']
            })

        return recommendations

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate both SVD and NCF models."""
        logger.info("Evaluating collaborative filtering models...")

        # SVD predictions
        svd_predictions = self.predict_svd(X_test[:, 0], X_test[:, 1])
        svd_rmse = np.sqrt(mean_squared_error(y_test, svd_predictions))
        svd_mae = mean_absolute_error(y_test, svd_predictions)

        # NCF predictions
        ncf_predictions = self.predict_ncf(X_test[:, 0], X_test[:, 1])
        ncf_rmse = np.sqrt(mean_squared_error(y_test, ncf_predictions))
        ncf_mae = mean_absolute_error(y_test, ncf_predictions)

        evaluation_results = {
            'svd': {
                'rmse': svd_rmse,
                'mae': svd_mae,
                'components': self.config['svd_components']
            },
            'ncf': {
                'rmse': ncf_rmse,
                'mae': ncf_mae,
                'embedding_dim': self.config['embedding_dim']
            },
            'dataset_stats': {
                'test_samples': len(X_test),
                'num_users': self.num_users,
                'num_movies': self.num_movies,
                'rating_range': f"{y_test.min():.1f} - {y_test.max():.1f}",
                'mean_rating': y_test.mean()
            }
        }

        return evaluation_results

    def save_models(self, filepath_prefix: str) -> None:
        """Save trained models."""
        logger.info(f"Saving collaborative filtering models...")

        # Save SVD model
        svd_data = {
            'model': self.svd_model,
            'user_factors': self.user_factors,
            'movie_factors': self.movie_factors,
            'global_mean': self.global_mean,
            'user_encoder': self.user_encoder,
            'movie_encoder': self.movie_encoder,
            'config': self.config
        }

        with open(f"{filepath_prefix}_svd.pkl", 'wb') as f:
            pickle.dump(svd_data, f)

        # NCF model is already saved during training
        logger.info("Models saved successfully")


def main():
    """Main function to demonstrate collaborative filtering."""
    print("🤝 Building Professional Collaborative Filtering Recommender")
    print("=" * 65)

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    # Initialize recommender
    recommender = CollaborativeFilteringRecommender()

    # Load data
    recommender.load_data('data/processed_movies.csv.gz')

    # Prepare collaborative filtering data
    print("\n📊 Preparing collaborative filtering data...")
    X_train, X_test, y_train, y_test = recommender.prepare_collaborative_data()

    # Train SVD model
    print("\n🔧 Training Matrix Factorization (SVD) model...")
    recommender.train_svd_model(X_train, y_train)

    # Train NCF model
    print(f"\n🧠 Training Neural Collaborative Filtering model...")
    recommender.train_ncf_model(X_train, X_test, y_train, y_test)

    # Evaluate models
    print(f"\n📈 Evaluating models...")
    evaluation = recommender.evaluate_models(X_test, y_test)

    print(f"\n📊 Model Performance:")
    print(f"SVD - RMSE: {evaluation['svd']['rmse']:.4f}, MAE: {evaluation['svd']['mae']:.4f}")
    print(f"NCF - RMSE: {evaluation['ncf']['rmse']:.4f}, MAE: {evaluation['ncf']['mae']:.4f}")

    # Test recommendations
    print(f"\n🎯 Sample User Recommendations:")
    print("-" * 50)

    for user_id in [0, 5, 10]:
        try:
            print(f"\n👤 User {user_id} - SVD Recommendations:")
            svd_recs = recommender.get_user_recommendations(user_id, top_k=5, model_type='svd')
            for i, rec in enumerate(svd_recs, 1):
                print(f"  {i}. {rec['title']} (Pred: {rec['predicted_rating']:.1f}, Actual: {rec['actual_rating']:.1f})")

            print(f"\n👤 User {user_id} - NCF Recommendations:")
            ncf_recs = recommender.get_user_recommendations(user_id, top_k=5, model_type='ncf')
            for i, rec in enumerate(ncf_recs, 1):
                print(f"  {i}. {rec['title']} (Pred: {rec['predicted_rating']:.1f}, Actual: {rec['actual_rating']:.1f})")

        except Exception as e:
            print(f"  Error for user {user_id}: {e}")

    # Save models
    print(f"\n💾 Saving models...")
    os.makedirs('results/models', exist_ok=True)
    recommender.save_models('results/models/collaborative_filtering')

    print(f"\n✅ Collaborative Filtering Models Built Successfully!")
    print(f"📊 Final Statistics:")
    print(f"  • Users: {evaluation['dataset_stats']['num_users']:,}")
    print(f"  • Movies: {evaluation['dataset_stats']['num_movies']:,}")
    print(f"  • Test samples: {evaluation['dataset_stats']['test_samples']:,}")
    print(f"  • Best model: {'NCF' if evaluation['ncf']['rmse'] < evaluation['svd']['rmse'] else 'SVD'}")


if __name__ == "__main__":
    main()