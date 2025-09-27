#!/usr/bin/env python3
"""
Quick demonstration of collaborative filtering models
"""

import pandas as pd
import numpy as np
from src.collaborative_filtering import CollaborativeFilteringRecommender
import torch

# Configure for quick demo
demo_config = {
    # SVD parameters
    'svd_components': 25,
    'svd_random_state': 42,

    # Neural CF parameters (reduced for speed)
    'embedding_dim': 32,
    'hidden_dims': [64, 32],
    'learning_rate': 0.001,
    'batch_size': 2048,
    'epochs': 20,
    'early_stopping': 3,

    # Data parameters (reduced for demo)
    'min_user_ratings': 3,
    'min_movie_ratings': 2,
    'test_size': 0.2,
}

def main():
    print("🤝 Collaborative Filtering Demo")
    print("=" * 40)

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")

    # Initialize with demo config
    recommender = CollaborativeFilteringRecommender(config=demo_config)

    # Load data
    print("\n📊 Loading data...")
    recommender.load_data('data/processed_movies.csv.gz')

    # Prepare data
    print("📈 Preparing collaborative data...")
    X_train, X_test, y_train, y_test = recommender.prepare_collaborative_data()

    # Train SVD model
    print("🔧 Training SVD model...")
    recommender.train_svd_model(X_train, y_train)

    # Train NCF model (reduced epochs for demo)
    print("🧠 Training Neural CF model...")
    recommender.train_ncf_model(X_train, X_test, y_train, y_test)

    # Evaluate
    print("📊 Evaluating models...")
    evaluation = recommender.evaluate_models(X_test, y_test)

    print(f"\n📈 Results:")
    print(f"SVD - RMSE: {evaluation['svd']['rmse']:.4f}")
    print(f"NCF - RMSE: {evaluation['ncf']['rmse']:.4f}")

    # Sample recommendations
    print(f"\n🎯 Sample Recommendations (User 0):")
    try:
        recs = recommender.get_user_recommendations(0, top_k=3, model_type='ncf')
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec['title']} (Pred: {rec['predicted_rating']:.1f})")
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n✅ Demo completed successfully!")
    print(f"📊 Stats: {evaluation['dataset_stats']['num_users']} users, {evaluation['dataset_stats']['num_movies']} movies")

if __name__ == "__main__":
    main()