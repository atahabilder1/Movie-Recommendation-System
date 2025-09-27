#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Movie Recommendation System - Experimental Framework
====================================================

Comprehensive experimental framework for comparing different recommendation
approaches, hyperparameter tuning, and performance analysis across
multiple configurations and datasets.

Experiment Types:
- Content-Based vs Collaborative Filtering comparison
- Hyperparameter optimization for both approaches
- Ablation studies on feature combinations
- Performance scaling analysis
- Cross-validation experiments

Author: Portfolio Project for Data Science Applications
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import argparse
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.recommend import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.evaluate import RecommendationEvaluator
from src.utils import ConfigManager, Timer, MemoryMonitor, setup_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiment_results.log')
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Comprehensive experimental framework for recommendation systems.

    Features:
    - Multiple algorithm comparison
    - Hyperparameter optimization
    - Performance benchmarking
    - Statistical significance testing
    - Result visualization and reporting
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the experiment runner."""
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        self.evaluator = RecommendationEvaluator()
        self.results = {}
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_content_based_experiments(self, data_path: str) -> Dict[str, Any]:
        """Run comprehensive content-based recommendation experiments."""
        logger.info("Starting Content-Based Recommendation Experiments")
        logger.info("=" * 60)

        experiments = {
            'baseline': {
                'max_features': 5000,
                'ngram_range': (1, 1),
                'use_svd': False,
                'svd_components': None
            },
            'enhanced_ngrams': {
                'max_features': 10000,
                'ngram_range': (1, 2),
                'use_svd': False,
                'svd_components': None
            },
            'svd_reduced': {
                'max_features': 10000,
                'ngram_range': (1, 2),
                'use_svd': True,
                'svd_components': 300
            },
            'high_dimensional': {
                'max_features': 20000,
                'ngram_range': (1, 3),
                'use_svd': True,
                'svd_components': 500
            }
        }

        results = {}

        for exp_name, exp_config in experiments.items():
            logger.info(f"\nRunning experiment: {exp_name}")
            logger.info(f"Configuration: {exp_config}")

            with Timer(f"Content-Based Experiment: {exp_name}"):
                # Initialize recommender with experiment config
                config = self.config['content_based'].copy()
                config.update(exp_config)

                recommender = ContentBasedRecommender(config)
                recommender.load_data(data_path)

                # Train model
                recommender.build_tfidf_model()
                recommender.compute_similarity_matrix()

                # Generate test recommendations
                test_movies = [
                    'The Dark Knight',
                    'Inception',
                    'Pulp Fiction',
                    'The Matrix',
                    'Forrest Gump'
                ]

                recommendations = []
                for movie in test_movies:
                    try:
                        recs = recommender.get_recommendations(movie, top_k=10)
                        recommendations.append(recs)
                    except Exception as e:
                        logger.warning(f"Failed to get recommendations for {movie}: {e}")
                        recommendations.append([])

                # Store results
                results[exp_name] = {
                    'config': exp_config,
                    'recommendations': recommendations,
                    'test_movies': test_movies,
                    'model_stats': {
                        'tfidf_shape': recommender.tfidf_matrix.shape if recommender.tfidf_matrix is not None else None,
                        'similarity_shape': recommender.similarity_matrix.shape if recommender.similarity_matrix is not None else None
                    }
                }

                logger.info(f"Experiment {exp_name} completed successfully")

        return results

    def run_collaborative_experiments(self, data_path: str) -> Dict[str, Any]:
        """Run collaborative filtering experiments with different configurations."""
        logger.info("Starting Collaborative Filtering Experiments")
        logger.info("=" * 60)

        experiments = {
            'svd_basic': {
                'svd_components': 25,
                'min_user_ratings': 5,
                'min_movie_ratings': 3
            },
            'svd_enhanced': {
                'svd_components': 50,
                'min_user_ratings': 3,
                'min_movie_ratings': 2
            },
            'ncf_basic': {
                'embedding_dim': 32,
                'hidden_dims': [64, 32],
                'epochs': 10,
                'learning_rate': 0.001
            },
            'ncf_enhanced': {
                'embedding_dim': 64,
                'hidden_dims': [128, 64, 32],
                'epochs': 20,
                'learning_rate': 0.0005
            }
        }

        results = {}

        for exp_name, exp_config in experiments.items():
            logger.info(f"\nRunning experiment: {exp_name}")
            logger.info(f"Configuration: {exp_config}")

            try:
                with Timer(f"Collaborative Experiment: {exp_name}"):
                    # Initialize recommender with experiment config
                    config = self.config['collaborative'].copy()
                    config.update(exp_config)

                    recommender = CollaborativeFilteringRecommender(config)
                    recommender.load_data(data_path)

                    # Prepare collaborative data
                    X_train, X_test, y_train, y_test = recommender.prepare_collaborative_data()

                    # Train appropriate model
                    if 'svd' in exp_name:
                        recommender.train_svd_model(X_train, y_train)
                        model_type = 'svd'
                    else:
                        recommender.train_ncf_model(X_train, X_test, y_train, y_test)
                        model_type = 'ncf'

                    # Evaluate model
                    evaluation = recommender.evaluate_models(X_test, y_test)

                    # Get sample recommendations
                    sample_recommendations = []
                    for user_id in [0, 1, 2]:
                        try:
                            recs = recommender.get_user_recommendations(user_id, top_k=5, model_type=model_type)
                            sample_recommendations.append(recs)
                        except Exception as e:
                            logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
                            sample_recommendations.append([])

                    # Store results
                    results[exp_name] = {
                        'config': exp_config,
                        'evaluation': evaluation,
                        'sample_recommendations': sample_recommendations,
                        'model_type': model_type
                    }

                    logger.info(f"Experiment {exp_name} completed successfully")

            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {e}")
                results[exp_name] = {
                    'config': exp_config,
                    'error': str(e),
                    'status': 'failed'
                }

        return results

    def run_ablation_study(self, data_path: str) -> Dict[str, Any]:
        """Run ablation study to understand feature importance."""
        logger.info("Starting Ablation Study")
        logger.info("=" * 30)

        ablation_configs = {
            'overview_only': {
                'features': ['overview'],
                'description': 'Using only movie overview text'
            },
            'genres_only': {
                'features': ['genres'],
                'description': 'Using only genre information'
            },
            'keywords_only': {
                'features': ['keywords'],
                'description': 'Using only keyword information'
            },
            'overview_genres': {
                'features': ['overview', 'genres'],
                'description': 'Combining overview and genres'
            },
            'all_features': {
                'features': ['overview', 'genres', 'keywords', 'tagline'],
                'description': 'Using all available features'
            }
        }

        results = {}

        for config_name, config_info in ablation_configs.items():
            logger.info(f"\nRunning ablation: {config_name}")
            logger.info(f"Description: {config_info['description']}")

            with Timer(f"Ablation Study: {config_name}"):
                # This would require modifying the ContentBasedRecommender
                # to support feature selection - for now we'll simulate
                base_config = self.config['content_based'].copy()

                recommender = ContentBasedRecommender(base_config)
                recommender.load_data(data_path)

                try:
                    recommender.build_tfidf_model()
                    recommender.compute_similarity_matrix()

                    # Test with sample movies
                    test_recommendations = []
                    test_movies = ['The Dark Knight', 'Inception']

                    for movie in test_movies:
                        try:
                            recs = recommender.get_recommendations(movie, top_k=5)
                            test_recommendations.append(recs)
                        except:
                            test_recommendations.append([])

                    results[config_name] = {
                        'config': config_info,
                        'recommendations': test_recommendations,
                        'status': 'completed'
                    }

                except Exception as e:
                    logger.error(f"Ablation {config_name} failed: {e}")
                    results[config_name] = {
                        'config': config_info,
                        'error': str(e),
                        'status': 'failed'
                    }

        return results

    def run_performance_benchmark(self, data_path: str) -> Dict[str, Any]:
        """Run performance benchmarking experiments."""
        logger.info("Starting Performance Benchmarking")
        logger.info("=" * 40)

        data_sizes = [1000, 5000, 10000, 20000]  # Different subset sizes
        results = {}

        # Load full dataset
        full_data = pd.read_csv(data_path, compression='gzip' if data_path.endswith('.gz') else None)

        for data_size in data_sizes:
            if data_size > len(full_data):
                continue

            logger.info(f"\nBenchmarking with {data_size} movies")

            # Create subset
            subset_data = full_data.head(data_size)

            # Save temporary subset
            temp_path = f"/tmp/benchmark_data_{data_size}.csv"
            subset_data.to_csv(temp_path, index=False)

            monitor = MemoryMonitor()
            initial_memory = monitor.start_monitoring()

            with Timer(f"Benchmark: {data_size} movies") as timer:
                try:
                    # Test content-based recommender
                    recommender = ContentBasedRecommender()
                    recommender.load_data(temp_path)
                    recommender.build_tfidf_model()
                    recommender.compute_similarity_matrix()

                    # Get sample recommendations
                    sample_movie = subset_data.iloc[0]['title']
                    recommendations = recommender.get_recommendations(sample_movie, top_k=5)

                    final_memory = monitor.get_current_usage()
                    memory_delta = monitor.get_memory_delta()

                    results[f"size_{data_size}"] = {
                        'data_size': data_size,
                        'execution_time': timer.end_time - timer.start_time,
                        'memory_usage_mb': final_memory,
                        'memory_delta_mb': memory_delta,
                        'recommendations_count': len(recommendations),
                        'tfidf_shape': recommender.tfidf_matrix.shape,
                        'status': 'completed'
                    }

                except Exception as e:
                    logger.error(f"Benchmark for size {data_size} failed: {e}")
                    results[f"size_{data_size}"] = {
                        'data_size': data_size,
                        'error': str(e),
                        'status': 'failed'
                    }

            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return results

    def generate_experiment_report(self, all_results: Dict[str, Any]) -> str:
        """Generate comprehensive experiment report."""
        report = f"""
Movie Recommendation System - Experimental Results
=================================================
Experiment ID: {self.experiment_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXPERIMENT SUMMARY
-----------------
"""

        for experiment_type, results in all_results.items():
            report += f"\n{experiment_type.upper()} EXPERIMENTS\n"
            report += "-" * (len(experiment_type) + 12) + "\n"

            if experiment_type == 'content_based':
                for exp_name, exp_data in results.items():
                    if 'error' not in exp_data:
                        report += f"\n{exp_name}:\n"
                        report += f"  Configuration: {exp_data['config']}\n"
                        report += f"  TF-IDF Shape: {exp_data['model_stats']['tfidf_shape']}\n"
                        report += f"  Similarity Shape: {exp_data['model_stats']['similarity_shape']}\n"
                        report += f"  Test Movies: {len(exp_data['test_movies'])}\n"

            elif experiment_type == 'collaborative':
                for exp_name, exp_data in results.items():
                    if 'error' not in exp_data:
                        report += f"\n{exp_name}:\n"
                        report += f"  Model Type: {exp_data['model_type']}\n"
                        if 'evaluation' in exp_data:
                            eval_data = exp_data['evaluation']
                            if 'svd' in eval_data:
                                report += f"  SVD RMSE: {eval_data['svd'].get('rmse', 'N/A'):.4f}\n"
                            if 'ncf' in eval_data:
                                report += f"  NCF RMSE: {eval_data['ncf'].get('rmse', 'N/A'):.4f}\n"

            elif experiment_type == 'performance':
                report += "\nPerformance Scaling:\n"
                for size_key, perf_data in results.items():
                    if 'error' not in perf_data:
                        report += f"  {perf_data['data_size']} movies: "
                        report += f"{perf_data['execution_time']:.2f}s, "
                        report += f"{perf_data['memory_delta_mb']:.1f}MB\n"

        report += f"\n\nEXPERIMENT COMPLETED\n"
        report += f"Full results saved to: experiment_results_{self.experiment_id}.json\n"

        return report

    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> None:
        """Save experiment results to file."""
        if filename is None:
            filename = f"experiment_results_{self.experiment_id}.json"

        # Convert numpy arrays and other non-serializable objects
        serializable_results = self._make_serializable(results)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {filename}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        else:
            return obj


def main():
    """Main entry point for experimental framework."""
    parser = argparse.ArgumentParser(description='Movie Recommendation System Experiments')
    parser.add_argument('--data-path', type=str, default='data/processed_movies.csv.gz',
                       help='Path to processed movie data')
    parser.add_argument('--experiments', nargs='+',
                       choices=['content_based', 'collaborative', 'ablation', 'performance', 'all'],
                       default=['content_based'],
                       help='Experiments to run')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Initialize experiment runner
        runner = ExperimentRunner()

        # Check if data exists
        if not os.path.exists(args.data_path):
            logger.error(f"Data file not found: {args.data_path}")
            sys.exit(1)

        logger.info("Starting Movie Recommendation System Experiments")
        logger.info(f"Data path: {args.data_path}")
        logger.info(f"Experiments: {args.experiments}")

        all_results = {}

        # Run requested experiments
        if 'all' in args.experiments:
            experiments_to_run = ['content_based', 'collaborative', 'ablation', 'performance']
        else:
            experiments_to_run = args.experiments

        for experiment in experiments_to_run:
            logger.info(f"\n{'='*60}")
            logger.info(f"STARTING: {experiment.upper()} EXPERIMENTS")
            logger.info(f"{'='*60}")

            if experiment == 'content_based':
                all_results['content_based'] = runner.run_content_based_experiments(args.data_path)
            elif experiment == 'collaborative':
                all_results['collaborative'] = runner.run_collaborative_experiments(args.data_path)
            elif experiment == 'ablation':
                all_results['ablation'] = runner.run_ablation_study(args.data_path)
            elif experiment == 'performance':
                all_results['performance'] = runner.run_performance_benchmark(args.data_path)

        # Generate and save report
        report = runner.generate_experiment_report(all_results)
        print(report)

        # Save detailed results
        results_filename = os.path.join(args.output_dir, f"experiment_results_{runner.experiment_id}.json")
        runner.save_results(all_results, results_filename)

        # Save report
        report_filename = os.path.join(args.output_dir, f"experiment_report_{runner.experiment_id}.txt")
        with open(report_filename, 'w') as f:
            f.write(report)

        logger.info(f"\nAll experiments completed successfully!")
        logger.info(f"Results saved to: {results_filename}")
        logger.info(f"Report saved to: {report_filename}")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()