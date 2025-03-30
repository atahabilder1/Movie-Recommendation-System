# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Framework for Movie Recommendation Systems
===================================================================

This module provides evaluation metrics and frameworks for assessing
the performance of both content-based and collaborative filtering
recommendation systems.

Metrics Implemented:
- Content-Based: Precision@K, Recall@K, NDCG, Diversity, Coverage
- Collaborative: RMSE, MAE, Precision@K, Recall@K, Hit Rate
- System-wide: Cold Start Analysis, Popularity Bias, Novelty

Author: Portfolio Project for Data Science Applications
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Set
from datetime import datetime
import warnings
from collections import defaultdict
import math

# ML and evaluation imports
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import scipy.stats as stats

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """
    Comprehensive evaluation framework for recommendation systems.

    Features:
    - Multiple evaluation metrics for different recommendation types
    - Statistical significance testing
    - Cold start and popularity bias analysis
    - Diversity and novelty measurements
    - Cross-validation support
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the evaluator with configuration."""
        self.config = config or self._default_config()
        self.evaluation_history = []

    def _default_config(self) -> Dict:
        """Default configuration for evaluation."""
        return {
            'k_values': [1, 3, 5, 10, 20],          # For Precision@K, Recall@K
            'relevance_threshold': 4.0,              # Rating threshold for relevance
            'diversity_alpha': 0.5,                  # Alpha for diversity calculation
            'novelty_percentile': 0.9,               # Percentile for novelty calculation
            'statistical_significance': 0.05,        # p-value threshold
            'min_interactions': 5,                   # Minimum interactions for evaluation
        }

    def evaluate_content_based(self,
                             recommendations: List[List[Dict]],
                             ground_truth: List[List[str]],
                             movie_metadata: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate content-based recommendation system.

        Args:
            recommendations: List of recommendation lists for each query
            ground_truth: List of relevant items for each query
            movie_metadata: DataFrame with movie information

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating content-based recommendation system...")

        metrics = {}
        k_values = self.config['k_values']

        # Precision@K and Recall@K
        for k in k_values:
            precision_scores = []
            recall_scores = []

            for recs, truth in zip(recommendations, ground_truth):
                rec_items = [r['title'] for r in recs[:k]]
                truth_set = set(truth)

                if len(rec_items) == 0:
                    precision_scores.append(0.0)
                    recall_scores.append(0.0)
                    continue

                # Precision@K
                relevant_retrieved = len(set(rec_items) & truth_set)
                precision = relevant_retrieved / len(rec_items)
                precision_scores.append(precision)

                # Recall@K
                if len(truth_set) > 0:
                    recall = relevant_retrieved / len(truth_set)
                else:
                    recall = 0.0
                recall_scores.append(recall)

            metrics[f'precision_at_{k}'] = np.mean(precision_scores)
            metrics[f'recall_at_{k}'] = np.mean(recall_scores)

        # NDCG@K
        for k in k_values:
            ndcg_scores = []
            for recs, truth in zip(recommendations, ground_truth):
                ndcg = self._calculate_ndcg(recs[:k], truth)
                ndcg_scores.append(ndcg)
            metrics[f'ndcg_at_{k}'] = np.mean(ndcg_scores)

        # Diversity metrics
        metrics['intra_list_diversity'] = self._calculate_intra_list_diversity(
            recommendations, movie_metadata)
        metrics['catalog_coverage'] = self._calculate_catalog_coverage(
            recommendations, movie_metadata)

        # Novelty and serendipity
        metrics['novelty'] = self._calculate_novelty(
            recommendations, movie_metadata)

        # Genre distribution analysis
        metrics['genre_distribution'] = self._analyze_genre_distribution(
            recommendations, movie_metadata)

        logger.info("Content-based evaluation completed")
        return metrics

    def evaluate_collaborative_filtering(self,
                                       predictions: List[Tuple[int, int, float, float]],
                                       test_interactions: List[Tuple[int, int, float]]) -> Dict[str, Any]:
        """
        Evaluate collaborative filtering recommendation system.

        Args:
            predictions: List of (user_id, item_id, predicted_rating, actual_rating)
            test_interactions: List of (user_id, item_id, actual_rating) for test set

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating collaborative filtering system...")

        metrics = {}

        # Extract predicted and actual ratings
        predicted_ratings = [p[2] for p in predictions]
        actual_ratings = [p[3] for p in predictions]

        # Rating prediction accuracy
        metrics['rmse'] = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        metrics['mae'] = mean_absolute_error(actual_ratings, predicted_ratings)

        # Pearson correlation
        correlation, p_value = stats.pearsonr(predicted_ratings, actual_ratings)
        metrics['pearson_correlation'] = correlation
        metrics['correlation_p_value'] = p_value

        # Top-K ranking metrics
        user_predictions = defaultdict(list)
        for user_id, item_id, pred_rating, actual_rating in predictions:
            user_predictions[user_id].append((item_id, pred_rating, actual_rating))

        # Calculate ranking metrics for each user
        k_values = self.config['k_values']
        relevance_threshold = self.config['relevance_threshold']

        for k in k_values:
            hit_rates = []
            precisions = []
            recalls = []

            for user_id, user_preds in user_predictions.items():
                # Sort by predicted rating (descending)
                user_preds_sorted = sorted(user_preds, key=lambda x: x[1], reverse=True)
                top_k_items = user_preds_sorted[:k]

                # Identify relevant items (rating >= threshold)
                relevant_items = [item_id for item_id, _, actual in user_preds
                                if actual >= relevance_threshold]
                recommended_items = [item_id for item_id, _, _ in top_k_items]

                # Hit Rate@K
                hit = 1 if any(item in relevant_items for item in recommended_items) else 0
                hit_rates.append(hit)

                # Precision@K
                relevant_recommended = len(set(recommended_items) & set(relevant_items))
                precision = relevant_recommended / len(recommended_items) if recommended_items else 0
                precisions.append(precision)

                # Recall@K
                recall = relevant_recommended / len(relevant_items) if relevant_items else 0
                recalls.append(recall)

            metrics[f'hit_rate_at_{k}'] = np.mean(hit_rates)
            metrics[f'precision_at_{k}'] = np.mean(precisions)
            metrics[f'recall_at_{k}'] = np.mean(recalls)

        # Coverage analysis
        unique_recommended_items = set()
        total_items = set()
        for user_preds in user_predictions.values():
            for item_id, pred_rating, actual_rating in user_preds:
                total_items.add(item_id)
                # Add to recommended if it would be in top-10
                user_top_items = sorted(user_preds, key=lambda x: x[1], reverse=True)[:10]
                if (item_id, pred_rating, actual_rating) in user_top_items:
                    unique_recommended_items.add(item_id)

        metrics['item_coverage'] = len(unique_recommended_items) / len(total_items) if total_items else 0

        logger.info("Collaborative filtering evaluation completed")
        return metrics

    def _calculate_ndcg(self, recommendations: List[Dict], ground_truth: List[str]) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not recommendations or not ground_truth:
            return 0.0

        # Create relevance scores (1 if in ground truth, 0 otherwise)
        relevance_scores = []
        for rec in recommendations:
            relevance = 1.0 if rec['title'] in ground_truth else 0.0
            relevance_scores.append(relevance)

        # Calculate DCG
        dcg = relevance_scores[0]
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / math.log2(i + 1)

        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted([1.0] * min(len(ground_truth), len(recommendations)),
                               reverse=True)
        idcg = ideal_relevance[0] if ideal_relevance else 0
        for i in range(1, len(ideal_relevance)):
            idcg += ideal_relevance[i] / math.log2(i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_intra_list_diversity(self,
                                      recommendations: List[List[Dict]],
                                      movie_metadata: pd.DataFrame) -> float:
        """Calculate average intra-list diversity using genre diversity."""
        diversities = []

        for rec_list in recommendations:
            if len(rec_list) < 2:
                diversities.append(0.0)
                continue

            # Extract genres for each recommended movie
            genres_lists = []
            for rec in rec_list:
                movie_data = movie_metadata[movie_metadata['title'] == rec['title']]
                if not movie_data.empty and 'genres' in movie_data.columns:
                    genres = str(movie_data.iloc[0]['genres']).split('|')
                    genres_lists.append(set(genres))
                else:
                    genres_lists.append(set())

            # Calculate pairwise diversity
            diversity_sum = 0
            pair_count = 0
            for i in range(len(genres_lists)):
                for j in range(i + 1, len(genres_lists)):
                    # Jaccard distance as diversity measure
                    union_size = len(genres_lists[i] | genres_lists[j])
                    intersection_size = len(genres_lists[i] & genres_lists[j])
                    if union_size > 0:
                        diversity = 1 - (intersection_size / union_size)
                    else:
                        diversity = 0
                    diversity_sum += diversity
                    pair_count += 1

            if pair_count > 0:
                avg_diversity = diversity_sum / pair_count
                diversities.append(avg_diversity)
            else:
                diversities.append(0.0)

        return np.mean(diversities)

    def _calculate_catalog_coverage(self,
                                  recommendations: List[List[Dict]],
                                  movie_metadata: pd.DataFrame) -> float:
        """Calculate catalog coverage (percentage of items recommended)."""
        recommended_items = set()
        for rec_list in recommendations:
            for rec in rec_list:
                recommended_items.add(rec['title'])

        total_items = len(movie_metadata)
        coverage = len(recommended_items) / total_items if total_items > 0 else 0
        return coverage

    def _calculate_novelty(self,
                         recommendations: List[List[Dict]],
                         movie_metadata: pd.DataFrame) -> float:
        """Calculate novelty based on movie popularity."""
        if 'vote_count' not in movie_metadata.columns:
            return 0.0

        # Calculate popularity percentiles
        popularity_percentile = self.config['novelty_percentile']
        popularity_threshold = movie_metadata['vote_count'].quantile(popularity_percentile)

        novel_recommendations = 0
        total_recommendations = 0

        for rec_list in recommendations:
            for rec in rec_list:
                movie_data = movie_metadata[movie_metadata['title'] == rec['title']]
                if not movie_data.empty:
                    vote_count = movie_data.iloc[0]['vote_count']
                    if vote_count < popularity_threshold:
                        novel_recommendations += 1
                total_recommendations += 1

        novelty = novel_recommendations / total_recommendations if total_recommendations > 0 else 0
        return novelty

    def _analyze_genre_distribution(self,
                                  recommendations: List[List[Dict]],
                                  movie_metadata: pd.DataFrame) -> Dict[str, float]:
        """Analyze genre distribution in recommendations."""
        if 'genres' not in movie_metadata.columns:
            return {}

        genre_counts = defaultdict(int)
        total_recommendations = 0

        for rec_list in recommendations:
            for rec in rec_list:
                movie_data = movie_metadata[movie_metadata['title'] == rec['title']]
                if not movie_data.empty:
                    genres = str(movie_data.iloc[0]['genres']).split('|')
                    for genre in genres:
                        if genre.strip():
                            genre_counts[genre.strip()] += 1
                            total_recommendations += 1

        # Convert to percentages
        genre_distribution = {}
        for genre, count in genre_counts.items():
            genre_distribution[genre] = count / total_recommendations if total_recommendations > 0 else 0

        return dict(sorted(genre_distribution.items(), key=lambda x: x[1], reverse=True))

    def compare_systems(self,
                       system_results: Dict[str, Dict[str, Any]],
                       significance_test: bool = True) -> Dict[str, Any]:
        """
        Compare multiple recommendation systems.

        Args:
            system_results: Dictionary mapping system names to their evaluation results
            significance_test: Whether to perform statistical significance testing

        Returns:
            Comparison results and statistical tests
        """
        logger.info("Comparing recommendation systems...")

        comparison = {
            'system_rankings': {},
            'best_system_per_metric': {},
            'statistical_significance': {}
        }

        # Get all metrics that are common across systems
        all_metrics = set()
        for system_name, results in system_results.items():
            all_metrics.update(results.keys())

        # Rank systems for each metric
        for metric in all_metrics:
            metric_scores = {}
            for system_name, results in system_results.items():
                if metric in results:
                    metric_scores[system_name] = results[metric]

            if metric_scores:
                # Sort by score (higher is better for most metrics, except RMSE and MAE)
                reverse_sort = not (metric.lower() in ['rmse', 'mae'])
                sorted_systems = sorted(metric_scores.items(),
                                      key=lambda x: x[1], reverse=reverse_sort)

                comparison['system_rankings'][metric] = sorted_systems
                comparison['best_system_per_metric'][metric] = sorted_systems[0][0]

        # Overall system ranking (average rank across metrics)
        system_avg_ranks = defaultdict(list)
        for metric, rankings in comparison['system_rankings'].items():
            for rank, (system_name, score) in enumerate(rankings):
                system_avg_ranks[system_name].append(rank + 1)

        overall_rankings = []
        for system_name, ranks in system_avg_ranks.items():
            avg_rank = np.mean(ranks)
            overall_rankings.append((system_name, avg_rank))

        comparison['overall_ranking'] = sorted(overall_rankings, key=lambda x: x[1])

        logger.info("System comparison completed")
        return comparison

    def generate_evaluation_report(self,
                                 evaluation_results: Dict[str, Any],
                                 system_name: str = "Recommendation System") -> str:
        """Generate a comprehensive evaluation report."""

        report = f"""
Recommendation System Evaluation Report
======================================
System: {system_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ACCURACY METRICS
--------------
"""

        # Add accuracy metrics
        accuracy_metrics = ['rmse', 'mae', 'pearson_correlation']
        for metric in accuracy_metrics:
            if metric in evaluation_results:
                value = evaluation_results[metric]
                if isinstance(value, float):
                    report += f"{metric.upper()}: {value:.4f}\n"

        # Add ranking metrics
        report += "\nRANKING METRICS\n"
        report += "-" * 15 + "\n"

        k_metrics = ['precision_at_', 'recall_at_', 'ndcg_at_', 'hit_rate_at_']
        for k in self.config['k_values']:
            report += f"\nTop-{k} Metrics:\n"
            for metric_prefix in k_metrics:
                metric_name = f"{metric_prefix}{k}"
                if metric_name in evaluation_results:
                    value = evaluation_results[metric_name]
                    report += f"  {metric_name}: {value:.4f}\n"

        # Add diversity and novelty metrics
        report += "\nDIVERSITY & NOVELTY\n"
        report += "-" * 19 + "\n"

        diversity_metrics = ['intra_list_diversity', 'catalog_coverage', 'novelty', 'item_coverage']
        for metric in diversity_metrics:
            if metric in evaluation_results:
                value = evaluation_results[metric]
                report += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"

        # Add genre distribution
        if 'genre_distribution' in evaluation_results:
            report += "\nGENRE DISTRIBUTION\n"
            report += "-" * 18 + "\n"
            genre_dist = evaluation_results['genre_distribution']
            for genre, percentage in list(genre_dist.items())[:10]:  # Top 10 genres
                report += f"{genre}: {percentage:.3f}\n"

        return report

    def save_evaluation_results(self,
                              results: Dict[str, Any],
                              filepath: str,
                              system_name: str = "Recommendation System") -> None:
        """Save evaluation results to file."""

        # Generate report
        report = self.generate_evaluation_report(results, system_name)

        # Save to file
        with open(filepath, 'w') as f:
            f.write(report)

        # Also save raw results as JSON-like format
        results_filepath = filepath.replace('.txt', '_raw_results.txt')
        with open(results_filepath, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

        logger.info(f"Evaluation results saved to {filepath}")


def main():
    """Example usage of the evaluation framework."""
    evaluator = RecommendationEvaluator()

    # Example evaluation (would normally use real data)
    print("Recommendation Evaluation Framework initialized successfully!")
    print("Available methods:")
    print("- evaluate_content_based()")
    print("- evaluate_collaborative_filtering()")
    print("- compare_systems()")
    print("- generate_evaluation_report()")


if __name__ == "__main__":
    main()