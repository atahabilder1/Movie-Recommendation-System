#!/usr/bin/env python3
"""
Execute EDA analysis and generate professional visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import ast
from collections import Counter
import os

warnings.filterwarnings('ignore')

# Configure matplotlib for high-quality output
plt.style.use('default')  # Use default style instead of seaborn-v0_8
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

print("📊 Starting Professional EDA Analysis")
print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Load processed dataset
print("\n🔄 Loading preprocessed movie dataset...")
movies_df = pd.read_csv('data/processed_movies.csv.gz', compression='gzip')
print(f"✅ Dataset loaded: {movies_df.shape}")

# Create results directory
os.makedirs('results/figures', exist_ok=True)

def safe_literal_eval(val):
    """Safely evaluate string representations of lists"""
    if pd.isna(val) or val == '':
        return []
    try:
        if isinstance(val, str):
            return ast.literal_eval(val)
        elif isinstance(val, list):
            return val
        else:
            return []
    except:
        return [genre.strip().strip("'\"") for genre in str(val).split(',') if genre.strip()]

print("\n📊 Generating Temporal Analysis...")
# 1. Temporal Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Temporal Analysis of Movie Releases', fontsize=16, fontweight='bold')

# Movies by release year
yearly_counts = movies_df['release_year'].value_counts().sort_index()
axes[0, 0].plot(yearly_counts.index, yearly_counts.values, linewidth=2, color='steelblue')
axes[0, 0].set_title('Movies Released by Year')
axes[0, 0].set_xlabel('Release Year')
axes[0, 0].set_ylabel('Number of Movies')
axes[0, 0].grid(True, alpha=0.3)

# Movies by decade
decade_counts = movies_df['release_decade'].value_counts().sort_index()
axes[0, 1].bar(decade_counts.index, decade_counts.values, color='lightcoral', alpha=0.8)
axes[0, 1].set_title('Movies by Decade')
axes[0, 1].set_xlabel('Decade')
axes[0, 1].set_ylabel('Number of Movies')
axes[0, 1].tick_params(axis='x', rotation=45)

# Movies by release month
month_counts = movies_df['release_month'].value_counts().sort_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
axes[1, 0].bar(range(1, 13), [month_counts.get(i, 0) for i in range(1, 13)],
               color='mediumseagreen', alpha=0.8)
axes[1, 0].set_title('Movie Releases by Month')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Number of Movies')
axes[1, 0].set_xticks(range(1, 13))
axes[1, 0].set_xticklabels(month_names, rotation=45)

# Movie age distribution
axes[1, 1].hist(movies_df['movie_age'], bins=30, color='plum', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Distribution of Movie Ages')
axes[1, 1].set_xlabel('Movie Age (Years)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('results/figures/eda_temporal_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("📊 Generating Rating Analysis...")
# 2. Rating Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Movie Ratings and Popularity Analysis', fontsize=16, fontweight='bold')

# Vote average distribution
axes[0, 0].hist(movies_df['vote_average'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(movies_df['vote_average'].mean(), color='red', linestyle='--',
                   label=f'Mean: {movies_df["vote_average"].mean():.2f}')
axes[0, 0].axvline(movies_df['vote_average'].median(), color='orange', linestyle='--',
                   label=f'Median: {movies_df["vote_average"].median():.2f}')
axes[0, 0].set_title('Distribution of Movie Ratings')
axes[0, 0].set_xlabel('Average Rating')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# Vote count distribution (log scale)
axes[0, 1].hist(np.log10(movies_df['vote_count']), bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Distribution of Vote Counts (Log Scale)')
axes[0, 1].set_xlabel('Log10(Vote Count)')
axes[0, 1].set_ylabel('Frequency')

# Rating vs Vote Count scatter
sample_data = movies_df.sample(n=min(5000, len(movies_df)))
axes[1, 0].scatter(sample_data['vote_count'], sample_data['vote_average'],
                   alpha=0.5, color='purple', s=20)
axes[1, 0].set_xscale('log')
axes[1, 0].set_title('Rating vs Vote Count')
axes[1, 0].set_xlabel('Vote Count (Log Scale)')
axes[1, 0].set_ylabel('Average Rating')

# Rating categories
rating_counts = movies_df['rating_category'].value_counts()
axes[1, 1].pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%',
               colors=['lightcoral', 'gold', 'lightgreen', 'mediumpurple'])
axes[1, 1].set_title('Distribution by Rating Category')

plt.tight_layout()
plt.savefig('results/figures/eda_ratings_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("📊 Generating Genre Analysis...")
# 3. Genre Analysis
all_genres = []
for genres_list in movies_df['genres']:
    genres = safe_literal_eval(genres_list)
    all_genres.extend(genres)

genre_counts = Counter(all_genres)
top_genres = dict(genre_counts.most_common(15))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Genre Analysis', fontsize=16, fontweight='bold')

# Top genres bar chart
genre_names = list(top_genres.keys())
genre_values = list(top_genres.values())
bars = axes[0, 0].bar(range(len(genre_names)), genre_values, color='steelblue', alpha=0.8)
axes[0, 0].set_title('Top 15 Movie Genres')
axes[0, 0].set_xlabel('Genres')
axes[0, 0].set_ylabel('Number of Movies')
axes[0, 0].set_xticks(range(len(genre_names)))
axes[0, 0].set_xticklabels(genre_names, rotation=45, ha='right')

# Add value labels on bars
for bar, value in zip(bars, genre_values):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{value:,}', ha='center', va='bottom', fontsize=9)

# Genre count distribution
axes[0, 1].hist(movies_df['genre_count'], bins=range(1, movies_df['genre_count'].max()+2),
                color='lightcoral', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Distribution of Genre Count per Movie')
axes[0, 1].set_xlabel('Number of Genres')
axes[0, 1].set_ylabel('Number of Movies')

# Average rating by top genres
genre_ratings = {}
for genre in genre_names[:10]:
    genre_movies = movies_df[movies_df['genres'].apply(
        lambda x: genre in safe_literal_eval(x)
    )]
    if len(genre_movies) > 0:
        genre_ratings[genre] = genre_movies['vote_average'].mean()

sorted_genres = sorted(genre_ratings.items(), key=lambda x: x[1], reverse=True)
genre_names_sorted = [item[0] for item in sorted_genres]
rating_values = [item[1] for item in sorted_genres]

bars = axes[1, 0].bar(range(len(genre_names_sorted)), rating_values,
                      color='mediumseagreen', alpha=0.8)
axes[1, 0].set_title('Average Rating by Genre (Top 10)')
axes[1, 0].set_xlabel('Genres')
axes[1, 0].set_ylabel('Average Rating')
axes[1, 0].set_xticks(range(len(genre_names_sorted)))
axes[1, 0].set_xticklabels(genre_names_sorted, rotation=45, ha='right')
axes[1, 0].set_ylim(5.5, max(rating_values) + 0.2)

# Add value labels
for bar, value in zip(bars, rating_values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)

# Genre popularity over decades
top_5_genres = genre_names[:5]
decades = sorted(movies_df['release_decade'].unique())

for genre in top_5_genres:
    decade_counts = []
    for decade in decades:
        decade_movies = movies_df[movies_df['release_decade'] == decade]
        genre_count = sum(decade_movies['genres'].apply(
            lambda x: genre in safe_literal_eval(x)
        ))
        decade_counts.append(genre_count)

    axes[1, 1].plot(decades, decade_counts, marker='o', label=genre, linewidth=2)

axes[1, 1].set_title('Genre Popularity Trends by Decade')
axes[1, 1].set_xlabel('Decade')
axes[1, 1].set_ylabel('Number of Movies')
axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/eda_genre_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("📊 Generating Content Analysis...")
# 4. Content Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Content Analysis for Recommendation Features', fontsize=16, fontweight='bold')

# Overview length distribution
axes[0, 0].hist(movies_df['overview_length'], bins=50, color='lightblue', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(movies_df['overview_length'].mean(), color='red', linestyle='--',
                   label=f'Mean: {movies_df["overview_length"].mean():.0f} chars')
axes[0, 0].set_title('Movie Overview Length Distribution')
axes[0, 0].set_xlabel('Overview Length (characters)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# Word count distribution
axes[0, 1].hist(movies_df['overview_word_count'], bins=50, color='lightcyan', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(movies_df['overview_word_count'].mean(), color='red', linestyle='--',
                   label=f'Mean: {movies_df["overview_word_count"].mean():.0f} words')
axes[0, 1].set_title('Overview Word Count Distribution')
axes[0, 1].set_xlabel('Number of Words')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

# Runtime distribution
axes[1, 0].hist(movies_df['runtime'], bins=50, color='lightsalmon', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(movies_df['runtime'].mean(), color='red', linestyle='--',
                   label=f'Mean: {movies_df["runtime"].mean():.0f} min')
axes[1, 0].set_title('Movie Runtime Distribution')
axes[1, 0].set_xlabel('Runtime (minutes)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

# Keyword count distribution
axes[1, 1].hist(movies_df['keyword_count'], bins=30, color='lightsteelblue', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(movies_df['keyword_count'].mean(), color='red', linestyle='--',
                   label=f'Mean: {movies_df["keyword_count"].mean():.1f} keywords')
axes[1, 1].set_title('Keyword Count Distribution')
axes[1, 1].set_xlabel('Number of Keywords')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('results/figures/eda_content_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("📊 Generating Correlation Analysis...")
# 5. Correlation Analysis
numerical_features = [
    'vote_average', 'vote_count', 'popularity', 'runtime',
    'overview_length', 'overview_word_count', 'genre_count',
    'keyword_count', 'movie_age', 'weighted_rating'
]

# Add financial features if available
if 'budget' in movies_df.columns and movies_df['budget'].sum() > 0:
    numerical_features.extend(['budget', 'revenue', 'profit', 'roi'])

available_features = [col for col in numerical_features if col in movies_df.columns]
correlation_df = movies_df[available_features].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_df, dtype=bool))
sns.heatmap(correlation_df, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Feature Correlation Matrix for Recommendation System', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/eda_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ EDA Analysis Complete!")
print("📊 Generated professional visualizations:")
print("  • Temporal Analysis (eda_temporal_analysis.png)")
print("  • Rating Analysis (eda_ratings_analysis.png)")
print("  • Genre Analysis (eda_genre_analysis.png)")
print("  • Content Analysis (eda_content_analysis.png)")
print("  • Correlation Matrix (eda_correlation_matrix.png)")

print(f"\n🎯 Dataset Summary:")
print(f"  • {len(movies_df):,} high-quality movies")
print(f"  • {len(movies_df.columns)} engineered features")
print(f"  • {movies_df['release_year'].min()}-{movies_df['release_year'].max()} time range")
print(f"  • {movies_df['vote_average'].mean():.2f} average rating")
print(f"  • {len(genre_counts)} unique genres")

print("\n🚀 Ready for Phase 3: Model Development!")