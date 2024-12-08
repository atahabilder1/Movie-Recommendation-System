# Results Directory

This directory contains all outputs from the movie recommendation system experiments.

## Structure

- **`figures/`** - Visualizations and plots (committed to git for README)
  - `eda_*.png` - Exploratory data analysis plots
  - `similarity_*.png` - Similarity matrix visualizations
  - `recommendations_*.png` - Recommendation result plots
  - `performance_*.png` - Model performance metrics

- **`metrics/`** - Performance metrics and evaluation results (ignored by git)
  - `evaluation_metrics.json` - Model performance scores
  - `recommendation_results.csv` - Sample recommendations

- **`models/`** - Saved models and similarity matrices (ignored by git)
  - `tfidf_vectorizer.pkl` - Trained TF-IDF vectorizer
  - `similarity_matrix.pkl` - Computed similarity matrix

## Naming Convention

### Figures
- `eda_genre_distribution.png` - Genre distribution bar chart
- `eda_description_wordcloud.png` - Word cloud of descriptions
- `similarity_heatmap.png` - Cosine similarity heatmap
- `recommendations_sample_results.png` - Sample recommendation results
- `performance_metrics.png` - Model performance visualization

### Metrics Files
- `evaluation_metrics.json` - Overall system performance
- `recommendation_results.csv` - Detailed recommendation outputs
- `similarity_scores.csv` - Similarity computation results

All figures in the `figures/` directory will be used in the main README to showcase project results.