# Collaborative Filtering — Methods Plan

## Overview

Collaborative Filtering (CF) predicts user preferences based on patterns from other users. This is the **mandatory multi-method** section — we implement **4+ distinct methods**.

## Method 1: User-Based Collaborative Filtering

**Concept**: Find users similar to the target user, aggregate their ratings to predict unknown ratings.

**Steps**:
1. Build user-item rating matrix
2. Compute pairwise user similarity (cosine similarity)
3. For each unseen item, predict rating as weighted average of k most similar users' ratings
4. Recommend top-N items with highest predicted ratings

**Similarity Metric**: Cosine similarity
```
sim(u, v) = (r_u · r_v) / (||r_u|| * ||r_v||)
```

**Parameters**: k = 10-30 neighbors

**When it works best**: Dense rating data, users with sufficient history

## Method 2: Item-Based Collaborative Filtering

**Concept**: Find items similar to items the user liked, recommend based on item similarity.

**Steps**:
1. Build item-item similarity matrix
2. For each item the user rated highly, find similar items
3. Predict rating as weighted sum of user's ratings on similar items
4. Recommend top-N

**Similarity Metric**: Adjusted cosine similarity (normalize user bias)

**Parameters**: k = 10-20 similar items

**When it works best**: More stable than user-based, scales better for large item catalogs

## Method 3: SVD (Matrix Factorization)

**Concept**: Decompose the user-item matrix into latent factors representing hidden features.

**Steps**:
1. Factorize rating matrix R ≈ P × Q^T (P = user-factors, Q = item-factors)
2. Predict rating: r̂_ui = μ + b_u + b_i + p_u · q_i^T
3. Use SGD to minimize RMSE
4. Recommend top-N by predicted ratings

**Parameters**: n_factors = 10-50, learning_rate = 0.01, regularization = 0.02

**When it works best**: Sparse data, captures latent patterns, best overall accuracy

## Method 4: KNN-Based Collaborative Filtering

**Concept**: Use K-Nearest Neighbors to find the neighborhood, then predict.

**Steps**:
1. Find k nearest neighbors (users or items) using distance metric
2. Predict rating as mean of neighbors' ratings
3. Weight by inverse distance

**Variants**: 
- KNN-User: neighbor users
- KNN-Item: neighbor items  
- KNN-with-means: subtract user mean before aggregation

**Parameters**: k = 5-20, metric = cosine/euclidean

## Method 5 (Bonus): Slope One

**Concept**: Simple linear prediction based on average rating differences between items.

**Steps**:
1. Compute average rating deviation: dev(i,j) = avg(r_ui - r_uj)
2. Predict: r̂_ui = (1/|R_i|) * Σ(r_uj + dev(j,i))

**When it works best**: Fast, simple, reasonable accuracy with dense data

## Implementation (`recommender/collaborative.py`)

```python
class CollaborativeFiltering:
    def __init__(self, ratings_df):
        self.ratings = ratings_df
        self.user_item_matrix = None
        self.item_similarity = None
        self.user_similarity = None
    
    def user_based_cf(self, user_id, n_recommendations=10, k=20): ...
    def item_based_cf(self, user_id, n_recommendations=10, k=15): ...
    def svd(self, user_id, n_recommendations=10, n_factors=20): ...
    def knn_cf(self, user_id, n_recommendations=10, k=10): ...
    def slope_one(self, user_id, n_recommendations=10): ...
```

## Expected Differences Between Methods

| Method | Sparsity Handling | Accuracy | Scalability | Interpretability |
|--------|-------------------|----------|-------------|------------------|
| User-Based CF | Poor | Good | Poor | Good |
| Item-Based CF | Medium | Good | Medium | Good |
| SVD | Excellent | Excellent | Excellent | Poor (latent) |
| KNN | Poor | Medium | Poor | Good |
| Slope One | Medium | Medium | Excellent | Good |
