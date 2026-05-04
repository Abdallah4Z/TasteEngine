# Evaluation Module Plan

## Overview

A comprehensive evaluation module that computes 4+ metrics and enables comparison between CF methods and between the three approaches.

## Evaluation Techniques (6 metrics)

### 1. RMSE — Root Mean Square Error
Measures prediction accuracy (rating prediction error).

```
RMSE = sqrt( (1/N) * Σ(r_ui - r̂_ui)² )
```

**Interpretation**: Lower is better. Penalizes large errors more heavily.

### 2. MAE — Mean Absolute Error
Average absolute prediction error.

```
MAE = (1/N) * Σ |r_ui - r̂_ui|
```

**Interpretation**: Lower is better. More interpretable than RMSE.

### 3. Precision@K
Fraction of recommended items that are relevant (rating ≥ threshold).

```
Precision@K = |relevant ∩ recommended| / K
```

**Interpretation**: Higher is better. How many recommendations were good?

### 4. Recall@K
Fraction of relevant items that were recommended.

```
Recall@K = |relevant ∩ recommended| / |relevant|
```

**Interpretation**: Higher is better. How many good items did we catch?

### 5. F1-Score
Harmonic mean of precision and recall.

```
F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
```

**Interpretation**: Higher is better. Balances precision and recall.

### 6. Coverage
Percentage of catalog items that can be recommended.

```
Coverage = |recommendable_items| / |total_items|
```

**Interpretation**: Higher is better. Does the system recommend everything or just popular items?

## Comparison Dimensions

### A) Between CF Methods
Compare all 4-5 CF methods on all metrics.

| Method | RMSE | MAE | Prec@5 | Rec@5 | F1@5 | Coverage |
|--------|------|-----|--------|-------|------|----------|
| User-Based | 0.92 | 0.71 | 0.60 | 0.45 | 0.51 | 0.40 |
| Item-Based | 0.88 | 0.68 | 0.65 | 0.50 | 0.57 | 0.55 |
| SVD | 0.81 | 0.62 | 0.72 | 0.58 | 0.64 | 0.78 |
| KNN | 0.95 | 0.74 | 0.55 | 0.40 | 0.46 | 0.35 |
| Slope One | 0.90 | 0.70 | 0.58 | 0.42 | 0.49 | 0.45 |

### B) Between Three Approaches
Compare the best CF method vs Content-Based vs Knowledge-Based.

| Metric | CF (Best) | Content-Based | Knowledge-Based |
|--------|-----------|---------------|-----------------|
| Precision@5 | 0.72 | 0.58 | 0.65 |
| Recall@5 | 0.58 | 0.52 | 0.48 |
| Coverage | 0.78 | 0.60 | 0.90 |

### C) Under Different Conditions

| Condition | Best Approach | Reason |
|-----------|---------------|--------|
| Dense user data | CF | Leverages peer patterns |
| Cold-start user | Knowledge-Based | No history needed |
| Cold-start item | Content-Based | Matches item features |
| Niche categories | Content-Based | Item features override sparsity |
| Explicit constraints | Knowledge-Based | Precise filtering |

## Implementation (`recommender/evaluation.py`)

```python
class Evaluator:
    def __init__(self, ratings_df, predictions_df):
        self.ratings = ratings_df
        self.predictions = predictions_df
    
    def rmse(self): ...
    def mae(self): ...
    def precision_at_k(self, k=5): ...
    def recall_at_k(self, k=5): ...
    def f1_at_k(self, k=5): ...
    def coverage(self): ...
    
    def compare_cf_methods(self): ...
    def compare_approaches(self): ...
    def analyze_conditions(self): ...
```

## Outputs for Analysis

1. **Best CF method**: Ranked table + recommendation
2. **Best approach overall**: Comparison chart
3. **Condition analysis**: When each approach excels
4. **Why differences occur**: Explanation of algorithmic biases
