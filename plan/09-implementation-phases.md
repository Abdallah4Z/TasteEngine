# Implementation Phases Plan

## Phase 1: Data Generation (Day 1)

**Files**: `data/generate_data.py`, `data/products.csv`, `data/users.csv`, `data/ratings.csv`, `data/interactions.csv`

**Tasks**:
- [ ] Create data generation script with realistic distributions
- [ ] Generate 500 products across 8 categories
- [ ] Generate 200 users with varied preference profiles
- [ ] Generate 5000+ ratings with controlled sparsity (~5%)
- [ ] Generate purchase history
- [ ] Export all CSVs
- [ ] Verify data integrity (no missing values, correct ranges)

**Validation**:
```python
python data/generate_data.py  # generates all CSVs
python -c "import pandas as pd; print(pd.read_csv('data/ratings.csv').shape)"
```

## Phase 2: Collaborative Filtering Engine (Day 2-3)

**Files**: `utils/similarity.py`, `recommender/collaborative.py`

**Tasks**:
- [ ] Implement `similarity.py` (cosine, pearson, adjusted cosine)
- [ ] Build user-item rating matrix utility
- [ ] Implement User-Based CF with cosine similarity
- [ ] Implement Item-Based CF with adjusted cosine
- [ ] Implement SVD with SGD optimization
- [ ] Implement KNN-Based CF
- [ ] Implement Slope One (bonus)
- [ ] Test each method with sample data
- [ ] Verify methods produce different results

**Validation**:
```python
cf = CollaborativeFiltering(ratings_df)
recs = cf.user_based_cf(user_id=1)
assert len(recs) == 10
assert cf.svd(user_id=1) != cf.user_based_cf(user_id=1)
```

## Phase 3: Content-Based Engine (Day 3)

**Files**: `recommender/content_based.py`

**Tasks**:
- [ ] Implement TF-IDF vectorization of product text features
- [ ] Build user profile from rated items
- [ ] Implement TF-IDF + cosine similarity recommendation
- [ ] Implement feature-based matching recommendation
- [ ] Test with cold-start users

**Validation**:
```python
cb = ContentBasedRecommender(products_df)
recs = cb.tfidf_recommend(user_id=1)
assert len(recs) == 10
```

## Phase 4: Knowledge-Based Engine (Day 4)

**Files**: `recommender/knowledge_based.py`

**Tasks**:
- [ ] Implement constraint-based filtering (budget, category, brand)
- [ ] Implement rule-based recommendation engine
- [ ] Implement utility-based scoring
- [ ] Test with explicit constraints

**Validation**:
```python
kb = KnowledgeBasedRecommender(products_df)
recs = kb.constraint_based({'budget_max': 500, 'category': 'Electronics'})
assert all(p.price <= 500 for p in recs)
```

## Phase 5: Evaluation Module (Day 4-5)

**Files**: `recommender/evaluation.py`

**Tasks**:
- [ ] Implement RMSE
- [ ] Implement MAE
- [ ] Implement Precision@K
- [ ] Implement Recall@K
- [ ] Implement F1-Score
- [ ] Implement Coverage
- [ ] Build CF methods comparison function
- [ ] Build approaches comparison function
- [ ] Build condition analysis function
- [ ] Prepare results for UI consumption (JSON)

**Validation**:
```python
evaluator = Evaluator(test_ratings, predictions)
assert evaluator.rmse() > 0  # meaningful value
results = evaluator.compare_cf_methods()
assert len(results) >= 4  # all CF methods
```

## Phase 6: Explanation Module (Day 5)

**Files**: `recommender/explainer.py`

**Tasks**:
- [ ] Implement CF explanation generator (4 templates)
- [ ] Implement Content-Based explanation generator (2 templates)
- [ ] Implement Knowledge-Based explanation generator (3 templates)
- [ ] Map explanations to recommendation results

**Validation**:
```python
exp = Explainer(products_df, users_df)
text = exp.explain_cf('user_based', 1, 42, {'sim_score': 0.84, 'count': 12})
assert 'similar users' in text
```

## Phase 7: Web UI — Backend (Day 6)

**Files**: `app.py`

**Tasks**:
- [ ] Set up Flask app with routes
- [ ] `/` — Home page route (pass user list + product preview)
- [ ] `/recommend` — POST: accept user_id + approach, return recommendations
- [ ] `/evaluate` — GET: run evaluation, return JSON results
- [ ] `/api/users` — JSON endpoint for user data
- [ ] `/api/products` — JSON endpoint for product data
- [ ] Error handling and input validation

## Phase 8: Web UI — Frontend (Day 6-7)

**Files**: `templates/index.html`, `templates/recommend.html`, `templates/evaluation.html`, `static/css/style.css`

**Tasks**:
- [ ] Build base HTML template with navigation
- [ ] Build home page with user selection and approach picker
- [ ] Build recommendations page with cards + tab switching
- [ ] Build evaluation dashboard with Chart.js
- [ ] Style with glassmorphism CSS
- [ ] Add responsive design
- [ ] Add loading states and animations
- [ ] Test all user flows

## Phase 9: Integration & Testing (Day 7-8)

**Tasks**:
- [ ] Wire Flask routes to all recommendation engines
- [ ] Wire evaluation results to dashboard templates
- [ ] End-to-end testing of all flows
- [ ] Fix bugs and edge cases
- [ ] Performance check (response times)
- [ ] Final polish of UI

## Phase 10: Analysis & Report (Day 8-9)

**Tasks**:
- [ ] Run full evaluation suite
- [ ] Document which CF method performs best
- [ ] Document which approach performs best
- [ ] Document conditions for each approach
- [ ] Prepare explanations for why differences occur
- [ ] Final review of all requirements

## Deliverables Summary

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | Synthetic dataset (products, users, ratings) | 📅 Phase 1 |
| 2 | User-Based Collaborative Filtering | 📅 Phase 2 |
| 3 | Item-Based Collaborative Filtering | 📅 Phase 2 |
| 4 | SVD Matrix Factorization | 📅 Phase 2 |
| 5 | KNN-Based Collaborative Filtering | 📅 Phase 2 |
| 6 | Content-Based (TF-IDF + Feature) | 📅 Phase 3 |
| 7 | Knowledge-Based (Constraint + Rule + Utility) | 📅 Phase 4 |
| 8 | Evaluation (6 metrics, 3 comparisons) | 📅 Phase 5 |
| 9 | Explanation module | 📅 Phase 6 |
| 10 | Web UI (3 pages, glassmorphism) | 📅 Phase 7-8 |
| 11 | Integration & testing | 📅 Phase 9 |
| 12 | Analysis report | 📅 Phase 10 |
