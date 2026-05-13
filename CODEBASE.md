# TasteEngine — Intelligent E-Commerce Recommender System

Course: **AIE425 — Intelligent Recommender System**  
University: **Alamein University**  
Stack: **Python, Flask, scikit-learn, pandas, numpy**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Data Layer (`data/`)](#3-data-layer)
4. [Utility Layer (`utils/`)](#4-utility-layer)
5. [Recommender Layer (`recommender/`)](#5-recommender-layer)
6. [Web Application (`app.py`)](#6-web-application)
7. [Frontend (`templates/`, `static/`)](#7-frontend)
8. [How Recommendations Work](#8-how-recommendations-work)
9. [Evaluation & Comparison](#9-evaluation--comparison)
10. [How to Run](#10-how-to-run)

---

## 1. Project Overview

TasteEngine is a **multi-approach recommender system** for e-commerce. It implements three distinct recommendation paradigms:

| Approach | # Methods | How It Works |
|---|---|---|
| **Collaborative Filtering** (🤝) | 5 | Leverages user-item rating patterns to find similar users/items |
| **Content-Based** (🏷️) | 2 | Uses product features (category, brand, name) to find similar items |
| **Knowledge-Based** (⚙️) | 3 | Applies explicit rules, constraints, and utility functions |

The system includes:
- A **synthetic dataset generator** (200 users, 200 products, 8000 ratings)
- **10 recommendation methods** across 3 approaches
- **6 evaluation metrics** for comparing methods and approaches
- An **explanation engine** that generates human-readable reasons for every recommendation
- A **web UI** for interactive exploration
- A **create-user flow** that generates personalized ratings using content-based techniques

---

## 2. Project Structure

```
TasteEngine/
├── app.py                          # Flask web server — routes, API endpoints, main logic
├── requirements.txt                # Python dependencies
├── CODEBASE.md                     # This file — full documentation
├── README.md                       # Project overview
├── .gitignore                      # Git ignore rules
├── data/
│   ├── generate_data.py            # Synthetic dataset generator (products, users, ratings)
│   ├── products.csv                # 200 products with category, brand, price, rating
│   ├── users.csv                   # 200 users with preferences, budget, brands
│   ├── ratings.csv                 # 8000 user-product ratings (1.0-5.0)
│   └── interactions.csv            # Synthetic purchase interactions
├── recommender/
│   ├── __init__.py                 # Package marker
│   ├── collaborative.py            # 5 CF methods: User-Based, Item-Based, SVD, KNN, Slope One
│   ├── content_based.py            # 2 CB methods: TF-IDF, Feature Matching
│   ├── knowledge_based.py          # 3 KB methods: Constraint, Rule, Utility
│   ├── evaluation.py               # 6 metrics + method/approach comparison
│   └── explainer.py                # Generates human-readable explanations
├── utils/
│   ├── __init__.py                 # Package marker
│   ├── helpers.py                  # Data loading, user-item matrix, preference extraction
│   └── similarity.py               # Similarity metrics: cosine, pearson, adjusted cosine, jaccard
├── templates/
│   ├── index.html                  # Landing/home page
│   ├── recommend.html              # Recommendation interface
│   ├── evaluation.html             # Evaluation & comparison dashboard
│   └── create.html                 # Create user with rating flow
├── static/
│   └── css/
│       └── style.css              # Full app styling (glassmorphism, responsive grid)
└── plan/
    ├── 01-overview.md
    ├── 02-dataset.md
    ├── 03-collaborative-filtering.md
    ├── 04-content-based.md
    ├── 05-knowledge-based.md
    ├── 06-evaluation.md
    ├── 07-explanation.md
    ├── 08-ui-ux.md
    └── 09-implementation-phases.md
```

---

## 3. Data Layer (`data/`)

### `generate_data.py`

Generates a complete synthetic e-commerce dataset. Run with `python data/generate_data.py`.

#### Data Files Produced

| File | Records | Columns |
|---|---|---|
| `products.csv` | 200 | product_id, name, category, subcategory, brand, price, avg_rating, num_reviews |
| `users.csv` | 200 | user_id, name, age, preferred_categories, budget_min, budget_max, favorite_brands |
| `ratings.csv` | 8,000 | user_id, product_id, rating (1.0-5.0) |
| `interactions.csv` | 2,000 | user_id, product_id, purchased, quantity |

#### Key Data Constants

```python
CATEGORIES = {
    "Electronics": ["Smartphones", "Laptops", "Headphones", "Tablets", "Cameras"],
    "Clothing": ["Men's", "Women's", "Kids'", "Accessories", "Footwear"],
    "Home & Kitchen": ["Furniture", "Appliances", "Cookware", "Decor", "Storage"],
    "Books": ["Fiction", "Non-Fiction", "Science", "Technology", "Self-Help"],
    "Sports": ["Fitness", "Outdoor", "Team Sports", "Cycling", "Swimming"],
    "Beauty": ["Skincare", "Makeup", "Haircare", "Fragrance", "Tools"],
    "Toys": ["Educational", "Action Figures", "Board Games", "Dolls", "Puzzles"],
    "Automotive": ["Car Care", "Interior", "Exterior", "Tools", "Electronics"],
}
```

8 categories × 5 subcategories × 5 products each = 200 products.  
Each product has a brand, price ($10-$1500), avg_rating (3.0-5.0), and num_reviews (10-500).

#### Functions

**`generate_products(n_per_category=8)`** — Creates 200 products across 8 categories and 40 subcategories. Each gets a realistic name, random brand from the category's brand pool, random price ($10-$1500), random rating (3.0-5.0), and random review count.

**`generate_users(n_users=200)`** — Creates 200 users with random names, ages (18-65), 1-3 preferred categories, a budget range ($10-$1700 total range), and 0-2 favorite brands.

**`generate_ratings(products_df, users_df, n_ratings=8000)`** — Creates 8000 user-product ratings. Each rating is calculated as:
- **Base**: 3.0
- **Category match**: +0.5 to +1.5 if product matches user's preferred category
- **Brand match**: +0.3 to +1.0 if product matches user's favorite brand
- **Budget match**: +0.0 to +0.5 if price is in user's budget range, else -0.0 to -0.5
- **Noise**: Normal distribution (mean=0, std=0.5)
- **Clamped**: Final rating between 1.0 and 5.0

**`generate_interactions(products_df, users_df, rating_df, n_purchases=2000)`** — Creates synthetic purchase interactions from a sample of ratings.

**`main()`** — Orchestrator that generates all data files sequentially.

---

## 4. Utility Layer (`utils/`)

### `utils/helpers.py`

Helper functions for data operations.

**`DATA_DIR`** — Path constant pointing to the `data/` directory.

**`load_data()`** — Reads `products.csv`, `users.csv`, and `ratings.csv` into pandas DataFrames. Returns `(products, users, ratings)`.

**`build_user_item_matrix(ratings_df)`** — Pivots the ratings table into a user×item matrix where rows = users, columns = products, values = ratings. Unrated pairs are NaN. This is the fundamental data structure for collaborative filtering.

**`get_user_preferences(users_df, user_id)`** — Extracts a single user's profile as a dictionary:
```python
{
    "preferred_categories": {"Electronics", "Books"},
    "budget_min": 75.16,
    "budget_max": 812.42,
    "favorite_brands": {"Sony"},
    "name": "Ben_1",
    "age": 27,
}
```

### `utils/similarity.py`

Similarity metrics used by the recommendation algorithms.

**`cosine_similarity(matrix)`** — Wrapper around scikit-learn's cosine similarity. Computes pairwise cosine similarity between all rows of a matrix. Used by User-Based CF to find similar users.

**`pearson_similarity(matrix)`** — Custom implementation of Pearson correlation coefficient. Centers each row by its mean, then computes cosine similarity on the centered data. Measures linear correlation between users' rating patterns.

**`adjusted_cosine_similarity(matrix)`** — Item-oriented version that subtracts each user's mean rating before computing cosine similarity on the transposed (item×item) matrix. Used by Item-Based CF to correct for users who rate everything high/low.

**`jaccard_similarity(set1, set2)`** — |intersection| / |union|. Measures overlap between two sets. Used for binary preference data.

---

## 5. Recommender Layer (`recommender/`)

### `recommender/collaborative.py` — Class `CollaborativeFiltering`

The collaborative filtering engine. Uses the user-item rating matrix to find patterns.

#### Constructor: `__init__(self, ratings_df)`

- Builds the user×item matrix from raw ratings
- Stores user IDs, item IDs, matrix dimensions
- Pre-computes per-user mean ratings and global mean rating
- All three are used as fallbacks for missing data

#### Method: `_get_user_index(user_id)` / `_get_item_index(item_id)`

Maps external user/item IDs to internal matrix indices.

---

**Method 1: `user_based_cf(user_id, n_recommendations=10, k=20)`**  

*Algorithm:* 
1. Compute cosine similarity between the target user and all other users
2. For each product the user hasn't rated:
   - Find the top-k most similar users who rated that product
   - Weight their ratings by similarity score
   - Predict = weighted average of neighbor ratings
3. Return top-N predictions sorted by predicted rating

*Strengths:* Intuitive, captures peer preferences.  
*Weaknesses:* Scales poorly (O(n²) user comparisons), sensitive to sparse data.

---

**Method 2: `item_based_cf(user_id, n_recommendations=10, k=15)`**  

*Algorithm:*
1. Compute adjusted cosine similarity between all pairs of items
2. For each unrated product:
   - Find the top-k most similar items among the user's rated set
   - Weight their ratings by item-item similarity
   - Predict = weighted average
3. Return top-N predictions

*Strengths:* Item similarities are more stable than user similarities, pre-computable.  
*Weaknesses:* Still requires at least some rated items from the user.

---

**Method 3: `svd(user_id, n_recommendations=10, n_factors=20, n_epochs=15, lr=0.01, reg=0.02)`**  

*Algorithm:*
1. Initialize user-factor matrix P and item-factor matrix Q randomly
2. Initialize bias terms bu (user) and bi (item)
3. For each epoch, iterate over all observed ratings:
   - Predict: r̂ = μ + bu[u] + bi[i] + P[u]·Q[i]
   - Compute error: err = r - r̂
   - Update: bu += lr·(err - reg·bu), bi += lr·(err - reg·bi)
   - Update: P[u] += lr·(err·Q[i] - reg·P[u]), Q[i] += lr·(err·P[u] - reg·Q[i])
4. Predict unrated items using learned factors

*Strengths:* Captures latent features, handles sparsity well, state-of-the-art baseline.  
*Weaknesses:* Slow to train, requires hyperparameter tuning (factors, epochs, learning rate).

---

**Method 4: `knn_cf(user_id, n_recommendations=10, k=10)`**  

*Algorithm:*
1. Use scikit-learn's NearestNeighbors with cosine metric to find k nearest neighbors
2. For each unrated product:
   - Get neighbor ratings
   - Weight by inverse distance (closer neighbors = more influence)
3. Return top-N predicted items

*Strengths:* Simple, non-parametric, adapts to local structure.  
*Weaknesses:* Sensitive to k choice, distance metric.

---

**Method 5: `slope_one(user_id, n_recommendations=10)`**  

*Algorithm:*
1. For each pair of items (i, j), compute average rating deviation: dev[i,j] = avg(rating_i - rating_j)
2. For each unrated product i:
   - For each rated product j: predicted_i = user_rating_j + dev[i,j]
   - Average all predictions
3. Return top-N

*Strengths:* Simple, fast prediction, no training needed.  
*Weaknesses:* O(n²) item-pair computation, assumes linear relationships.

---

**Method: `recommend(method, user_id, n_recommendations=10, **kwargs)`**  

Router/dispatcher that maps method names to the five CF algorithms.

**Method: `get_all_methods()`** — Returns the list of all 5 CF method names.

---

### `recommender/content_based.py` — Class `ContentBasedRecommender`

Recommends products similar to what the user has already liked, based on product features.

#### Constructor: `__init__(self, products_df)`

- Creates a `text_features` column: `{category} {subcategory} {brand} {name}`
- Builds a TF-IDF matrix from these text features using scikit-learn's TfidfVectorizer (English stop words removed)
- Stores product IDs for result mapping

---

**Method 1: `tfidf_recommend(user_profile_items, n_recommendations=10)`**  

*Algorithm:*
1. Takes the list of product IDs the user rated highly (≥3.5)
2. Averages their TF-IDF vectors to create a "user profile vector"
3. Computes cosine similarity between profile and ALL products
4. Excludes already-rated items
5. Returns top-N most similar products with similarity scores

*Strengths:* No cold-start for items, captures nuanced feature similarities.  
*Weaknesses:* Needs rated items to build profile, tends to overspecialize.

---

**Method 2: `feature_match_recommend(preferences, n_recommendations=10)`**  

*Algorithm:*
1. Score every product based on explicit preference matching:
   - Category match: +40 points
   - Brand match: +30 points  
   - Budget fit: +20 points
2. Return top-N highest-scoring products

*Strengths:* Works with zero user history (cold-start), transparent.  
*Weaknesses:* Simple scoring, no nuanced similarity.

---

**Method: `recommend(method, user_profile_items=None, preferences=None, n_recommendations=10)`**  

Router for the two content-based methods.

---

### `recommender/knowledge_based.py` — Class `KnowledgeBasedRecommender`

Uses explicit domain knowledge, rules, and constraints rather than statistical patterns.

#### Constructor: `__init__(self, products_df)`  

Stores the product catalog for filtering and scoring.

---

**Method 1: `constraint_based(constraints, n_recommendations=10)`**  

*Algorithm:*
1. Start with all products
2. Sequentially filter by constraints:
   - `budget_max`: price ≤ max
   - `budget_min`: price ≥ min
   - `category`: product category is in preferred list
   - `brand` (soft): if user has favorite brands, restrict to those
   - `min_rating`: product avg_rating ≥ threshold
   - `subcategory` (soft): restrict to preferred subcategories if provided
3. Sort remaining by avg_rating descending
4. Return top-N

*Strengths:* Deterministic, transparent, works with no history.  
*Weaknesses:* Can return empty results if constraints are too strict.

---

**Method 2: `rule_based(context, n_recommendations=10)`**  

*Algorithm:*
1. Uses hard-coded business rules:
   - If user interacted with Laptops → recommend accessories (Laptop Bag, Mouse, etc.)
   - If user interacted with Smartphones → recommend Phone Case, Charger, etc.
   - Category match: +20 points
   - Budget fit: +15 points
   - Brand match: +10 points
2. Score and rank products
3. Return top-N

*Strengths:* Mimics real-world cross-selling, domain-specific.  
*Weaknesses:* Rules must be manually defined, not data-driven.

---

**Method 3: `utility_based(preferences, weights=None, n_recommendations=10)`**  

*Algorithm:*
1. For each product, compute a multi-attribute utility score:
   - **Price utility**: How close is the price to the middle of the user's budget range? (weight: 0.2)
   - **Category utility**: Does the category match preferences? (weight: 0.3)
   - **Brand utility**: Does the brand match? (weight: 0.2)
   - **Rating utility**: Normalized avg_rating (weight: 0.3)
2. U = Σ(weight_i × utility_i)
3. Return top-N by total utility

*Strengths:* Flexible weighting, customizable trade-offs.  
*Weaknesses:* Weights must be set manually.

---

**Method: `recommend(method, constraints=None, context=None, preferences=None, n_recommendations=10)`**  

Router for the three knowledge-based methods.

---

### `recommender/evaluation.py` — Class `Evaluator`

Comprehensive evaluation suite with 6 metrics.

#### Constructor: `__init__(self, ratings_df, predictions_df=None)`

Stores the full ratings dataset and optional pre-computed predictions.

#### Metrics

**`rmse(y_true, y_pred)`** — Root Mean Squared Error. Penalizes large errors quadratically. Range: 0-∞ (lower is better). Formula: √(mean((true - pred)²))

**`mae(y_true, y_pred)`** — Mean Absolute Error. Average absolute difference. Range: 0-∞ (lower is better). Formula: mean(|true - pred|)

**`precision_at_k(recommended_items, relevant_items, k=5)`** — Of the top-k recommended items, how many are relevant? Range: 0-1 (higher is better). Formula: |top-k ∩ relevant| / k

**`recall_at_k(recommended_items, relevant_items, k=5)`** — Of all relevant items, how many appear in the top-k? Range: 0-1 (higher is better). Formula: |top-k ∩ relevant| / |relevant|

**`f1_at_k(recommended_items, relevant_items, k=5)`** — Harmonic mean of precision and recall. Range: 0-1 (higher is better). Formula: 2·P·R / (P+R)

**`coverage(recommended_items_list, total_items)`** — What fraction of the total product catalog was recommended to at least one user? Range: 0-1 (higher is better). Measures catalog exploration vs. concentration.

#### Evaluation Methods

**`set_test_ratings(test_ratings)`** — Sets the held-out test set for evaluation.

**`_get_relevant_for_user(user_id, rating_threshold=3.5)`** — Gets the list of products a user rated ≥ threshold in the test set. These are "ground truth" relevant items.

**`evaluate_cf_method(method_name, cf_instance, test_ratings, k=5, rating_threshold=3.5, max_users=20)`** — Full evaluation pipeline for a single CF method:
1. For each test user: get recommendations, collect predicted ratings
2. Compute RMSE and MAE on rating prediction accuracy
3. Compute Precision@k, Recall@k, F1@k for each user
4. Compute Coverage across all users
5. Returns a dict with all 6 metrics

**`compare_cf_methods(cf_instance, test_ratings, k=5)`** — Runs `evaluate_cf_method` for all 5 CF methods and returns a list of result dicts.

**`evaluate_approach(approach_name, recommender_fn, test_users, products_df, k=5)`** — Evaluates a recommendation function (any approach) by computing Precision@k, Recall@k, and Coverage.

**`compare_approaches(cf_instance, cb_instance, kb_instance, test_ratings, products_df, k=5)`** — Compares all 3 approaches by running their recommenders on test users and collecting metrics.

---

### `recommender/explainer.py` — Class `Explainer`

Generates human-readable explanations for every recommendation, customized by approach and method.

#### Constructor: `__init__(self, products_df, users_df)`

Pre-builds dictionaries for O(1) product and user lookups.

#### Helper Methods

**`_get_product(product_id)`** / **`_get_user(user_id)`** — Fast lookups from cached dicts.

**`_fmt_score(score)`** — Converts a numeric similarity score (0-1) into a verbal label: "Excellent match" (≥0.9), "Strong match" (≥0.7), "Good match" (≥0.5), "Moderate match" (≥0.3), "Partial match" (<0.3).

**`_pref_list(user, key)`** — Extracts a user's preferred categories or brands as a clean list, handling CSV string or set formats.

#### Explanation Templates

**`explain_cf(method, user_id, product_id, details=None)`**

| Method | Example Explanation |
|---|---|
| user_based | "Users with Electronics taste also liked iPhone 15 Pro" |
| item_based | "Matches your Electronics preferences — similar to items you've rated highly" |
| svd | "Fits your profile: Apple's Electronics — top latent factor match" |
| knn | "Popular among peers who also like Electronics" |
| slope_one | "Frequently chosen by users who liked the same Electronics products" |

**`explain_content(method, user_id, product_id, details=None)`**

| Method | Example Explanation |
|---|---|
| tfidf | "Content matches: Electronics / Smartphones — strongly aligns with your past likes" |
| feature_match | "Matches your preferences: Electronics, Apple" |

**`explain_knowledge(method, user_id, product_id, details=None)`**

| Method | Example Explanation |
|---|---|
| constraint | "Fits your criteria: $999 within budget · Electronics · Apple" |
| rule | "Electronics buyers commonly add iPhone 15 Pro — cross-sell match" |
| utility | "Top utility score: Electronics × Apple matches your preference weights" |

**`get_explanation(approach, method, user_id, product_id, details=None)`** — Universal router that delegates to the appropriate explain method.

---

---

## 5b. Algorithm Deep Dives — Code-Level Explanations

### Collaborative Filtering — User-Based (`user_based_cf`)

```python
def user_based_cf(self, user_id, n_recommendations=10, k=20):
```

**Step-by-step code walkthrough:**

**Step 1: Get the user's matrix index**
```python
u_idx = self._get_user_index(user_id)   # line 47
if u_idx is None:
    return []                            # user not found in training data
```
`_get_user_index` does `np.where(self.user_ids == user_id)[0]` — it finds where in the array of user IDs the target user lives. The matrix is indexed by position, not by ID, so this mapping is essential.

**Step 2: Compute user-user similarity matrix**
```python
matrix_filled = np.nan_to_num(self.user_item_matrix, nan=self.global_mean)  # line 51
sim_matrix = cosine_similarity(matrix_filled)                                 # line 52
user_sim = sim_matrix[u_idx]                                                  # line 53
user_sim[u_idx] = 0                                                           # line 54
```
- `np.nan_to_num` replaces NaN (unrated pairs) with the global mean rating (≈3.0). This is an imputation strategy — we assume unrated items would get an average rating.
- `cosine_similarity` computes the cosine of the angle between every pair of user vectors. Result: a N×N matrix where `sim_matrix[a][b]` = how similar user a is to user b.
- We extract the row for our target user (`user_sim`), then set `user_sim[u_idx] = 0` so a user is not considered similar to themselves.

**Step 3: Find unrated items**
```python
user_ratings = self.user_item_matrix[u_idx]      # line 56
unseen = np.where(np.isnan(user_ratings))[0]     # line 57
```
`np.isnan` finds which items the user hasn't rated (NaN in the matrix). These are the candidates for recommendation.

**Step 4: Predict ratings for each unrated item**
```python
for i_idx in unseen:                                    # line 62
    similar_users = np.argsort(user_sim)[::-1][:k]      # line 63
```
- `np.argsort` returns indices sorted by similarity ascending. `[::-1]` reverses to descending. `[:k]` takes top k.

```python
    valid = []
    for su in similar_users:                            # line 64-67
        if not np.isnan(self.user_item_matrix[su, i_idx]) and user_sim[su] > 0:
            valid.append(su)
```
- We filter to neighbors who actually RATED this item and have positive similarity. A neighbor with 0 or negative similarity shouldn't influence the prediction.

```python
    if not valid:
        continue                                        # skip if no valid neighbors
    sim_vals = user_sim[valid]                          # line 70
    ratings_vals = self.user_item_matrix[valid, i_idx]  # line 71
    pred = np.average(ratings_vals, weights=sim_vals)   # line 72
```
- The prediction is a **weighted average**: each neighbor's rating is weighted by how similar they are to the target user.

**The math:**
$$r_{u,i} = \frac{\sum_{v \in N_k(u)} sim(u,v) \cdot r_{v,i}}{\sum_{v \in N_k(u)} sim(u,v)}$$

Where:
- $r_{u,i}$ = predicted rating for user u on item i
- $N_k(u)$ = k nearest neighbors of user u
- $sim(u,v)$ = cosine similarity between user u and user v
- $r_{v,i}$ = actual rating given by neighbor v to item i

**Step 5: Return top-N**
```python
predictions.sort(key=lambda x: x[1], reverse=True)    # line 75
return predictions[:n_recommendations]                 # line 76
```

---

### Collaborative Filtering — Item-Based (`item_based_cf`)

```python
def item_based_cf(self, user_id, n_recommendations=10, k=15):
```

**The key insight:** Item-item similarities are more stable than user-user similarities because items' attributes don't change, while user tastes evolve. We can pre-compute the item-item similarity matrix once.

**Step 1: Compute adjusted cosine similarity between items**
```python
item_sim = adjusted_cosine_similarity(self.user_item_matrix)  # line 90
```
In `utils/similarity.py`, `adjusted_cosine_similarity` does:
```python
user_mean = np.nanmean(matrix, axis=1, keepdims=True)  # each user's avg rating
matrix_centered = matrix - user_mean                    # subtract user bias
return sklearn_cosine(matrix_centered.T)                # transpose = item×item
```
**Why adjusted cosine?** Users have different rating scales (some always rate 4+, others use the full 1-5 range). By subtracting each user's mean, we correct for this bias before computing similarity. The `.T` transposes from user×item to item×item.

**Step 2: For each unrated item, find similar items the user HAS rated**
```python
sim_to_rated = item_sim[i_idx, rated]   # line 100: similarities to user's rated items
best = np.argsort(sim_to_rated)[::-1][:k]  # line 101: top k most similar
```
`rated` is an array of indices where the user has non-NaN ratings. `item_sim[i_idx, rated]` gets the similarity between the target item `i_idx` and every item the user has rated.

**Step 3: Weighted average prediction**
```python
neighbor_indices = [rated[r[0]] for r in valid]   # line 105: actual item indices
sim_vals = [r[1] for r in valid]                   # line 106: similarity scores
rating_vals = user_ratings[neighbor_indices]       # line 107: user's ratings
pred = np.average(rating_vals, weights=sim_vals)   # line 108: weighted average
```

**The math:**
$$r_{u,i} = \frac{\sum_{j \in I_u} sim(i,j) \cdot r_{u,j}}{\sum_{j \in I_u} sim(i,j)}$$

Where $I_u$ is the set of items user u has rated.

---

### Collaborative Filtering — SVD / Matrix Factorization (`svd`)

```python
def svd(self, user_id, n_recommendations=10, n_factors=20, n_epochs=15, lr=0.01, reg=0.02):
```

**The core idea:** Instead of computing direct similarities, we learn latent (hidden) factors that explain observed ratings. For example, a latent factor might represent "Action Movie Appeal" — users have a value for this factor, and movies have a value for this factor. The rating is the dot product of user factors and item factors.

**Step 1: Initialize factors and biases**
```python
P = np.random.normal(0, 0.1, (n_u, n_factors))   # line 131: user-factor matrix (200×20)
Q = np.random.normal(0, 0.1, (n_i, n_factors))   # line 132: item-factor matrix (200×20)
bu = np.zeros(n_u)                                 # line 133: user bias
bi = np.zeros(n_i)                                 # line 134: item bias
```
- `P[u]` is a 20-dimensional vector representing user u's preference on 20 latent factors
- `Q[i]` is a 20-dimensional vector representing item i's "strength" on those same factors
- `bu[u]` captures how much user u tends to rate above/below average
- `bi[i]` captures how much item i tends to be rated above/below average

**Step 2: Build list of observed ratings**
```python
observed = []
for u in range(n_u):
    for i in range(n_i):
        if not np.isnan(self.user_item_matrix[u, i]):
            observed.append((u, i))               # lines 136-140
```
This collects all (user, item) pairs where we have a rating. We only train on observed data.

**Step 3: SGD training loop**
```python
for epoch in range(n_epochs):                     # line 142
    np.random.shuffle(observed)                   # line 143: shuffle for SGD
    for u, i in observed:                         # line 144
        r = self.user_item_matrix[u, i]           # actual rating
        pred = self.global_mean + bu[u] + bi[i] + np.dot(P[u], Q[i])  # line 146
        err = r - pred                            # line 147: prediction error
        
        bu[u] += lr * (err - reg * bu[u])         # line 148: update user bias
        bi[i] += lr * (err - reg * bi[i])         # line 149: update item bias
        P[u] += lr * (err * Q[i] - reg * P[u])   # line 150: update user factors
        Q[i] += lr * (err * P[u] - reg * Q[i])    # line 151: update item factors
```

**The prediction formula:**
$$\hat{r}_{u,i} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$$

Where:
- $\mu$ = global mean rating
- $b_u$ = bias of user u
- $b_i$ = bias of item i
- $\mathbf{p}_u$ = user u's latent factor vector
- $\mathbf{q}_i$ = item i's latent factor vector

**The update rules (SGD with regularization):**
$$b_u \leftarrow b_u + \eta \cdot (e_{ui} - \lambda \cdot b_u)$$
$$b_i \leftarrow b_i + \eta \cdot (e_{ui} - \lambda \cdot b_i)$$
$$\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta \cdot (e_{ui} \cdot \mathbf{q}_i - \lambda \cdot \mathbf{p}_u)$$
$$\mathbf{q}_i \leftarrow \mathbf{q}_i + \eta \cdot (e_{ui} \cdot \mathbf{p}_u - \lambda \cdot \mathbf{q}_i)$$

Where $e_{ui} = r_{ui} - \hat{r}_{ui}$ is the prediction error, $\eta$ is the learning rate (lr), and $\lambda$ is the regularization strength (reg).

**Why it works:** The dot product $P[u]·Q[i]$ captures the interaction between user preferences and item attributes in latent space. If a user loves action movies (high first factor) and a movie is action-packed (high first factor), their dot product will be large, predicting a high rating.

**Step 4: Predict unrated items**
```python
for i_idx in unseen:
    pred = self.global_mean + bu[u_idx] + bi[i_idx] + np.dot(P[u_idx], Q[i_idx])
```
We use the learned factors to predict ratings for items the user hasn't seen. No retraining needed — the factors capture general patterns.

---

### Collaborative Filtering — KNN (`knn_cf`)

```python
def knn_cf(self, user_id, n_recommendations=10, k=10):
```

**Step 1: Find k nearest neighbors using scikit-learn**
```python
nn = NearestNeighbors(n_neighbors=min(k + 1, self.n_users), metric="cosine")  # line 178
nn.fit(matrix_imputed)                                                          # line 179
distances, indices = nn.kneighbors(matrix_imputed[u_idx].reshape(1, -1))       # line 180
neighbor_indices = indices[0][1:]        # line 181: skip self (first neighbor is self)
```
scikit-learn's `NearestNeighbors` with cosine metric efficiently finds the k most similar users using optimized data structures (ball tree or KD-tree). We add +1 and then slice `[1:]` to exclude the user themselves (who is always the closest match).

**Step 2: Inverse-distance-weighted prediction**
```python
weights = 1.0 / np.array(neighbor_dists)    # line 195: closer = more weight
pred = np.average(neighbor_ratings, weights=weights)  # line 196
```
Instead of using raw cosine similarity as weights (like User-Based CF), KNN uses **inverse distance**. The `1e-6` added to distances prevents division by zero. This means very close neighbors have very high influence, while distant ones contribute little.

---

### Collaborative Filtering — Slope One (`slope_one`)

```python
def slope_one(self, user_id, n_recommendations=10):
```

**Step 1: For each pair of items, compute the average rating deviation**
```python
for u in range(self.n_users):                                              # line 228
    if not np.isnan(self.user_item_matrix[u, i_idx]) and \
       not np.isnan(self.user_item_matrix[u, j_idx]):                     # line 229
        dev += self.user_item_matrix[u, i_idx] - self.user_item_matrix[u, j_idx]  # line 230
        count += 1
avg_dev = dev / count                                    # line 233: average deviation
```

**The insight:** If the average deviation between item A and item B is +0.5 (meaning people rate A 0.5 higher than B), and a user rated B as 4, we predict they'd rate A as 4.5.

**Step 2: Aggregate predictions**
```python
numerator += (user_ratings[j_idx] + avg_dev)   # line 234: predicted rating from item j
denominator += 1                                # line 235: count of rated items used
pred = numerator / denominator                  # line 237: average prediction
```
For each rated item j, we compute what the user would rate item i, given the deviation between i and j. Say user rated items A=3, B=4:
- For unrated item X:
  - dev(X, A) = +0.5 → predicted from A: 3 + 0.5 = 3.5
  - dev(X, B) = -0.2 → predicted from B: 4 + (-0.2) = 3.8
  - Final prediction: (3.5 + 3.8) / 2 = 3.65

**Complexity:** O(n² × m) where n = items and m = users. Each pair of items requires scanning all users. This is the slowest method for training but fastest for prediction.

---

### Content-Based — TF-IDF (`tfidf_recommend`)

```python
def tfidf_recommend(self, user_profile_items, n_recommendations=10):
```

**Step 1: Build user profile from liked items**
```python
profile_indices = []
for pid in user_profile_items:
    mask = self.products["product_id"] == pid    # find product in catalog
    if mask.any():
        idx = self.products[mask].index[0]
        profile_indices.append(idx)               # index into TF-IDF matrix

profile_vector = np.asarray(self.tfidf_matrix[profile_indices].mean(axis=0))  # line 35
```

The user's "taste profile" is the **centroid** (average) of all TF-IDF vectors for items they liked. This represents the "typical" product the user enjoys.

**How TF-IDF works:**

TF-IDF = Term Frequency × Inverse Document Frequency

- **Term Frequency:** How often a word appears in a product's text (e.g., "Laptops" appears once in "Electronics Laptops Dell XPS 15")
- **Inverse Document Frequency:** log(Total Products / Products containing this word). Rare words get higher weight.

The vectorizer builds a vocabulary from all product texts, then each product becomes a sparse vector where each dimension is a word's TF-IDF score.

**Step 2: Compute cosine similarity**
```python
sim_scores = cosine_similarity(profile_vector, self.tfidf_matrix).flatten()  # line 35
```
Compares the user's profile vector to every product vector. Products with similar word patterns get high scores.

**Why it works:** Products in the same category (e.g., "Electronics") share words like "Electronics" and specific subcategory/brand names. A user who liked "Galaxy S24" (Smartphones category, Samsung brand) will have a profile that weights those terms highly, so other Samsung phones or Electronics products rank higher.

---

### Knowledge-Based — Constraint (`constraint_based`)

```python
def constraint_based(self, constraints, n_recommendations=10):
```

**Sequential filtering pipeline:**
```python
filtered = self.products.copy()                       # start with all 200 products

if "budget_max" in constraints:
    filtered = filtered[filtered["price"] <= constraints["budget_max"]]  # price ≤ max

if "category" in constraints and constraints["category"]:
    filtered = filtered[filtered["category"].isin(constraints["category"])]  # match cat
```
Each constraint acts as a **hard filter** — products that don't satisfy ALL constraints are removed. This is equivalent to SQL `WHERE price <= 500 AND category IN ('Electronics', 'Books')`.

**Soft brand filter:**
```python
brand_match = filtered[filtered["brand"].isin(constraints["brand"])]  # line 21
if brand_match is not None and not brand_match.empty:
    filtered = brand_match                            # only if matches exist
```
Brand filtering is "soft" — if no products match the brand constraint, it's ignored. This prevents returning empty results.

**Final ranking:**
```python
filtered = filtered.sort_values("avg_rating", ascending=False)  # line 32
```
After all filters, products are ranked by average rating. This means the "best" (highest-rated) products that meet the user's constraints are recommended.

---

### Knowledge-Based — Utility (`utility_based`)

```python
def utility_based(self, preferences, weights=None, n_recommendations=10):
```

**Multi-attribute utility theory:**
Each product gets a score that is a weighted sum of four normalized utilities:

```python
U = w_price · u_price + w_cat · u_cat + w_brand · u_brand + w_rating · u_rating
```

Where:
- `w_price = 0.2`, `w_cat = 0.3`, `w_brand = 0.2`, `w_rating = 0.3`

**Price utility** (how close to the middle of budget):
```python
price_score = 1.0 - abs(product["price"] - midpoint) / (range/2)   # line 89
```
A product priced exactly at the midpoint of the user's budget range gets score 1.0. At the edges of the range, score approaches 0. Outside the range, score is clamped to 0.

**Category utility:**
```python
cat_score = 1.0 if product["category"] in pref_cats else 0.0    # line 95
```
Binary: the product's category either matches the user's preferences or not.

**Brand utility:**
```python
brand_score = 1.0 if product["brand"] in fav_brands else 0.0     # line 98
```
Binary: same logic as category.

**Rating utility:**
```python
rating_score = product["avg_rating"] / max_rating                # line 101
```
Normalized product rating (0 to 1). A product with 5.0 avg gets score 1.0, 3.0 gets 0.6.

---

### Explanation Engine — How It Generates Text

```python
def explain_cf(self, method, user_id, product_id, details=None):
```

Each method has a **template** that gets filled with product information:

```python
templates = {
    "user_based": f"Users with {cat} taste also liked {pname}",
    "item_based": f"Matches your {cat} preferences — similar to items you've rated highly",
    "svd": f"Fits your profile: {brand}'s {cat} — top latent factor match",
    "knn": f"Popular among peers who also like {cat}",
    "slope_one": f"Frequently chosen by users who liked the same {cat} products",
}
```

The product's `category`, `brand`, and `name` are extracted from the cache and inserted into the appropriate template. This makes each explanation specific to both the recommendation method AND the individual product.

For example, if User-Based CF recommends "iPhone 15 Pro" (category: Electronics), the explanation becomes:
> "Users with Electronics taste also liked iPhone 15 Pro"

---

## 6. Web Application (`app.py`)

Flask web server that ties everything together. Uses global in-memory DataFrames loaded at startup from CSV files.

### Module-Level Initialization

```python
products, users, ratings = load_data()                 # Load CSV data
cf = CollaborativeFiltering(ratings)                    # Full-data CF
cb = ContentBasedRecommender(products)                  # CB engine
kb = KnowledgeBasedRecommender(products)                # KB engine
explainer = Explainer(products, users)                  # Explanation engine
evaluator = Evaluator(ratings)                          # Evaluation suite

TRAIN = ratings.sample(frac=0.8, random_state=42)      # 80% training
TEST = ratings.drop(TRAIN.index)                        # 20% testing
cf_train = CollaborativeFiltering(TRAIN)                # Train-only CF for eval

USER_OPTIONS = [...]                                    # Cached user list for UI
CATEGORIES = [...]                                      # Unique categories
BRANDS = [...]                                          # Unique brands
APPROACHES = {...}                                      # Approach/method definitions
```

### Helper Functions

**`get_product_info(product_id)`** — Lookup a product by ID, returns a dict with all fields.

**`get_user_rated_items(user_id)`** — Returns list of product IDs the user rated ≥ 3.5 (their "liked" items for content-based).

**`_generate_analysis(cf_results, approach_results, best_cf, best_approach)`** — Generates data-driven analysis text for the evaluation page. Examines actual metrics to explain *which method/approach won and why*, not just generic text. Produces 4 sections: method analysis, approach analysis, conditions, and why-differences.

**`_save_and_generate_ratings(user_id, user_cats, user_brands, budget_min, budget_max, manual_ratings)`** — For the create-user flow: takes 5 manual ratings and auto-generates ~35 more using profile matching (category, brand, budget) plus noise.

**`get_category_icon(category)`** / **`stars_html(rating)`** — UI helpers for rendering product cards.

### Page Routes

| Route | Method | Function | Description |
|---|---|---|---|
| `/` | GET | `index()` | Landing page with user selector + approach overview |
| `/recommend` | GET | `recommend_page()` | Recommendation interface with approach/method selection |
| `/evaluate` | GET | `evaluate_page()` | Evaluation dashboard with tables, charts, and analysis |
| `/create` | GET | `create_user_page()` | Create-user form with rating flow |

### API Routes

| Route | Method | Function | Description |
|---|---|---|---|
| `/api/users` | GET | `api_users()` | List all users |
| `/api/users` | POST | `api_create_user()` | Create new user (basic, no ratings) |
| `/api/user/<id>` | GET | `api_user()` | Get single user preferences |
| `/api/user/<id>/preferences` | PUT | `api_update_preferences()` | Update user budget |
| `/api/products` | GET | `api_products()` | List products, optional category filter |
| `/api/products/filter` | GET | `api_products_filter()` | Advanced product filter (category, brand, price, text search) |
| `/api/recommend` | POST | `api_recommend()` | Generate recommendations for a user with a specific approach/method |
| `/api/evaluate` | GET | `api_evaluate()` | Full evaluation: all CF methods + all 3 approaches + analysis |
| `/api/evaluate/cf/<method>` | GET | `api_evaluate_cf()` | Evaluate single CF method |
| `/api/evaluate/cf/<method>/stream` | GET | `api_evaluate_cf_stream()` | Streamed evaluation (for slow methods like SVD) |
| `/api/evaluate/approaches` | GET | `api_evaluate_approaches()` | Evaluate all 3 approaches (Precision/Recall only) |
| `/api/create/step1` | POST | `api_create_step1()` | Create user + optionally save ratings + auto-generate |

### `api_recommend()` — The Core Recommendation Endpoint

Request body:
```json
{
    "user_id": 1,
    "approach": "cf",
    "method": "item_based",
    "n": 10
}
```

For each approach:
- **CF**: Calls `cf.recommend(method, user_id)` → gets products + scores → generates CF explanations
- **Content-Based**: Gets user's highly-rated items → calls `cb.recommend(method, ...)` → generates CB explanations
- **Knowledge-Based**: Builds constraints/context from user preferences → calls `kb.recommend(method, ...)` → generates KB explanations

Response includes product info, prediction score, and human-readable explanation for each recommendation.

### `api_evaluate()` — The Evaluation Endpoint

Runs:
1. `compare_cf_methods()` — All 5 CF methods across 20 test users, 6 metrics each
2. Approach comparators — Item-Based CF vs Content-Based TF-IDF vs Knowledge-Based Constraint
3. `_generate_analysis()` — Data-driven reasoning text

Returns:
```json
{
    "cf_methods": [{"method": "item_based", "RMSE": 0.9318, "MAE": 0.7343, ...}, ...],
    "best_cf_method": "item_based",
    "approaches": [{"approach": "Content-Based", "Precision@5": 0.0526, ...}, ...],
    "best_approach": "Content-Based",
    "analysis": {
        "method": "...explanation text...",
        "approach": "...explanation text...",
        "conditions": "...scenario analysis...",
        "why": "...fundamental differences..."
    }
}
```

### Create User Flow

1. User fills profile form (name, age, categories, brands, budget)
2. System shows 5 products from their preferred categories to rate (1-5 stars)
3. User rates all 5
4. System creates the user in `users.csv` and `USER_OPTIONS`
5. System auto-generates ~35 more ratings using profile matching + noise
6. All 3 approaches (CF, CB, KB) now work immediately — no cold-start
7. Saves to `ratings.csv` for persistence

---

## 7. Frontend

### Templates

**`index.html`** — Landing page with:
- Glass-morphism hero section
- User selector dropdown with profile card (avatar, name, age, budget, category/brand tags)
- Quick action buttons (Recommend, Evaluate, Create User)
- Approach overview cards (CF, CB, KB)
- LocalStorage persistence for selected user

**`recommend.html`** — Interactive recommendation page:
- User selector with profile mini-card
- Number of recommendations slider
- Approach toggle buttons (CF/CB/KB) with method sub-buttons
- Product grid with cards (icon, name, brand, price, rating stars, explanation)
- "Compare All Approaches" tab showing side-by-side results for all 10 methods
- Loading spinner during API calls

**`evaluation.html`** — Evaluation dashboard:
- "Run Evaluation" button triggers the full evaluation API
- CF methods comparison table (RMSE, MAE, Precision@5, Recall@5, F1@5, Coverage)
- Best CF method highlighted + badge
- Chart.js bar chart comparing RMSE/MAE across methods
- Approach comparison table (3 approaches × 2 metrics)
- Analysis section with 4 blocks:
  - **Which method performs best?** — Data-driven explanation
  - **Which approach performs best?** — Data-driven explanation
  - **Under what conditions does each perform better?** — 5 scenarios
  - **Why do differences occur?** — Algorithmic mechanism explanation

**`create.html`** — Two-step create user form:
- Step 1: Name, age, category/brand selector buttons, budget range
- Step 2: 5 products with star-rating UI
- Submission creates user + auto-generates ratings + redirects to recommend

### Styling (`static/css/style.css`)

Design system: **Dark glassmorphism** theme.

Key CSS features:
- `body::before` — Ambient gradient glow effect (purple/pink radial gradients)
- `.glass` — Frosted glass cards with `backdrop-filter: blur(16px)`
- `.nav` — Sticky top nav with blur background
- `.product-grid` — Auto-fill responsive grid (min 280px cards)
- `.eval-table` — Clean table with tabular-nums, uppercase headers
- `.star-rating` — Custom star rating component
- `.approach-grid` — 3-column responsive grid for approach cards
- Animations: `fadeIn` for tab content, `spin` for loading spinner
- Custom scrollbar styling

---

## 8. How Recommendations Work

### Data Flow

```
User selects a user + approach + method
        │
        ▼
API receives: {user_id, approach, method, n}
        │
        ├── CF ──────► rating matrix → similarity → predict unrated items
        │
        ├── Content ──► rated items (≥3.5) → TF-IDF profile → cosine similarity
        │               OR user preferences → feature matching scores
        │
        └── Knowledge ─► user constraints → filter products → score/rank
                          OR business rules → score products
                          OR utility function → compute weighted scores
        │
        ▼
Products ranked by score, top-N returned
        │
        ▼
Explanations generated per method
        │
        ▼
Displayed as product cards with scores + explanations
```

### Cold Start Handling

| Scenario | CF | Content-Based (TF-IDF) | Content-Based (Feature) | Knowledge-Based |
|---|---|---|---|---|
| **New user, no ratings** | ❌ Fails | ❌ Fails | ✅ Works (profile) | ✅ Works (constraints) |
| **New item** | ❌ Fails | ✅ Works (features) | ✅ Works | ✅ Works |
| **Existing user** | ✅ | ✅ | ✅ | ✅ |

### Create User Flow (Cold Start Solution)

When a user is created via `/create`:
1. They fill their profile (categories, brands, budget)
2. They rate 5 products manually
3. System auto-generates ~35 more ratings using:
   - Category match → +0.5 to +1.5 on base rating
   - Brand match → +0.3 to +1.0
   - Budget fit → +0.0 to +0.5
   - Budget misfit → -0.0 to -0.5
   - Random noise → Normal(0, 0.5)
4. Total: 40 ratings (same as existing users)
5. Result: ALL 5 CF methods + TF-IDF + Feature Match + KB all work immediately

---

## 9. Evaluation & Comparison

### Requirements from Project Description

The system must:
- ✅ Use **at least 4 evaluation techniques** → We use **6** (RMSE, MAE, Precision@k, Recall@k, F1@k, Coverage)
- ✅ **Compare between CF methods** → Table of all 5 with all 6 metrics
- ✅ **Compare between the 3 approaches** → CF vs Content-Based vs Knowledge-Based
- ✅ **Answer: Which method performs best?** → Data-driven analysis in evaluation page
- ✅ **Answer: Which approach performs best?** → Data-driven analysis in evaluation page
- ✅ **Answer: Under what conditions does each perform better?** → 5 scenario analysis
- ✅ **Answer: Why do differences occur?** → Algorithmic mechanism explanation
- ✅ **Focus on analysis and reasoning, not just displaying results** → Each metric is accompanied by text explaining why

### Typical Results (Synthetic Dataset)

| Method | RMSE ↓ | MAE ↓ | Precision@5 ↑ | Recall@5 ↑ | F1@5 ↑ | Coverage ↑ |
|---|---|---|---|---|---|---|
| User-Based | ~1.01 | ~0.82 | ~0.03 | ~0.04 | ~0.04 | ~0.51 |
| Item-Based | **~0.93** | **~0.73** | ~0.02 | ~0.04 | ~0.03 | **~0.85** |
| SVD | ~0.95 | ~0.77 | ~0.02 | ~0.04 | ~0.03 | ~0.27 |
| KNN | ~0.98 | ~0.78 | **~0.04** | **~0.07** | **~0.05** | ~0.55 |
| Slope One | ~0.97 | ~0.76 | ~0.01 | ~0.01 | ~0.01 | ~0.30 |

| Approach | Precision@5 ↑ | Recall@5 ↑ |
|---|---|---|
| Collaborative Filtering | ~0.02 | ~0.04 |
| **Content-Based** | **~0.05** | **~0.10** |
| Knowledge-Based | ~0.03 | ~0.05 |

### Why Item-Based CF Wins RMSE

Item-item similarities are more stable than user-user similarities because:
- Items have fixed attributes (category, brand, price) that don't change
- Users' rating behaviors vary widely (some rate everything 4+, some are strict)
- Adjusted cosine similarity corrects for user bias

### Why Content-Based Wins Approach Comparison

On this synthetic dataset:
- Products have rich, discriminative text features (category, subcategory, brand, name)
- TF-IDF vectorization captures these effectively
- CB doesn't suffer from the sparsity that limits CF's precision
- KB is limited by the breadth of user constraints

### When Each Approach Excels

| Scenario | Best Approach | Reason |
|---|---|---|
| **Dense user-item ratings** | CF | Leverages peer behavior patterns |
| **Cold-start user (no history)** | Knowledge-Based | No ratings needed, works with constraints alone |
| **Cold-start item (new product)** | Content-Based | Matches item features to user profile |
| **Explicit constraints (budget, brand)** | Knowledge-Based | Precise, deterministic filtering |
| **Sparse / niche categories** | Content-Based | Item attributes override user co-rating sparsity |

### Why Differences Occur (Algorithmic Mechanisms)

- **CF**: Relies on collective behavior of similar users/items — powerful with dense data but fails in cold-start scenarios. Discovers patterns users might not explicitly state.
- **Content-Based**: Depends on feature engineering and TF-IDF similarity — avoids item cold-start but tends to overspecialize (recommends only items similar to past likes).
- **Knowledge-Based**: Deterministic and fully interpretable — never recommends outside explicit constraints but requires manual input and cannot discover unexpected preferences.

The optimal choice depends on data availability:
- **CF** for dense interaction logs
- **Content-Based** for rich product metadata
- **Knowledge-Based** for goal-driven sessions with clear constraints

---

## 10. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Generate fresh data (optional)
python data/generate_data.py

# Start the web app
python app.py

# Open in browser
open http://localhost:5000
```

### Usage

| Page | URL | What to Do |
|---|---|---|
| Home | `/` | Select a user, view their profile |
| Recommend | `/recommend` | Pick user, approach, method → get recommendations |
| Evaluate | `/evaluate` | Click "Run Evaluation" → see metrics + analysis |
| Create User | `/create` | Fill profile → rate 5 products → auto-generated ratings |
