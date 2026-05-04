# Content-Based Recommendation Plan

## Overview

Content-Based Filtering recommends items based on the similarity between item features and user preferences. It does not rely on other users' data.

## Method 1: TF-IDF + Cosine Similarity

**Concept**: Vectorize product descriptions/categories using TF-IDF, compute cosine similarity between items and user profile.

**Steps**:
1. Build TF-IDF matrix from product descriptions + categories
2. Build user profile vector (average of TF-IDF vectors of items the user rated highly)
3. Compute cosine similarity between user profile and all items
4. Recommend top-N most similar items not yet interacted with

**Example**:
```
User liked: "Samsung Galaxy S24" (Electronics, Smartphone)
Recommended: "iPhone 15 Pro" (cosine sim: 0.87) because same category
```

## Method 2: Feature-Based Matching

**Concept**: Match items to user preferences using structured features (category, brand, price range).

**Steps**:
1. Extract user preferences: preferred_categories, favorite_brands, budget
2. Score each item based on feature match:
   - Category match: +40 points
   - Brand match: +30 points
   - Price in range: +20 points
   - Subcategory match: +10 points
3. Recommend top-N highest scoring items

**When it works best**: Cold-start users (no rating history), user explicitly sets preferences

## Implementation (`recommender/content_based.py`)

```python
class ContentBasedRecommender:
    def __init__(self, products_df):
        self.products = products_df
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
    
    def tfidf_recommend(self, user_id, user_preferences, n=10): ...
    def feature_match_recommend(self, user_preferences, n=10): ...
    def build_user_profile(self, rated_items): ...
```

## Explanation Generation

- TF-IDF: "Recommended because it matches items you've liked before (similarity: {score}%)"
- Feature: "Recommended because it matches your preferred category: {category}"
