# Knowledge-Based Recommendation Plan

## Overview

Knowledge-Based Recommendation uses explicit domain knowledge, rules, and user-provided constraints. It requires no historical rating data — ideal for cold-start scenarios.

## Method 1: Constraint-Based Filtering

**Concept**: Filter products based on explicit hard constraints provided by the user.

**Constraints**:
- Price range: min_price ≤ product_price ≤ max_price
- Category: product_category ∈ selected_categories
- Brand: product_brand ∈ preferred_brands
- Rating threshold: avg_rating ≥ min_rating

**Steps**:
1. Apply all constraints sequentially (AND logic)
2. Return remaining products
3. Sort by match score (how many constraints satisfied)
4. Recommend top-N

## Method 2: Rule-Based Recommendation

**Concept**: Apply domain-specific if-then rules to generate recommendations.

**Rules**:
```
IF user bought a laptop THEN recommend laptop accessories (mouse, bag, charger)
IF user bought a smartphone THEN recommend phone case, screen protector
IF user is looking at a book THEN recommend books by same author
IF user budget < $50 THEN recommend items from "Budget Picks" category
IF user searches "gaming" THEN recommend gaming-related items
```

**Steps**:
1. Check user's current interaction (cart, search, browse)
2. Fire matching rules
3. Aggregate and rank results
4. Recommend top-N

## Method 3: Utility-Based Recommendation

**Concept**: Score items based on weighted utility function derived from user preferences.

**Utility Function**:
```
U(item, user) = w₁ * price_score + w₂ * category_score + w₃ * brand_score + w₄ * rating_score
```

Where each score is normalized 0-1 and weights sum to 1.

**Steps**:
1. User sets preference weights (or use defaults)
2. Compute utility for all items
3. Sort by utility descending
4. Recommend top-N

## Implementation (`recommender/knowledge_based.py`)

```python
class KnowledgeBasedRecommender:
    def __init__(self, products_df):
        self.products = products_df
    
    def constraint_based(self, constraints, n=10): ...
    def rule_based(self, user_context, n=10): ...
    def utility_based(self, preferences, weights, n=10): ...
```

## Explanation Generation

- Constraint: "Recommended because it meets your criteria: budget ${budget}, category: {cat}"
- Rule: "Recommended because you bought a {trigger_item} — {rule_reason}"
- Utility: "Recommended with {score}% match to your preferences"

## When Each Performs Best

| Method | Best For |
|--------|----------|
| Constraint-Based | Users who know exactly what they want |
| Rule-Based | Users browsing specific categories |
| Utility-Based | Users who want trade-off recommendations |
