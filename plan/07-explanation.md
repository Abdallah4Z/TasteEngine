# Explanation & Interpretability Plan

## Overview

Every recommendation must include a clear, human-readable explanation of why it was suggested. Explanations differ by approach and build user trust.

## CF Explanation Templates

| Method | Template | Example |
|--------|----------|---------|
| User-Based | "Recommended because **{sim_user_count} similar users** (avg {sim_score}% match) also purchased this item" | "Recommended because 12 similar users (avg 84% match) also purchased this item" |
| Item-Based | "Recommended because it is **{sim_score}% similar** to {item_name} which you liked" | "Recommended because it is 91% similar to 'Samsung Galaxy S24' which you liked" |
| SVD | "Recommended based on your **latent preference profile** — this item scores highly for your taste pattern" | "Recommended based on your latent preference profile — this item scores highly for your taste pattern" |
| KNN | "Recommended because **{k} nearest neighbors** with similar taste purchased this" | "Recommended because 15 nearest neighbors with similar taste purchased this" |

## Content-Based Explanation Templates

| Method | Template | Example |
|--------|----------|---------|
| TF-IDF | "Recommended because it **matches items you've liked** (content similarity: {score}%)" | "Recommended because it matches items you've liked (content similarity: 87%)" |
| Feature Match | "Recommended because it matches your **preferred category: {category}**" | "Recommended because it matches your preferred category: Electronics" |

## Knowledge-Based Explanation Templates

| Method | Template | Example |
|--------|----------|---------|
| Constraint | "Recommended because it **meets your criteria**: budget ${budget}, brand: {brand}" | "Recommended because it meets your criteria: budget $500, brand: Samsung" |
| Rule | "Recommended because **you purchased {trigger_item}** — customers who buy this also buy {item}" | "Recommended because you purchased a Laptop — customers who buy this also buy Laptop Bag" |
| Utility | "Recommended with **{score}% utility match** based on your preference weights" | "Recommended with 92% utility match based on your preference weights" |

## Implementation (`recommender/explainer.py`)

```python
class Explainer:
    def __init__(self, products_df, users_df):
        self.products = products_df
        self.users = users_df
    
    def explain_cf(self, method, user_id, item_id, details): ...
    def explain_content(self, method, user_id, item_id, details): ...
    def explain_knowledge(self, method, user_id, item_id, details): ...
    
    def get_explanation(self, approach, method, user_id, item_id, details): ...
```

## UI Display

Explanations appear below each recommended item as a small badge/text:

```
[Product Image]
Samsung Galaxy S24 — $799.99
★★★★☆ (4.2)
└─ 💡 Recommended because 12 similar users purchased this (CF - User-Based)
```

Icons per approach:
- CF: 👥 or 🤝
- Content-Based: 🏷️ or 📋
- Knowledge-Based: ⚙️ or 🎯
