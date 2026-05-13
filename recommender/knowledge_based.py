import numpy as np
import pandas as pd


class KnowledgeBasedRecommender:
    """Knowledge-Based recommendation engine.
    
    Uses explicit domain knowledge, rules, and constraints rather than
    statistical patterns. Includes constraint-based filtering, rule-based
    cross-selling, and multi-attribute utility scoring.
    """

    def __init__(self, products_df):
        """Store product catalog for filtering and scoring."""
        self.products = products_df.copy()

    def constraint_based(self, constraints, n_recommendations=10):
        """Constraint-based filtering.
        
        Sequentially filters products by user constraints (budget, category,
        brand, min_rating, subcategory). Products matching ALL hard constraints
        are sorted by avg_rating. Works with zero user history.
        """
        filtered = self.products.copy()

        if "budget_max" in constraints:
            filtered = filtered[filtered["price"] <= constraints["budget_max"]]
        if "budget_min" in constraints:
            filtered = filtered[filtered["price"] >= constraints["budget_min"]]
        if "category" in constraints and constraints["category"]:
            filtered = filtered[filtered["category"].isin(constraints["category"])]

        brand_match = None
        if "brand" in constraints and constraints["brand"]:
            brand_match = filtered[filtered["brand"].isin(constraints["brand"])]
        if brand_match is not None and not brand_match.empty:
            filtered = brand_match

        if "min_rating" in constraints:
            filtered = filtered[filtered["avg_rating"] >= constraints["min_rating"]]
        if "subcategory" in constraints and constraints["subcategory"]:
            sub_match = filtered[filtered["subcategory"].isin(constraints["subcategory"])]
            if not sub_match.empty:
                filtered = sub_match

        filtered = filtered.sort_values("avg_rating", ascending=False)
        results = [(int(row["product_id"]), float(row["avg_rating"])) for _, row in filtered.head(n_recommendations).iterrows()]
        return results

    def rule_based(self, context, n_recommendations=10):
        """Rule-based recommendation with domain business rules.
        
        Uses pre-defined cross-selling rules (e.g., laptop buyers also buy
        accessories). Scores products based on interaction context, preferred
        categories, budget, and brand preferences. Mimics real-world e-commerce.
        """
        rules = {
            "laptop": {"category": "Electronics", "subcategory": "Laptops"},
            "smartphone": {"category": "Electronics", "subcategory": "Smartphones"},
            "book": {"category": "Books"},
            "camera": {"category": "Electronics", "subcategory": "Cameras"},
            "headphone": {"category": "Electronics", "subcategory": "Headphones"},
        }

        accessoires_map = {
            "Laptops": ["Laptop Bag", "Mouse", "Keyboard", "Cooling Pad"],
            "Smartphones": ["Phone Case", "Screen Protector", "Charger", "Power Bank"],
            "Cameras": ["Camera Bag", "Tripod", "Memory Card", "Lens Kit"],
            "Headphones": ["Headphone Stand", "Cable", "Carrying Case", "Ear Pads"],
        }

        interacted_category = context.get("interacted_category", "")
        scores = []

        for _, product in self.products.iterrows():
            score = 0
            if interacted_category and product["category"] == "Electronics":
                if product["subcategory"] in accessoires_map.get(interacted_category, []):
                    score += 50
            if context.get("preferred_categories") and product["category"] in context["preferred_categories"]:
                score += 20
            if context.get("budget_min", 0) <= product["price"] <= context.get("budget_max", 999999):
                score += 15
            if context.get("favorite_brands") and product["brand"] in context["favorite_brands"]:
                score += 10
            scores.append((int(product["product_id"]), score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_recommendations]

    def utility_based(self, preferences, weights=None, n_recommendations=10):
        """Multi-attribute utility-based recommendation.
        
        Computes a weighted utility score for each product:
        - Price utility: closeness to midpoint of user's budget (20%)
        - Category utility: matches preferred categories? (30%)
        - Brand utility: matches favorite brands? (20%)
        - Rating utility: normalized product rating (30%)
        Returns top-N products by total utility.
        """
        if weights is None:
            weights = {"price": 0.2, "category": 0.3, "brand": 0.2, "rating": 0.3}

        price_min = preferences.get("budget_min", 0)
        price_max = preferences.get("budget_max", 999999)
        pref_cats = preferences.get("preferred_categories", set())
        fav_brands = preferences.get("favorite_brands", set())

        max_price = self.products["price"].max()
        min_price = self.products["price"].min()
        max_rating = self.products["avg_rating"].max()

        scores = []
        for _, product in self.products.iterrows():
            u = 0.0

            if price_max > price_min:
                price_score = 1.0 - abs(product["price"] - (price_min + price_max) / 2) / ((price_max - price_min) / 2)
                price_score = max(0, min(1, price_score))
            else:
                price_score = 1.0 if price_min <= product["price"] <= price_max else 0.0
            u += weights["price"] * price_score

            cat_score = 1.0 if product["category"] in pref_cats else 0.0
            u += weights["category"] * cat_score

            brand_score = 1.0 if product["brand"] in fav_brands else 0.0
            u += weights["brand"] * brand_score

            rating_score = product["avg_rating"] / max_rating if max_rating > 0 else 0
            u += weights["rating"] * rating_score

            scores.append((int(product["product_id"]), round(u, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_recommendations]

    def recommend(self, method, constraints=None, context=None, preferences=None, n_recommendations=10):
        """Router: dispatches to the appropriate knowledge-based method."""
        if method == "constraint":
            return self.constraint_based(constraints or {}, n_recommendations)
        elif method == "rule":
            return self.rule_based(context or {}, n_recommendations)
        elif method == "utility":
            return self.utility_based(preferences or {}, n_recommendations=n_recommendations)
        else:
            raise ValueError(f"Unknown method: {method}")
