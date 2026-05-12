import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.helpers import load_data


class ContentBasedRecommender:
    def __init__(self, products_df):
        self.products = products_df.copy()
        self.products["text_features"] = (
            self.products["category"].fillna("") + " " +
            self.products["subcategory"].fillna("") + " " +
            self.products["brand"].fillna("") + " " +
            self.products["name"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.products["text_features"])
        self.product_ids = self.products["product_id"].values

    def tfidf_recommend(self, user_profile_items, n_recommendations=10):
        if not user_profile_items:
            return []

        profile_indices = []
        for pid in user_profile_items:
            mask = self.products["product_id"] == pid
            if mask.any():
                idx = self.products[mask].index[0]
                profile_indices.append(idx)

        if not profile_indices:
            return []

        profile_vector = np.asarray(self.tfidf_matrix[profile_indices].mean(axis=0))
        sim_scores = cosine_similarity(profile_vector, self.tfidf_matrix).flatten()

        exclude = set(profile_indices)
        ranked = sorted(
            [(i, sim_scores[i]) for i in range(len(sim_scores)) if i not in exclude],
            key=lambda x: x[1], reverse=True
        )

        results = []
        for idx, score in ranked[:n_recommendations]:
            results.append((int(self.product_ids[idx]), float(score)))
        return results

    def feature_match_recommend(self, preferences, n_recommendations=10):
        preferred_cats = preferences.get("preferred_categories", set())
        favorite_brands = preferences.get("favorite_brands", set())
        budget_min = preferences.get("budget_min", 0)
        budget_max = preferences.get("budget_max", 999999)

        scores = []
        for _, product in self.products.iterrows():
            score = 0
            if product["category"] in preferred_cats:
                score += 40
            if product["brand"] in favorite_brands:
                score += 30
            if budget_min <= product["price"] <= budget_max:
                score += 20
            scores.append((int(product["product_id"]), score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_recommendations]

    def recommend(self, method, user_profile_items=None, preferences=None, n_recommendations=10):
        if method == "tfidf":
            return self.tfidf_recommend(user_profile_items or [], n_recommendations)
        elif method == "feature_match":
            return self.feature_match_recommend(preferences or {}, n_recommendations)
        else:
            raise ValueError(f"Unknown method: {method}")
