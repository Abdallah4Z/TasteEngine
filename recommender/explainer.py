class Explainer:
    def __init__(self, products_df, users_df):
        self.products = products_df
        self.users = users_df
        self._product_cache = {}
        self._user_cache = {}
        for _, row in products_df.iterrows():
            self._product_cache[row["product_id"]] = row
        for _, row in users_df.iterrows():
            self._user_cache[row["user_id"]] = row

    def _get_product(self, product_id):
        return self._product_cache.get(product_id, {})

    def _get_user(self, user_id):
        return self._user_cache.get(user_id, {})

    def _fmt_score(self, score):
        if score >= 0.9:
            return "Excellent match"
        elif score >= 0.7:
            return "Strong match"
        elif score >= 0.5:
            return "Good match"
        elif score >= 0.3:
            return "Moderate match"
        else:
            return "Partial match"

    def _pref_list(self, user, key):
        raw = user.get(key, "")
        if raw is None or isinstance(raw, float):
            return []
        if isinstance(raw, str):
            return [p.strip() for p in raw.split(",") if p.strip()][:2]
        return list(raw)[:2] if raw else []

    def explain_cf(self, method, user_id, product_id, details=None):
        product = self._get_product(product_id)
        user = self._get_user(user_id)
        pname = product.get("name", f"Item #{product_id}")
        cat = product.get("category", "")
        brand = product.get("brand", "")
        price = product.get("price", 0)
        details = details or {}
        sim = min(details.get('sim_score', 0), 1.0)
        pref_cats = self._pref_list(user, "preferred_categories")

        templates = {
            "user_based": f"Users with {cat} taste also liked {pname}",
            "item_based": f"Matches your {cat} preferences — similar to items you've rated highly",
            "svd": f"Fits your profile: {brand}'s {cat} — top latent factor match" if brand and cat else f"Fits your profile: {cat} — top latent factor match",
            "knn": f"Popular among peers who also like {cat}" if cat else f"Popular among users with similar taste",
            "slope_one": f"Frequently chosen by users who liked the same {cat} products" if cat else f"Frequently chosen by users with your taste",
        }
        return templates.get(method, f"Recommended based on collaborative filtering")

    def explain_content(self, method, user_id, product_id, details=None):
        product = self._get_product(product_id)
        user = self._get_user(user_id)
        pname = product.get("name", f"Item #{product_id}")
        cat = product.get("category", "")
        brand = product.get("brand", "")
        sub = product.get("subcategory", "")
        details = details or {}
        score = min(details.get('score', 0), 1.0)
        pref_cats = self._pref_list(user, "preferred_categories")

        templates = {
            "tfidf": f"Content matches: {cat} / {sub} — strongly aligns with your past likes" if sub else f"Content matches your preferred {cat} items",
            "feature_match": f"Matches your preferences: {cat}, {brand}" if brand else f"Matches your preferred category: {cat}",
        }
        return templates.get(method, f"Recommended based on item features")

    def explain_knowledge(self, method, user_id, product_id, details=None):
        product = self._get_product(product_id)
        user = self._get_user(user_id)
        pname = product.get("name", f"Item #{product_id}")
        cat = product.get("category", "")
        brand = product.get("brand", "")
        price = product.get("price", 0)
        budget_max = details.get("budget_max", 0)
        details = details or {}
        pref_brands = self._pref_list(user, "favorite_brands")

        templates = {
            "constraint": f"Fits your criteria: ${price:.0f} within budget · {cat} · {brand}" if brand else f"Fits your criteria: ${price:.0f} within budget · {cat}",
            "rule": f"{cat} buyers commonly add {pname} — cross-sell match" if cat else f"Customers also purchased {pname}",
            "utility": f"Top utility score: {cat} × {brand} matches your preference weights" if brand else f"Top utility score: {cat} matches your preference weights",
        }
        return templates.get(method, f"Recommended based on your requirements")

    def get_explanation(self, approach, method, user_id, product_id, details=None):
        if approach == "cf":
            return self.explain_cf(method, user_id, product_id, details)
        elif approach == "content":
            return self.explain_content(method, user_id, product_id, details)
        elif approach == "knowledge":
            return self.explain_knowledge(method, user_id, product_id, details)
        return ""
