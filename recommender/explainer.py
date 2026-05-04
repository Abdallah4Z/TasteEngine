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

    def explain_cf(self, method, user_id, product_id, details=None):
        product = self._get_product(product_id)
        product_name = product.get("name", f"Item #{product_id}")
        details = details or {}

        sim = min(details.get('sim_score', 0), 1.0)
        templates = {
            "user_based": f"Recommended because {details.get('count', 'several')} similar users (avg {sim*100:.0f}% match) also purchased {product_name}",
            "item_based": f"Recommended because it is {sim*100:.0f}% similar to items you liked",
            "svd": f"Recommended based on your taste profile — {product_name} matches your latent preferences",
            "knn": f"Recommended because {details.get('count', 'several')} nearest neighbors with similar taste purchased this",
            "slope_one": f"Recommended based on average preference patterns for {product_name}",
        }
        return templates.get(method, f"Recommended based on collaborative filtering")

    def explain_content(self, method, user_id, product_id, details=None):
        product = self._get_product(product_id)
        product_name = product.get("name", f"Item #{product_id}")
        category = product.get("category", "")
        details = details or {}

        score = min(details.get('score', 0), 1.0)
        templates = {
            "tfidf": f"Recommended because {product_name} matches items you've liked before (content similarity: {score*100:.0f}%)",
            "feature_match": f"Recommended because it matches your preferred category: {category}",
        }
        return templates.get(method, f"Recommended based on item features")

    def explain_knowledge(self, method, user_id, product_id, details=None):
        product = self._get_product(product_id)
        product_name = product.get("name", f"Item #{product_id}")
        brand = product.get("brand", "")
        price = product.get("price", 0)
        details = details or {}

        templates = {
            "constraint": f"Recommended because {product_name} meets your criteria: budget ${details.get('budget_max', 'N/A')}, brand: {brand}",
            "rule": f"Recommended because customers who bought {details.get('trigger_item', 'similar items')} also purchased {product_name}",
            "utility": f"Recommended with {details.get('score', 0)*100:.1f}% utility match based on your preference weights",
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
