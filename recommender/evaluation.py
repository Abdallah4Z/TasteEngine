import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Evaluator:
    """Comprehensive evaluation suite for recommendation methods.
    
    Provides 6 metrics: RMSE, MAE, Precision@k, Recall@k, F1@k, Coverage.
    Supports comparing CF methods against each other and evaluating
    different approaches (CF vs Content-Based vs Knowledge-Based).
    """

    def __init__(self, ratings_df, predictions_df=None):
        """Initialize with full ratings dataset."""
        self.ratings = ratings_df
        self.predictions = predictions_df
        self._test_ratings = None

    def set_test_ratings(self, test_ratings):
        """Set the held-out test set for evaluation."""
        self._test_ratings = test_ratings

    def _get_relevant_for_user(self, user_id, rating_threshold=3.5):
        """Get ground-truth relevant items for a user (ratings ≥ threshold in test set)."""
        if self._test_ratings is not None:
            relevant = self._test_ratings[
                (self._test_ratings["user_id"] == user_id) &
                (self._test_ratings["rating"] >= rating_threshold)
            ]["product_id"].tolist()
        else:
            relevant = self.ratings[
                (self.ratings["user_id"] == user_id) &
                (self.ratings["rating"] >= rating_threshold)
            ]["product_id"].tolist()
        return relevant

    def rmse(self, y_true, y_pred):
        """Root Mean Squared Error — penalizes large errors quadratically. Lower is better."""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def mae(self, y_true, y_pred):
        """Mean Absolute Error — average absolute difference. Lower is better."""
        return float(mean_absolute_error(y_true, y_pred))

    def precision_at_k(self, recommended_items, relevant_items, k=5):
        """Of top-k recommended items, how many are relevant? Range [0,1], higher is better."""
        top_k = recommended_items[:k]
        hits = len(set(top_k) & set(relevant_items))
        return hits / k if k > 0 else 0

    def recall_at_k(self, recommended_items, relevant_items, k=5):
        """Of all relevant items, how many appear in top-k? Range [0,1], higher is better."""
        top_k = recommended_items[:k]
        hits = len(set(top_k) & set(relevant_items))
        return hits / len(relevant_items) if len(relevant_items) > 0 else 0

    def f1_at_k(self, recommended_items, relevant_items, k=5):
        """Harmonic mean of Precision@k and Recall@k. Range [0,1], higher is better."""
        p = self.precision_at_k(recommended_items, relevant_items, k)
        r = self.recall_at_k(recommended_items, relevant_items, k)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def coverage(self, recommended_items_list, total_items):
        """Fraction of total product catalog recommended to at least one user. Higher is better."""
        recommended_set = set()
        for items in recommended_items_list:
            recommended_set.update(items)
        return len(recommended_set) / total_items if total_items > 0 else 0

    def evaluate_cf_method(self, method_name, cf_instance, test_ratings, k=5, rating_threshold=3.5, max_users=20):
        """Full evaluation of a single CF method.
        
        For each test user: gets recommendations, compares predicted ratings
        to actual (→ RMSE, MAE), checks top-k relevance (→ Precision, Recall,
        F1), and measures catalog coverage. Returns dict with all 6 metrics.
        """
        y_true = []
        y_pred = []

        all_recommended = []
        all_relevant_counts = []

        user_ids = test_ratings["user_id"].unique()[:max_users]
        for user_id in user_ids:
            user_test = test_ratings[test_ratings["user_id"] == user_id]
            actual_items = user_test[user_test["rating"] >= rating_threshold]["product_id"].tolist()

            try:
                recs = cf_instance.recommend(method_name, user_id, n_recommendations=20)
            except Exception:
                recs = []
            rec_items = [r[0] for r in recs]
            all_recommended.append(rec_items)
            all_relevant_counts.append(len(actual_items))

            for _, row in user_test.iterrows():
                y_true.append(row["rating"])
                found = False
                for rec_id, pred_rating in recs:
                    if rec_id == row["product_id"]:
                        y_pred.append(pred_rating)
                        found = True
                        break
                if not found:
                    y_pred.append(2.5)

        user_precisions = []
        user_recalls = []
        user_f1s = []
        for i, user_id in enumerate(user_ids):
            user_test = test_ratings[test_ratings["user_id"] == user_id]
            relevant = user_test[user_test["rating"] >= rating_threshold]["product_id"].tolist()
            if not relevant or i >= len(all_recommended):
                continue
            rec_items = all_recommended[i]
            user_precisions.append(self.precision_at_k(rec_items, relevant, k))
            user_recalls.append(self.recall_at_k(rec_items, relevant, k))
            user_f1s.append(self.f1_at_k(rec_items, relevant, k))

        total_items = len(cf_instance.item_ids) if hasattr(cf_instance, "item_ids") else None

        return {
            "method": method_name,
            "RMSE": self.rmse(y_true[-len(y_pred):], y_pred),
            "MAE": self.mae(y_true[-len(y_pred):], y_pred),
            f"Precision@{k}": round(np.mean(user_precisions), 4) if user_precisions else 0,
            f"Recall@{k}": round(np.mean(user_recalls), 4) if user_recalls else 0,
            f"F1@{k}": round(np.mean(user_f1s), 4) if user_f1s else 0,
            "Coverage": self.coverage(all_recommended, total_items) if total_items else 0,
        }

    def compare_cf_methods(self, cf_instance, test_ratings, k=5):
        """Evaluate all 5 CF methods and return comparison list."""
        methods = cf_instance.get_all_methods()
        results = []
        for method in methods:
            try:
                result = self.evaluate_cf_method(method, cf_instance, test_ratings, k)
                results.append(result)
            except Exception as e:
                results.append({"method": method, "error": str(e)})
        return results

    def evaluate_approach(self, approach_name, recommender_fn, test_users, products_df, k=5):
        """Evaluate a recommendation approach by computing Precision@k, Recall@k, and Coverage."""

        all_recommended = []
        precisions = []
        recalls = []

        for user_id in test_users:
            try:
                recs = recommender_fn(user_id)
            except Exception:
                recs = []
            rec_items = [r[0] for r in recs]
            all_recommended.append(rec_items)

            if hasattr(self, "_get_relevant_for_user"):
                relevant = self._get_relevant_for_user(user_id)
                precisions.append(self.precision_at_k(rec_items, relevant, k))
                recalls.append(self.recall_at_k(rec_items, relevant, k))

        return {
            "approach": approach_name,
            f"Precision@{k}": round(np.mean(precisions), 4) if precisions else 0,
            f"Recall@{k}": round(np.mean(recalls), 4) if recalls else 0,
            "Coverage": self.coverage(all_recommended, len(products_df)),
        }

    def compare_approaches(self, cf_instance, cb_instance, kb_instance, test_ratings, products_df, k=5):
        """Compare all 3 approaches (CF, Content-Based, Knowledge-Based) on test users."""

        self.set_test_ratings(test_ratings)
        test_users = test_ratings["user_id"].unique()[:20]

        def cf_recommender(uid):
            return cf_instance.recommend("item_based", uid, n_recommendations=10)

        results = []
        try:
            cf_result = self.evaluate_approach("Collaborative Filtering", cf_recommender, test_users, products_df, k)
            results.append(cf_result)
        except Exception as e:
            results.append({"approach": "Collaborative Filtering", "error": str(e)})

        def cb_recommender(uid):
            profile = test_ratings[
                (test_ratings["user_id"] == uid) & (test_ratings["rating"] >= 3.5)
            ]["product_id"].tolist()
            return cb_instance.recommend("tfidf", user_profile_items=profile, n_recommendations=10)

        try:
            cb_result = self.evaluate_approach("Content-Based", cb_recommender, test_users, products_df, k)
            results.append(cb_result)
        except Exception as e:
            results.append({"approach": "Content-Based", "error": str(e)})

        def kb_recommender(uid):
            user_row = self.ratings[self.ratings["user_id"] == uid]
            if user_row.empty:
                return []
            prefs = {}
            prefs["budget_min"] = 0
            prefs["budget_max"] = 999999
            prefs["preferred_categories"] = set()
            prefs["favorite_brands"] = set()
            constraints = {
                "budget_min": prefs.get("budget_min", 0),
                "budget_max": prefs.get("budget_max", 999999),
            }
            return kb_instance.recommend("constraint", constraints=constraints, n_recommendations=10)

        try:
            kb_result = self.evaluate_approach("Knowledge-Based", kb_recommender, test_users, products_df, k)
            results.append(kb_result)
        except Exception as e:
            results.append({"approach": "Knowledge-Based", "error": str(e)})

        return results
