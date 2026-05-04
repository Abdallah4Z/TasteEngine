import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from flask import Flask, render_template, jsonify, request
from utils.helpers import load_data, get_user_preferences
from recommender.collaborative import CollaborativeFiltering
from recommender.content_based import ContentBasedRecommender
from recommender.knowledge_based import KnowledgeBasedRecommender
from recommender.evaluation import Evaluator
from recommender.explainer import Explainer

app = Flask(__name__)

products, users, ratings = load_data()
cf = CollaborativeFiltering(ratings)
cb = ContentBasedRecommender(products)
kb = KnowledgeBasedRecommender(products)
explainer = Explainer(products, users)
evaluator = Evaluator(ratings)

TRAIN = ratings.sample(frac=0.8, random_state=42)
TEST = ratings.drop(TRAIN.index)
cf_train = CollaborativeFiltering(TRAIN)

USER_IDS = sorted(users["user_id"].tolist())
USER_OPTIONS = []
for _, u in users.iterrows():
    USER_OPTIONS.append({
        "id": int(u["user_id"]),
        "name": u["name"],
        "age": int(u["age"]),
        "categories": u["preferred_categories"].split(",") if isinstance(u["preferred_categories"], str) else [],
        "budget_min": float(u["budget_min"]),
        "budget_max": float(u["budget_max"]),
        "brands": u["favorite_brands"].split(",") if isinstance(u["favorite_brands"], str) else [],
    })

CATEGORIES = sorted(products["category"].unique().tolist())
BRANDS = sorted(products["brand"].unique().tolist())

APPROACHES = {
    "cf": {
        "label": "Collaborative Filtering",
        "icon": "🤝",
        "methods": [
            {"id": "user_based", "label": "User-Based CF"},
            {"id": "item_based", "label": "Item-Based CF"},
            {"id": "svd", "label": "SVD (Matrix Factorization)"},
            {"id": "knn", "label": "KNN-Based CF"},
            {"id": "slope_one", "label": "Slope One"},
        ],
    },
    "content": {
        "label": "Content-Based",
        "icon": "🏷️",
        "methods": [
            {"id": "tfidf", "label": "TF-IDF Similarity"},
            {"id": "feature_match", "label": "Feature Matching"},
        ],
    },
    "knowledge": {
        "label": "Knowledge-Based",
        "icon": "⚙️",
        "methods": [
            {"id": "constraint", "label": "Constraint-Based"},
            {"id": "rule", "label": "Rule-Based"},
            {"id": "utility", "label": "Utility-Based"},
        ],
    },
}


def get_product_info(product_id):
    row = products[products["product_id"] == product_id]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "id": int(row["product_id"]),
        "name": row["name"],
        "category": row["category"],
        "subcategory": row["subcategory"],
        "brand": row["brand"],
        "price": float(row["price"]),
        "avg_rating": float(row["avg_rating"]),
        "num_reviews": int(row["num_reviews"]),
    }


@app.route("/")
def index():
    return render_template("index.html",
                           users=USER_OPTIONS,
                           categories=CATEGORIES,
                           brands=BRANDS,
                           approaches=APPROACHES)


@app.route("/recommend")
def recommend_page():
    return render_template("recommend.html",
                           users=USER_OPTIONS,
                           categories=CATEGORIES,
                           brands=BRANDS,
                           approaches=APPROACHES)


@app.route("/evaluate")
def evaluate_page():
    return render_template("evaluation.html",
                           users=USER_OPTIONS,
                           categories=CATEGORIES,
                           brands=BRANDS,
                           approaches=APPROACHES)


@app.route("/api/users")
def api_users():
    return jsonify(USER_OPTIONS)


@app.route("/api/user/<int:user_id>")
def api_user(user_id):
    prefs = get_user_preferences(users, user_id)
    return jsonify(prefs)


@app.route("/api/products")
def api_products():
    cat = request.args.get("category")
    if cat:
        filtered = products[products["category"] == cat]
    else:
        filtered = products
    results = []
    for _, row in filtered.iterrows():
        results.append(get_product_info(row["product_id"]))
    return jsonify(results)


def get_user_rated_items(user_id):
    user_ratings = ratings[ratings["user_id"] == user_id]
    return user_ratings[user_ratings["rating"] >= 3.5]["product_id"].tolist()


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.json
    user_id = data.get("user_id")
    approach = data.get("approach")
    method = data.get("method")
    n_recs = data.get("n", 10)

    if not user_id or not approach or not method:
        return jsonify({"error": "Missing required parameters"}), 400

    user_rated = get_user_rated_items(user_id)
    prefs = get_user_preferences(users, user_id)

    try:
        if approach == "cf":
            recs = cf.recommend(method, user_id, n_recommendations=n_recs)
            explanations = []
            for pid, score in recs:
                details = {"sim_score": score, "count": 10}
                explanation = explainer.explain_cf(method, user_id, pid, details)
                product = get_product_info(pid)
                explanations.append({**product, "score": round(score, 4), "explanation": explanation})

        elif approach == "content":
            recs = cb.recommend(method, user_profile_items=user_rated, preferences=prefs, n_recommendations=n_recs)
            explanations = []
            for pid, score in recs:
                details = {"score": score}
                explanation = explainer.explain_content(method, user_id, pid, details)
                product = get_product_info(pid)
                explanations.append({**product, "score": round(score, 4), "explanation": explanation})

        elif approach == "knowledge":
            constraints = {
                "budget_min": prefs.get("budget_min", 0),
                "budget_max": prefs.get("budget_max", 999999),
                "category": list(prefs.get("preferred_categories", set())),
                "brand": list(prefs.get("favorite_brands", set())),
            }
            context = {
                "interacted_category": "",
                "preferred_categories": prefs.get("preferred_categories", set()),
                "budget_min": prefs.get("budget_min", 0),
                "budget_max": prefs.get("budget_max", 999999),
                "favorite_brands": prefs.get("favorite_brands", set()),
            }
            recs = kb.recommend(method, constraints=constraints, context=context,
                                preferences=prefs, n_recommendations=n_recs)
            explanations = []
            for pid, score in recs:
                details = {"score": score, "budget_max": prefs.get("budget_max", 0), "trigger_item": ""}
                explanation = explainer.explain_knowledge(method, user_id, pid, details)
                product = get_product_info(pid)
                explanations.append({**product, "score": round(score, 4), "explanation": explanation})

        else:
            return jsonify({"error": f"Unknown approach: {approach}"}), 400

        return jsonify({"recommendations": explanations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate")
def api_evaluate():
    try:
        cf_results = evaluator.compare_cf_methods(cf_train, TEST, k=5)

        best_cf = None
        best_rmse = float("inf")
        for r in cf_results:
            if "error" not in r and r["RMSE"] < best_rmse:
                best_rmse = r["RMSE"]
                best_cf = r["method"]

        approach_results = []

        def run_cf_for_user(uid):
            return cf.recommend("svd", uid, n_recommendations=10)

        cf_precisions = []
        cf_recalls = []
        test_users = TEST["user_id"].unique()[:20]
        for uid in test_users:
            recs = run_cf_for_user(uid)
            rec_items = [r[0] for r in recs]
            relevant = TEST[(TEST["user_id"] == uid) & (TEST["rating"] >= 3.5)]["product_id"].tolist()
            if relevant:
                cf_precisions.append(evaluator.precision_at_k(rec_items, relevant, 5))
                cf_recalls.append(evaluator.recall_at_k(rec_items, relevant, 5))
        if cf_precisions:
            approach_results.append({
                "approach": "Collaborative Filtering",
                "Precision@5": round(np.mean(cf_precisions), 4),
                "Recall@5": round(np.mean(cf_recalls), 4),
            })

        return jsonify({
            "cf_methods": cf_results,
            "best_cf_method": best_cf,
            "approaches": approach_results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
