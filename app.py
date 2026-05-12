import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json as json_module
import numpy as np
from flask import Flask, render_template, jsonify, request, Response, stream_with_context
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

for _ in cf_train.train_svd_generator():
    pass
for _ in cf_train.compute_slope_one_dev_generator():
    pass

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
                           active_page="home",
                           users=USER_OPTIONS,
                           categories=CATEGORIES,
                           brands=BRANDS,
                           approaches=APPROACHES)


@app.route("/recommend")
def recommend_page():
    return render_template("recommend.html",
                           active_page="recommend",
                           users=USER_OPTIONS,
                           categories=CATEGORIES,
                           brands=BRANDS,
                           approaches=APPROACHES)


@app.route("/evaluate")
def evaluate_page():
    return render_template("evaluation.html",
                           active_page="evaluate",
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


CF_METHOD_NAMES = ["user_based", "item_based", "svd", "knn", "slope_one"]
CF_METHOD_LABELS = {
    "user_based": "User-Based",
    "item_based": "Item-Based",
    "svd": "SVD",
    "knn": "KNN",
    "slope_one": "Slope One",
}

@app.route("/api/evaluate/cf/<method>")
def api_evaluate_cf(method):
    if method not in CF_METHOD_NAMES:
        return jsonify({"error": f"Unknown CF method: {method}"}), 400
    try:
        result = evaluator.evaluate_cf_method(method, cf_train, TEST, k=5)
        return jsonify(result)
    except Exception as e:
        return jsonify({"method": method, "error": str(e)})


@app.route("/api/evaluate/cf/<method>/stream")
def api_evaluate_cf_stream(method):
    if method not in ("svd", "slope_one"):
        return jsonify({"error": f"Streaming not supported for {method}"}), 400

    def generate():
        try:
            if method == "svd":
                gen = cf_train.train_svd_generator()
                if gen is not None:
                    for epoch, total in gen:
                        yield json_module.dumps({"type": "progress", "current": epoch, "total": total}) + "\n"

            elif method == "slope_one":
                gen = cf_train.compute_slope_one_dev_generator()
                if gen is not None:
                    for item, total in gen:
                        yield json_module.dumps({"type": "progress", "current": item, "total": total}) + "\n"

            yield json_module.dumps({"type": "phase", "label": "Evaluating users..."}) + "\n"
            result = evaluator.evaluate_cf_method(method, cf_train, TEST, k=5)
            yield json_module.dumps({"type": "result", "data": result}) + "\n"
        except Exception as e:
            yield json_module.dumps({"type": "error", "message": str(e)}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


@app.route("/api/evaluate/approaches")
def api_evaluate_approaches():
    try:
        evaluator.set_test_ratings(TEST)
        test_users = TEST["user_id"].unique()[:20]

        def approach_precision_recall(recommender_fn):
            precisions, recalls = [], []
            for uid in test_users:
                try:
                    recs = recommender_fn(uid)
                except Exception:
                    recs = []
                rec_items = [r[0] for r in recs]
                relevant = evaluator._get_relevant_for_user(uid)
                if relevant:
                    precisions.append(evaluator.precision_at_k(rec_items, relevant, 5))
                    recalls.append(evaluator.recall_at_k(rec_items, relevant, 5))
            return precisions, recalls

        def cf_recommender(uid):
            return cf_train.recommend("item_based", uid, n_recommendations=10)

        train_ratings = ratings[~ratings.index.isin(TEST.index)]
        def cb_recommender(uid):
            profile = train_ratings[
                (train_ratings["user_id"] == uid) & (train_ratings["rating"] >= 3.5)
            ]["product_id"].tolist()
            return cb.recommend("tfidf", user_profile_items=profile, n_recommendations=10)

        def kb_recommender(uid):
            prefs = get_user_preferences(users, uid)
            constraints = {
                "budget_min": prefs.get("budget_min", 0),
                "budget_max": prefs.get("budget_max", 999999),
                "category": list(prefs.get("preferred_categories", set())),
                "brand": list(prefs.get("favorite_brands", set())),
            }
            return kb.recommend("constraint", constraints=constraints, n_recommendations=10)

        results = []
        cf_p, cf_r = approach_precision_recall(cf_recommender)
        if cf_p:
            results.append({
                "approach": "Collaborative Filtering",
                "Precision@5": round(np.mean(cf_p), 4),
                "Recall@5": round(np.mean(cf_r), 4),
            })
        cb_p, cb_r = approach_precision_recall(cb_recommender)
        if cb_p:
            results.append({
                "approach": "Content-Based",
                "Precision@5": round(np.mean(cb_p), 4),
                "Recall@5": round(np.mean(cb_r), 4),
            })
        kb_p, kb_r = approach_precision_recall(kb_recommender)
        if kb_p:
            results.append({
                "approach": "Knowledge-Based",
                "Precision@5": round(np.mean(kb_p), 4),
                "Recall@5": round(np.mean(kb_r), 4),
            })

        best = max(results, key=lambda a: a.get("Precision@5", 0))["approach"] if results else None
        return jsonify({"approaches": results, "best_approach": best})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/products/filter")
def api_products_filter():
    cat = request.args.get("category")
    brand = request.args.get("brand")
    price_min = request.args.get("price_min", type=float)
    price_max = request.args.get("price_max", type=float)
    q = request.args.get("q", "").lower()
    filtered = products.copy()
    if cat:
        filtered = filtered[filtered["category"] == cat]
    if brand:
        filtered = filtered[filtered["brand"] == brand]
    if price_min is not None:
        filtered = filtered[filtered["price"] >= price_min]
    if price_max is not None:
        filtered = filtered[filtered["price"] <= price_max]
    if q:
        filtered = filtered[filtered["name"].str.lower().str.contains(q, na=False)]
    results = []
    for _, row in filtered.iterrows():
        results.append(get_product_info(row["product_id"]))
    return jsonify({
        "total": len(results),
        "products": results,
    })


@app.route("/api/user/<int:user_id>/preferences", methods=["PUT"])
def api_update_preferences(user_id):
    data = request.json
    user_idx = users[users["user_id"] == user_id].index
    if user_idx.empty:
        return jsonify({"error": "User not found"}), 404
    if "budget_min" in data:
        users.at[user_idx[0], "budget_min"] = data["budget_min"]
    if "budget_max" in data:
        users.at[user_idx[0], "budget_max"] = data["budget_max"]
    prefs = get_user_preferences(users, user_id)
    for u in USER_OPTIONS:
        if u["id"] == user_id:
            u["budget_min"] = float(prefs.get("budget_min", 0))
            u["budget_max"] = float(prefs.get("budget_max", 999999))
            break
    return jsonify({"success": True, "preferences": prefs})


@app.route("/htmx/recommend", methods=["POST"])
def htmx_recommend():
    data = request.json or request.form
    user_id = data.get("user_id", type=int)
    approach = data.get("approach", "cf")
    method = data.get("method", "user_based")
    n_recs = data.get("n", 10, type=int)

    if not user_id:
        return '<div class="empty-state"><div class="empty-icon">⚠️</div><p>Please select a user first.</p></div>'

    user_rated = get_user_rated_items(user_id)
    prefs = get_user_preferences(users, user_id)

    try:
        if approach == "cf":
            recs = cf.recommend(method, user_id, n_recommendations=n_recs)
        elif approach == "content":
            recs = cb.recommend(method, user_profile_items=user_rated, preferences=prefs, n_recommendations=n_recs)
        elif approach == "knowledge":
            constraints = {
                "budget_min": prefs.get("budget_min", 0),
                "budget_max": prefs.get("budget_max", 999999),
                "category": list(prefs.get("preferred_categories", set())),
                "brand": list(prefs.get("favorite_brands", set())),
            }
            recs = kb.recommend(method, constraints=constraints, preferences=prefs, n_recommendations=n_recs)
        else:
            return f'<div class="empty-state"><div class="empty-icon">❌</div><p>Unknown approach: {approach}</p></div>'
    except Exception as e:
        return f'<div class="empty-state"><div class="empty-icon">❌</div><p>{str(e)}</p></div>'

    if not recs:
        return '<div class="empty-state"><div class="empty-icon">📭</div><p>No recommendations found.</p></div>'

    html = '<div class="product-grid">'
    for pid, score in recs:
        product = get_product_info(pid)
        if not product:
            continue
        explanation = "Recommended based on your preferences."
        html += f'''
        <div class="product-card">
            <div class="product-icon">{get_category_icon(product["category"])}</div>
            <div class="product-name">{product["name"]}</div>
            <div class="product-meta">{product["brand"]} · {product["subcategory"]}</div>
            <div class="compact-row">
                <div class="product-price">${product["price"]:.2f}</div>
                <div class="product-rating">{stars_html(product["avg_rating"])} {product["avg_rating"]}</div>
            </div>
            <div class="product-explanation">{explanation}</div>
        </div>'''
    html += '</div>'
    return html


def get_category_icon(category):
    icons = {
        "Electronics": "💻", "Clothing": "👕", "Home & Kitchen": "🏠",
        "Books": "📚", "Sports": "⚽", "Beauty": "💄", "Toys": "🧸", "Automotive": "🚗"
    }
    return icons.get(category, "📦")


def stars_html(rating):
    f = int(rating)
    return "★" * f + "☆" * (5 - f)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
