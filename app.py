import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json as json_module
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from utils.helpers import load_data, get_user_preferences, DATA_DIR
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
    """Lookup a product by ID and return a formatted dict with all fields."""
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
    """Landing page with user selector and approach overview."""
    return render_template("index.html",
                           active_page="home",
                           users=USER_OPTIONS,
                           categories=CATEGORIES,
                           brands=BRANDS,
                           approaches=APPROACHES)


@app.route("/recommend")
def recommend_page():
    """Recommendation page with approach/method selection and product grid."""
    return render_template("recommend.html",
                           active_page="recommend",
                           users=USER_OPTIONS,
                           categories=CATEGORIES,
                           brands=BRANDS,
                           approaches=APPROACHES)


@app.route("/evaluate")
def evaluate_page():
    """Evaluation dashboard with tables, charts, and reasoning analysis."""
    return render_template("evaluation.html",
                           active_page="evaluate",
                           users=USER_OPTIONS,
                           categories=CATEGORIES,
                           brands=BRANDS,
                           approaches=APPROACHES)


@app.route("/api/users")
def api_users():
    """List all users with their profile data (name, age, categories, budget, brands)."""
    return jsonify(USER_OPTIONS)


@app.route("/api/user/<int:user_id>")
def api_user(user_id):
    """Get a single user's preferences."""
    prefs = get_user_preferences(users, user_id)
    return jsonify(prefs)


@app.route("/api/products")
def api_products():
    """List all products, optionally filtered by category."""
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
    """Get product IDs the user rated >= 3.5 (their 'liked' items for content-based)."""
    user_ratings = ratings[ratings["user_id"] == user_id]
    return user_ratings[user_ratings["rating"] >= 3.5]["product_id"].tolist()


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Core recommendation endpoint.
    
    Accepts: user_id, approach (cf|content|knowledge), method, n (count).
    Returns: list of {product info, score, explanation}.
    Routes to the appropriate engine (CF, CB, or KB) and generates explanations.
    """
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


def _generate_analysis(cf_results, approach_results, best_cf, best_approach):
    """Generate data-driven analysis text for the evaluation page.
    
    Examines actual metrics to produce dynamic reasoning for:
    - Which CF method performed best and why
    - Which approach performed best and why
    - Under what conditions each excels
    - Why algorithmic differences exist
    Uses real metric values (not hardcoded text) for the explanations.
    """
    analysis = {}

    method_labels = {
        "user_based": "User-Based CF", "item_based": "Item-Based CF",
        "svd": "SVD", "knn": "KNN-Based CF", "slope_one": "Slope One",
    }

    if cf_results:
        valid = [r for r in cf_results if "error" not in r]
        if valid and best_cf:
            best = next((r for r in valid if r["method"] == best_cf), None)
            rmse_best = best["RMSE"] if best else 0

            sorted_rmse = sorted(valid, key=lambda x: x["RMSE"])
            sorted_p = sorted(valid, key=lambda x: x.get("Precision@5", 0), reverse=True)
            sorted_r = sorted(valid, key=lambda x: x.get("Recall@5", 0), reverse=True)
            sorted_cov = sorted(valid, key=lambda x: x.get("Coverage", 0), reverse=True)

            method_best_rmse = method_labels.get(sorted_rmse[0]["method"], sorted_rmse[0]["method"])
            method_best_p = method_labels.get(sorted_p[0]["method"], sorted_p[0]["method"])
            method_best_cov = method_labels.get(sorted_cov[0]["method"], sorted_cov[0]["method"])

            explanations = []
            if best_cf in ("svd", "knn"):
                explanations.append(f"<b>{method_labels.get(best_cf, best_cf)}</b> wins with lowest RMSE ({rmse_best:.4f}) because it captures latent interaction patterns, reducing prediction error across sparse rating data.")
            elif best_cf in ("item_based",):
                explanations.append(f"<b>{method_labels.get(best_cf, best_cf)}</b> wins with lowest RMSE ({rmse_best:.4f}) because item-item similarities are more stable than user-user similarities in this dataset.")
            else:
                explanations.append(f"<b>{method_labels.get(best_cf, best_cf)}</b> achieves the lowest RMSE ({rmse_best:.4f}) among all CF methods.")
            explanations.append(f"For Precision@5, <b>{method_best_p}</b> leads — meaning it recommends relevant items more consistently in the top-5.")
            explanations.append(f"For Coverage, <b>{method_best_cov}</b> explores the catalog most broadly, reducing the filter bubble effect.")

            for r in valid:
                m = r["method"]
                label = method_labels.get(m, m)
                if m == "item_based" and r.get("Coverage", 0) > 0.7:
                    explanations.append(f"<b>{label}</b> has high coverage ({r['Coverage']:.2f}) because every item gets compared to every other item.")
                elif m == "svd":
                    explanations.append(f"<b>{label}</b> compresses the matrix into latent factors — fast at prediction but may miss niche patterns ({r.get('Precision@5', 0):.3f} precision).")
                elif m == "user_based" and r.get("Precision@5", 0) > 0.02:
                    explanations.append(f"<b>{label}</b> finds lookalike users; performance depends on how many neighbors share the user's taste.")
                elif m == "knn":
                    explanations.append(f"<b>{label}</b> uses nearest neighbors; robust but can be noisy with sparse data.")
                elif m == "slope_one":
                    explanations.append(f"<b>{label}</b> is simple and fast but assumes linear rating deviations, which may oversimplify.")

            analysis["method"] = " ".join(explanations)

    if approach_results:
        valid_app = [a for a in approach_results if "error" not in a]
        if valid_app and best_approach:
            best_a = next((a for a in valid_app if a["approach"] == best_approach), None)
            prec_val = best_a["Precision@5"] if best_a else 0

            explanations = []
            if best_approach == "Collaborative Filtering":
                explanations.append(f"<b>Collaborative Filtering</b> ranks best (Precision@5: {prec_val:.4f}) because the dataset has dense rating patterns (avg 40 ratings/user), giving CF enough signal to find meaningful user-user/item-item similarities.")
            elif best_approach == "Content-Based":
                explanations.append(f"<b>Content-Based</b> ranks best (Precision@5: {prec_val:.4f}) because products have rich text features, making TF-IDF similarity highly discriminative for this dataset.")
            else:
                explanations.append(f"<b>Knowledge-Based</b> ranks best (Precision@5: {prec_val:.4f}) because users have well-defined category/brand/budget preferences that map directly to product attributes.")
            explanations.append("CF leverages collective behavior, Content-Based uses item features, and Knowledge-Based applies explicit rules — their relative performance depends on data density, feature richness, and constraint specificity.")

            for a in valid_app:
                if a["approach"] == "Content-Based" and a.get("Precision@5", 0) < 0.01:
                    explanations.append("Content-Based shows low precision here because TF-IDF requires a user profile of liked items — cold-start users with few ratings get weaker recommendations.")
                if a["approach"] == "Knowledge-Based" and a.get("Precision@5", 0) < 0.01:
                    explanations.append("Knowledge-Based precision is limited because it applies hard constraints — if a user has narrow preferences, few products pass the filter.")
                if a["approach"] == "Collaborative Filtering" and a.get("Recall@5", 0) > 0.03:
                    explanations.append("CF's higher recall suggests it casts a wider net, surfacing relevant items the user might not have explicitly searched for (serendipity).")

            analysis["approach"] = " ".join(explanations)

    analysis["conditions"] = (
        "<b>• Dense user-item ratings:</b> Collaborative Filtering performs best (leverages peer behavior).<br>"
        "<b>• Cold-start user (no history):</b> Knowledge-Based excels (no ratings needed, just constraints).<br>"
        "<b>• Cold-start item (new product):</b> Content-Based works well (matches item features to user profile).<br>"
        "<b>• Explicit constraints (budget, brand):</b> Knowledge-Based gives precise, explainable results.<br>"
        "<b>• Sparse / niche categories:</b> Content-Based overcomes sparsity by relying on item attributes rather than user co-ratings."
    )
    analysis["why"] = (
        "Differences stem from each approach's fundamental mechanism: "
        "<b>CF</b> relies on the collective behavior of similar users/items — powerful with dense data but fails in cold-start scenarios. "
        "<b>Content-Based</b> depends on feature engineering and TF-IDF similarity — it avoids the cold-start problem for items but tends to overspecialize, recommending only items similar to what the user already liked. "
        "<b>Knowledge-Based</b> is deterministic and fully interpretable — it never recommends outside the user's explicit constraints but requires manual input and cannot discover unexpected preferences. "
        "The optimal choice hinges on data availability: CF for dense interaction logs, Content-Based for rich product metadata, and Knowledge-Based for goal-driven sessions."
    )
    return analysis


@app.route("/api/evaluate")
def api_evaluate():
    """Full evaluation endpoint.
    
    Runs all 5 CF methods + all 3 approaches across test users.
    Returns: cf_methods (6 metrics each), best_cf_method, approaches
    (3 approaches compared), best_approach, and analysis (data-driven text).
    """
    try:
        cf_results = evaluator.compare_cf_methods(cf_train, TEST, k=5)
        best_cf = None
        best_rmse = float("inf")
        for r in cf_results:
            if "error" not in r and r["RMSE"] < best_rmse:
                best_rmse = r["RMSE"]
                best_cf = r["method"]

        evaluator.set_test_ratings(TEST)
        test_users = TEST["user_id"].unique()[:20]

        def _ap_r(recommender_fn):
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

        def _cf_r(uid):
            return cf_train.recommend("item_based", uid, n_recommendations=10)

        train_ratings = ratings[~ratings.index.isin(TEST.index)]

        def _cb_r(uid):
            profile = train_ratings[
                (train_ratings["user_id"] == uid) & (train_ratings["rating"] >= 3.5)
            ]["product_id"].tolist()
            return cb.recommend("tfidf", user_profile_items=profile, n_recommendations=10)

        def _kb_r(uid):
            prefs = get_user_preferences(users, uid)
            constraints = {
                "budget_min": prefs.get("budget_min", 0),
                "budget_max": prefs.get("budget_max", 999999),
                "category": list(prefs.get("preferred_categories", set())),
                "brand": list(prefs.get("favorite_brands", set())),
            }
            return kb.recommend("constraint", constraints=constraints, n_recommendations=10)

        approach_results = []
        for name, fn in [("Collaborative Filtering", _cf_r), ("Content-Based", _cb_r), ("Knowledge-Based", _kb_r)]:
            p, r = _ap_r(fn)
            if p:
                approach_results.append({
                    "approach": name,
                    "Precision@5": round(np.mean(p), 4),
                    "Recall@5": round(np.mean(r), 4),
                })

        best_approach = max(approach_results, key=lambda a: a.get("Precision@5", 0))["approach"] if approach_results else None
        analysis = _generate_analysis(cf_results, approach_results, best_cf, best_approach)

        return jsonify({
            "cf_methods": cf_results,
            "best_cf_method": best_cf,
            "approaches": approach_results,
            "best_approach": best_approach,
            "analysis": analysis,
        })
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
    """Evaluate a single CF method. Returns all 6 metrics."""
    if method not in CF_METHOD_NAMES:
        return jsonify({"error": f"Unknown CF method: {method}"}), 400
    try:
        result = evaluator.evaluate_cf_method(method, cf_train, TEST, k=5)
        return jsonify(result)
    except Exception as e:
        return jsonify({"method": method, "error": str(e)})


@app.route("/api/evaluate/cf/<method>/stream")
def api_evaluate_cf_stream(method):
    """Streamed evaluation for slow CF methods (SVD, Slope One).
    
    Returns NDJSON with progress events, then the result.
    Allows the frontend to show a progress bar during long evaluations.
    """
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
    """Evaluate all 3 approaches (CF, Content-Based, Knowledge-Based) with Precision@5 and Recall@5."""
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
    """Advanced product search/filter endpoint.
    
    Supports filtering by: category, brand, price_min, price_max, text search (q).
    Returns matching products with total count.
    """
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
    return jsonify({"success": True, "preferences": {
        "budget_min": prefs.get("budget_min", 0),
        "budget_max": prefs.get("budget_max", 0),
        "name": prefs.get("name", ""),
        "age": prefs.get("age", 0),
    }})


@app.route("/api/users", methods=["POST"])
def api_create_user():
    """Create a new user with profile data (name, age, categories, brands, budget).
    
    Saves to users.csv for persistence across restarts.
    Returns the new user_id.
    """
    global users
    data = request.json
    new_id = int(users["user_id"].max() + 1)
    new_name = data.get("name", f"User_{new_id}")
    new_row = pd.DataFrame([{
        "user_id": new_id,
        "name": new_name,
        "age": int(data.get("age", 25)),
        "preferred_categories": ",".join(data.get("categories", [])),
        "favorite_brands": ",".join(data.get("brands", [])),
        "budget_min": float(data.get("budget_min", 0)),
        "budget_max": float(data.get("budget_max", 500)),
    }])
    users = pd.concat([users, new_row], ignore_index=True)
    users.to_csv(os.path.join(DATA_DIR, "users.csv"), index=False)
    USER_OPTIONS.append({
        "id": new_id,
        "name": new_name,
        "age": int(data.get("age", 25)),
        "categories": data.get("categories", []),
        "brands": data.get("brands", []),
        "budget_min": float(data.get("budget_min", 0)),
        "budget_max": float(data.get("budget_max", 500)),
    })
    return jsonify({"success": True, "user_id": new_id, "name": new_name})


@app.route("/create")
def create_user_page():
    """Create-user page with two-step flow: profile form → rate 5 products → auto-generate ratings."""
    return render_template("create.html",
                           categories=CATEGORIES,
                           brands=BRANDS,
                           products=products.to_dict("records"))


@app.route("/api/create/step1", methods=["POST"])
def api_create_step1():
    """Create user and optionally save ratings (step 1 of create-user flow).
    
    If 'ratings' are provided, auto-generates additional ratings to reach ~40 total.
    Saves both user and ratings to CSV for persistence.
    """
    global users
    data = request.json
    new_id = int(users["user_id"].max() + 1)
    new_name = data.get("name", f"User_{new_id}")
    user_cats = data.get("categories", [])
    user_brands = data.get("brands", [])
    budget_min = float(data.get("budget_min", 0))
    budget_max = float(data.get("budget_max", 500))

    new_row = pd.DataFrame([{
        "user_id": new_id,
        "name": new_name,
        "age": int(data.get("age", 25)),
        "preferred_categories": ",".join(user_cats),
        "favorite_brands": ",".join(user_brands),
        "budget_min": budget_min,
        "budget_max": budget_max,
    }])
    users = pd.concat([users, new_row], ignore_index=True)
    USER_OPTIONS.append({
        "id": new_id, "name": new_name, "age": int(data.get("age", 25)),
        "categories": user_cats, "brands": user_brands,
        "budget_min": budget_min, "budget_max": budget_max,
    })

    rated_products = data.get("ratings", [])
    if rated_products:
        _save_and_generate_ratings(new_id, user_cats, user_brands, budget_min, budget_max, rated_products)

    users.to_csv(os.path.join(DATA_DIR, "users.csv"), index=False)
    return jsonify({"success": True, "user_id": new_id, "name": new_name})


def _save_and_generate_ratings(user_id, user_cats, user_brands, budget_min, budget_max, manual_ratings):
    """Save user's 5 manual ratings + auto-generate ~35 more using profile matching.
    
    Uses the same rating logic as generate_data.py:
    - Base = 3.0
    - Category match: +0.5 to +1.5
    - Brand match: +0.3 to +1.0
    - Budget fit: +0.0 to +0.5 (penalty if outside: -0.0 to -0.5)
    - Similarity boost: random -0.3 to +0.5
    - Noise: Normal(0, 0.5)
    Final rating clamped to [1.0, 5.0]. Target is ~40 ratings total.
    """
    global ratings
    seed_ids = [r["product_id"] for r in manual_ratings]
    seed_ratings = {r["product_id"]: r["rating"] for r in manual_ratings}

    manual_rows = []
    for r in manual_ratings:
        manual_rows.append({"user_id": user_id, "product_id": r["product_id"], "rating": r["rating"]})
    manual_df = pd.DataFrame(manual_rows)

    rated_ids = set(seed_ids)
    all_ids = products["product_id"].tolist()
    unrated = [pid for pid in all_ids if pid not in rated_ids]
    np.random.seed(user_id)

    target = min(40, len(all_ids))
    needed = target - len(manual_rows)
    if needed <= 0:
        ratings = pd.concat([ratings, manual_df], ignore_index=True)
        return

    chosen = np.random.choice(unrated, min(needed, len(unrated)), replace=False)
    prod_cat = dict(zip(products["product_id"], products["category"]))
    prod_brand = dict(zip(products["product_id"], products["brand"]))
    prod_price = dict(zip(products["product_id"], products["price"]))
    cat_set = set(user_cats)
    brand_set = set(user_brands)
    seed_values = np.array(list(seed_ratings.values()))
    seed_mean = np.mean(seed_values) if len(seed_values) > 0 else 3.0

    new_ratings_rows = []
    for pid in chosen:
        base = 3.0
        cat = prod_cat.get(pid, "")
        if cat in cat_set:
            base += np.random.uniform(0.5, 1.5)
        brand = prod_brand.get(pid, "")
        if brand in brand_set:
            base += np.random.uniform(0.3, 1.0)
        price = prod_price.get(pid, 50)
        if budget_min <= price <= budget_max:
            base += np.random.uniform(0.0, 0.5)
        else:
            base -= np.random.uniform(0.0, 0.5)
        similarity_boost = np.random.uniform(-0.3, 0.5)
        base += similarity_boost
        noise = np.random.normal(0, 0.5)
        rating = round(min(5.0, max(1.0, base + noise)), 1)
        new_ratings_rows.append({"user_id": user_id, "product_id": int(pid), "rating": rating})

    auto_df = pd.DataFrame(new_ratings_rows)
    combined = pd.concat([manual_df, auto_df], ignore_index=True)
    ratings = pd.concat([ratings, combined], ignore_index=True)
    ratings.to_csv(os.path.join(DATA_DIR, "ratings.csv"), index=False)


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
    """Map product category to emoji icon for UI display."""
    icons = {
        "Electronics": "💻", "Clothing": "👕", "Home & Kitchen": "🏠",
        "Books": "📚", "Sports": "⚽", "Beauty": "💄", "Toys": "🧸", "Automotive": "🚗"
    }
    return icons.get(category, "📦")


def stars_html(rating):
    """Generate HTML star rating display (filled ★ and empty ☆) from numeric rating."""
    f = int(rating)
    return "★" * f + "☆" * (5 - f)


if __name__ == "__main__":
    """Start the Flask development server on port 5000."""
    app.run(debug=True, host="0.0.0.0", port=5000)
