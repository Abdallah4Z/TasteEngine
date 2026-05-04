import pandas as pd
import os


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_data():
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    return products, users, ratings


def build_user_item_matrix(ratings_df):
    return ratings_df.pivot_table(
        index="user_id", columns="product_id", values="rating"
    )


def get_user_preferences(users_df, user_id):
    user = users_df[users_df["user_id"] == user_id]
    if user.empty:
        return {}
    user = user.iloc[0]
    return {
        "preferred_categories": set(
            user["preferred_categories"].split(",")
        ) if isinstance(user["preferred_categories"], str) else set(),
        "budget_min": user["budget_min"],
        "budget_max": user["budget_max"],
        "favorite_brands": set(
            user["favorite_brands"].split(",")
        ) if isinstance(user["favorite_brands"], str) else set(),
        "name": user["name"],
        "age": user["age"],
    }
