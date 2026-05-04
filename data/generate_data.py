import pandas as pd
import numpy as np
import os

np.random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = {
    "Electronics": ["Smartphones", "Laptops", "Headphones", "Tablets", "Cameras"],
    "Clothing": ["Men's", "Women's", "Kids'", "Accessories", "Footwear"],
    "Home & Kitchen": ["Furniture", "Appliances", "Cookware", "Decor", "Storage"],
    "Books": ["Fiction", "Non-Fiction", "Science", "Technology", "Self-Help"],
    "Sports": ["Fitness", "Outdoor", "Team Sports", "Cycling", "Swimming"],
    "Beauty": ["Skincare", "Makeup", "Haircare", "Fragrance", "Tools"],
    "Toys": ["Educational", "Action Figures", "Board Games", "Dolls", "Puzzles"],
    "Automotive": ["Car Care", "Interior", "Exterior", "Tools", "Electronics"],
}

BRANDS_BY_CATEGORY = {
    "Electronics": ["Samsung", "Apple", "Sony", "LG", "Dell", "HP", "Bose", "Canon"],
    "Clothing": ["Nike", "Adidas", "Zara", "H&M", "Levi's", "Puma", "Under Armour"],
    "Home & Kitchen": ["IKEA", "KitchenAid", "Ninja", "Dyson", "OXO", "Cuisinart"],
    "Books": ["Penguin", "HarperCollins", "Random House", "Simon & Schuster", "Macmillan"],
    "Sports": ["Nike", "Adidas", "Wilson", "Spalding", "The North Face", "Columbia"],
    "Beauty": ["L'Oreal", "Maybelline", "Clinique", "Estee Lauder", "Neutrogena"],
    "Toys": ["LEGO", "Mattel", "Hasbro", "Fisher-Price", "Melissa & Doug"],
    "Automotive": ["Meguiar's", "Armor All", "WeatherTech", "Michelin", "Bosch"],
}

PRODUCT_NAMES = {
    "Electronics": {
        "Smartphones": ["Galaxy S24", "iPhone 15 Pro", "Xperia 1 V", "Pixel 8 Pro", "LG G4"],
        "Laptops": ["ThinkPad X1", "MacBook Pro", "XPS 15", "Surface Laptop", "Spectre x360"],
        "Headphones": ["WH-1000XM5", "AirPods Pro", "QuietComfort 45", "Galaxy Buds", "Momentum 4"],
        "Tablets": ["iPad Pro", "Galaxy Tab S9", "Surface Pro", "Fire HD 10", "Lenovo Tab"],
        "Cameras": ["EOS R5", "Alpha A7 IV", "Z8", "X-T5", "Lumix S5"],
    },
    "Clothing": {
        "Men's": ["Classic Fit Jeans", "Cotton T-Shirt", "Blazer", "Chino Pants", "Hoodie"],
        "Women's": ["Summer Dress", "Yoga Pants", "Leather Jacket", "Silk Blouse", "Maxi Skirt"],
        "Kids'": ["Colorful Leggings", "Graphic Tee", "Denim Jacket", "Plaid Shirt", "Joggers"],
        "Accessories": ["Leather Belt", "Sunglasses", "Wool Scarf", "Baseball Cap", "Watch"],
        "Footwear": ["Running Shoes", "Sandals", "Boots", "Loafers", "Sneakers"],
    },
    "Home & Kitchen": {
        "Furniture": ["Sofa Set", "Coffee Table", "Bookshelf", "Desk Chair", "Bed Frame"],
        "Appliances": ["Air Fryer", "Blender", "Coffee Maker", "Microwave", "Toaster"],
        "Cookware": ["Non-Stick Pan", "Chef's Knife", "Cutting Board", "Pot Set", "Baking Sheet"],
        "Decor": ["Table Lamp", "Wall Art", "Vase", "Throw Pillow", "Candle Set"],
        "Storage": ["Plastic Bins", "Shelf Organizer", "Shoe Rack", "Closet System", "Drawer Divider"],
    },
    "Books": {
        "Fiction": ["The Silent Echo", "Midnight Sun", "Ocean's Memory", "The Last Garden", "Crimson Peak"],
        "Non-Fiction": ["Atomic Habits", "Sapiens", "Educated", "The Power of Now", "Outliers"],
        "Science": ["A Brief History of Time", "The Gene", "Cosmos", "The Selfish Gene", "Six Easy Pieces"],
        "Technology": ["Clean Code", "Design Patterns", "Introduction to Algorithms", "Structure and Interpretation", "The Pragmatic Programmer"],
        "Self-Help": ["The 7 Habits", "How to Win Friends", "Think and Grow Rich", "The Subtle Art", "Daring Greatly"],
    },
    "Sports": {
        "Fitness": ["Adjustable Dumbbells", "Yoga Mat", "Resistance Bands", "Jump Rope", "Foam Roller"],
        "Outdoor": ["Camping Tent", "Hiking Backpack", "Sleeping Bag", "Portable Stove", "Water Filter"],
        "Team Sports": ["Soccer Ball", "Basketball", "Volleyball", "Baseball Glove", "Football"],
        "Cycling": ["Mountain Bike", "Helmet", "Bike Pump", "Cycling Jersey", "Bike Lock"],
        "Swimming": ["Swim Goggles", "Kickboard", "Swim Cap", "Ear Plugs", "Waterproof Bag"],
    },
    "Beauty": {
        "Skincare": ["Moisturizer", "Face Serum", "Sunscreen", "Eye Cream", "Face Mask"],
        "Makeup": ["Foundation", "Lipstick", "Mascara", "Eyeshadow Palette", "Concealer"],
        "Haircare": ["Shampoo", "Conditioner", "Hair Oil", "Hair Dryer", "Straightener"],
        "Fragrance": ["Eau de Parfum", "Cologne", "Body Spray", "Perfume Oil", "Rollerball"],
        "Tools": ["Makeup Brushes", "Sponge Set", "Tweezers", "Mirror", "Travel Case"],
    },
    "Toys": {
        "Educational": ["Science Kit", "Building Blocks", "Math Puzzle", "Robot Kit", "Microscope Set"],
        "Action Figures": ["Superhero Figure", "Animal Set", "Dinosaur Set", "Space Explorer", "Fantasy Warrior"],
        "Board Games": ["Strategy Game", "Family Game", "Card Game", "Trivia Game", "Cooperative Game"],
        "Dolls": ["Fashion Doll", "Baby Doll", "Dollhouse", "Doll Clothes Set", "Puppet Set"],
        "Puzzles": ["1000 Piece Puzzle", "Floor Puzzle", "3D Puzzle", "Wooden Puzzle", "Brain Teaser"],
    },
    "Automotive": {
        "Car Care": ["Car Shampoo", "Wax Kit", "Microfiber Cloth", "Tire Cleaner", "Interior Wipes"],
        "Interior": ["Seat Covers", "Floor Mats", "Steering Wheel Cover", "Air Freshener", "Phone Mount"],
        "Exterior": ["Car Cover", "Mud Flaps", "Window Visors", "License Plate Frame", "Side Moldings"],
        "Tools": ["Jump Starter", "Tire Inflator", "Tool Kit", "Car Jack", "Emergency Kit"],
        "Electronics": ["Dash Cam", "GPS Navigator", "Bluetooth Adapter", "Backup Camera", "Car Charger"],
    },
}


def generate_products(n_per_category=8):
    products = []
    pid = 1
    for category, subcategories in CATEGORIES.items():
        brands = BRANDS_BY_CATEGORY[category]
        names = PRODUCT_NAMES[category]
        for subcategory in subcategories:
            subcat_names = names.get(subcategory, ["Generic Item"])
            for i in range(min(n_per_category, len(subcat_names))):
                name = subcat_names[i]
                brand = np.random.choice(brands)
                price = round(np.random.uniform(9.99, 1499.99), 2)
                avg_rating = round(np.random.uniform(3.0, 5.0), 1)
                num_reviews = np.random.randint(10, 500)
                products.append({
                    "product_id": pid,
                    "name": name,
                    "category": category,
                    "subcategory": subcategory,
                    "brand": brand,
                    "price": price,
                    "avg_rating": avg_rating,
                    "num_reviews": num_reviews,
                })
                pid += 1
    return pd.DataFrame(products)


def generate_users(n_users=200):
    first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace",
                   "Henry", "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia",
                   "Paul", "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy",
                   "Xander", "Yara", "Zack", "Ava", "Ben", "Clara", "David"]
    users = []
    for uid in range(1, n_users + 1):
        name = np.random.choice(first_names) + f"_{uid}"
        age = np.random.randint(18, 65)
        n_cats = np.random.randint(1, 4)
        preferred_categories = list(np.random.choice(list(CATEGORIES.keys()), n_cats, replace=False))
        budget_min = round(np.random.uniform(10, 200), 2)
        budget_max = round(budget_min + np.random.uniform(50, 1500), 2)
        n_brands = np.random.randint(0, 3)
        all_brands = list({b for brands in BRANDS_BY_CATEGORY.values() for b in brands})
        favorite_brands = list(np.random.choice(all_brands, n_brands, replace=False)) if n_brands > 0 else []
        users.append({
            "user_id": uid,
            "name": name,
            "age": age,
            "preferred_categories": ",".join(preferred_categories),
            "budget_min": budget_min,
            "budget_max": budget_max,
            "favorite_brands": ",".join(favorite_brands),
        })
    return pd.DataFrame(users)


def generate_ratings(products_df, users_df, n_ratings=5000, sparsity_factor=0.05):
    ratings = []
    n_users = len(users_df)
    n_products = len(products_df)

    user_cat_prefs = {}
    for _, u in users_df.iterrows():
        user_cat_prefs[u["user_id"]] = set(u["preferred_categories"].split(",")) if u["preferred_categories"] else set()

    user_brand_prefs = {}
    for _, u in users_df.iterrows():
        user_brand_prefs[u["user_id"]] = set(u["favorite_brands"].split(",")) if u["favorite_brands"] else set()

    possible_pairs = []
    for uid in range(1, n_users + 1):
        for pid in range(1, n_products + 1):
            possible_pairs.append((uid, pid))

    selected_indices = np.random.choice(len(possible_pairs), min(n_ratings, len(possible_pairs)), replace=False)
    selected_pairs = [possible_pairs[i] for i in selected_indices]

    prod_cat = dict(zip(products_df["product_id"], products_df["category"]))
    prod_brand = dict(zip(products_df["product_id"], products_df["brand"]))
    prod_price = dict(zip(products_df["product_id"], products_df["price"]))

    for uid, pid in selected_pairs:
        base = 3.0

        cat = prod_cat.get(pid, "")
        if uid in user_cat_prefs and cat in user_cat_prefs[uid]:
            base += np.random.uniform(0.5, 1.5)

        brand = prod_brand.get(pid, "")
        if uid in user_brand_prefs and brand in user_brand_prefs[uid]:
            base += np.random.uniform(0.3, 1.0)

        price = prod_price.get(pid, 50)
        user_row = users_df[users_df["user_id"] == uid].iloc[0]
        if user_row["budget_min"] <= price <= user_row["budget_max"]:
            base += np.random.uniform(0.0, 0.5)
        else:
            base -= np.random.uniform(0.0, 0.5)

        noise = np.random.normal(0, 0.5)
        rating = round(min(5.0, max(1.0, base + noise)), 1)

        ratings.append({
            "user_id": uid,
            "product_id": pid,
            "rating": rating,
        })

    return pd.DataFrame(ratings)


def generate_interactions(products_df, users_df, rating_df, n_purchases=2000):
    interactions = []
    selected = rating_df.sample(min(n_purchases, len(rating_df)))
    for _, row in selected.iterrows():
        interactions.append({
            "user_id": row["user_id"],
            "product_id": row["product_id"],
            "purchased": True,
            "quantity": np.random.randint(1, 4),
        })
    return pd.DataFrame(interactions)


def main():
    print("Generating products...")
    products = generate_products(n_per_category=8)
    products.to_csv(os.path.join(DATA_DIR, "products.csv"), index=False)
    print(f"  {len(products)} products generated")

    print("Generating users...")
    users = generate_users(n_users=200)
    users.to_csv(os.path.join(DATA_DIR, "users.csv"), index=False)
    print(f"  {len(users)} users generated")

    print("Generating ratings...")
    ratings = generate_ratings(products, users, n_ratings=8000)
    ratings.to_csv(os.path.join(DATA_DIR, "ratings.csv"), index=False)
    print(f"  {len(ratings)} ratings generated")

    print("Generating interactions...")
    interactions = generate_interactions(products, users, ratings)
    interactions.to_csv(os.path.join(DATA_DIR, "interactions.csv"), index=False)
    print(f"  {len(interactions)} interactions generated")

    print("All data generated successfully!")


if __name__ == "__main__":
    main()
