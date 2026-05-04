# Synthetic Dataset Plan

## Purpose

Generate a realistic e-commerce dataset to train and evaluate all recommendation approaches. Synthetic data gives full control over data characteristics (sparsity, user segments, product diversity).

## Dataset Components

### 1. Products (`products.csv`) — ~500 items

| Field | Type | Example |
|-------|------|---------|
| product_id | int | 1 |
| name | str | "Samsung Galaxy S24" |
| category | str | "Electronics" |
| subcategory | str | "Smartphones" |
| brand | str | "Samsung" |
| price | float | 799.99 |
| description | str | "Latest Samsung flagship with AI features" |
| features | str | "6.8in, 256GB, 12GB RAM" |
| avg_rating | float | 4.5 |
| num_reviews | int | 234 |

**Categories**: Electronics, Clothing, Home & Kitchen, Books, Sports, Beauty, Toys, Automotive

### 2. Users (`users.csv`) — ~200 users

| Field | Type | Example |
|-------|------|---------|
| user_id | int | 1 |
| name | str | "Alice" |
| age | int | 28 |
| gender | str | "Female" |
| preferred_categories | str | "Electronics,Books" |
| budget_min | float | 100.0 |
| budget_max | float | 1500.0 |
| favorite_brands | str | "Samsung,Sony" |

**User segments**: budget-conscious, brand-loyal, category-specific, high-spenders

### 3. Ratings (`ratings.csv`) — ~5000 entries

| Field | Type | Example |
|-------|------|---------|
| user_id | int | 1 |
| product_id | int | 42 |
| rating | float | 4.0 |
| timestamp | int | 1700000000 |

**Sparsity**: ~5% density (controlled to allow meaningful CF comparison)

### 4. Interactions / Purchases (`interactions.csv`)

| Field | Type | Example |
|-------|------|---------|
| user_id | int | 1 |
| product_id | int | 42 |
| purchased | bool | True |
| quantity | int | 1 |

## Generation Logic

1. Create products with realistic brand-category mappings
2. Create users with varied preference profiles
3. Generate ratings based on:
   - User's preferred categories → higher ratings
   - User's favorite brands → higher ratings
   - Price affinity (if within budget) → higher ratings
   - Random noise (10-15%) for realism
4. Create train/test split (80/20) for evaluation

## Cold-Start Simulation

- 10% of users have < 3 ratings (cold-start for CF)
- 10% of products have < 2 ratings (new items)
- This allows meaningful comparison of approaches under different conditions
