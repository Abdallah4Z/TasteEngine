# Project Overview

## Intelligent E-Commerce Recommendation System

A multi-approach recommender system for an e-commerce platform that generates personalized product recommendations using three different recommendation approaches, with comprehensive evaluation and comparison.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python + Flask |
| Frontend | HTML / CSS / JS (glassmorphism + minimalist design) |
| Data | Synthetic dataset (NumPy + Pandas) |
| ML/Similarity | Scikit-learn, NumPy, Pandas |
| Visualization | Chart.js (in-browser charts) |

## Key Requirements

1. **Three approaches**: Collaborative Filtering, Content-Based, Knowledge-Based
2. **4+ CF methods**: User-Based, Item-Based, SVD, KNN
3. **4+ evaluation techniques**: RMSE, MAE, Precision@K, Recall@K, F1, Coverage
4. **Explainable recommendations**: Human-readable reasons for each suggestion
5. **Comparison & analysis**: Which method/approach performs best and why

## System Flow

```
User Input → Select Approach → Recommendation Engine → 
    → Evaluation Module → Comparisons → UI Display
```

## Architecture

```
recommender-project/
├── app.py                    # Flask entry point (routes)
├── requirements.txt          # Dependencies
├── data/
│   ├── generate_data.py      # Synthetic dataset generator
│   ├── products.csv          # Product catalog (500 items)
│   ├── users.csv             # User profiles (200 users)
│   ├── ratings.csv           # User-item ratings (5000+)
│   └── interactions.csv      # Purchase history
├── recommender/
│   ├── __init__.py
│   ├── collaborative.py      # 4+ CF methods
│   ├── content_based.py      # Content-based methods
│   ├── knowledge_based.py    # Knowledge-based methods
│   ├── evaluation.py         # 4+ evaluation techniques
│   └── explainer.py          # Explanation generator
├── utils/
│   ├── similarity.py         # Similarity metrics (cosine, pearson, etc.)
│   └── helpers.py            # Common utilities
├── static/
│   └── css/
│       └── style.css         # Professional UI styles
├── templates/
│   ├── index.html            # Home page
│   ├── recommend.html        # Recommendation results
│   └── evaluation.html       # Evaluation & comparison dashboard
└── plan/
    ├── 01-overview.md
    ├── 02-dataset.md
    ├── 03-collaborative-filtering.md
    ├── 04-content-based.md
    ├── 05-knowledge-based.md
    ├── 06-evaluation.md
    ├── 07-explanation.md
    ├── 08-ui-ux.md
    └── 09-implementation-phases.md
```
