# TasteEngine

A multi-approach recommender system for e-commerce platforms that generates personalized product recommendations using collaborative filtering, content-based filtering, and knowledge-based techniques, with comprehensive evaluation and comparison.

## Features

- **Collaborative Filtering** — 5 methods: User-Based, Item-Based, SVD (Matrix Factorization), KNN-Based, and Slope One
- **Content-Based Filtering** — TF-IDF similarity and feature matching based on user preferences
- **Knowledge-Based Filtering** — Constraint-based, rule-based, and utility-based recommendation strategies
- **Evaluation Suite** — RMSE, MAE, Precision@k, Recall@k, F1@k, and Coverage metrics
- **Explanation Engine** — Human-readable explanations for every recommendation
- **Interactive Web UI** — Built with Flask, allowing users to select approaches, compare results, and explore products
- **Synthetic Dataset** — Generates realistic e-commerce data (products, users, ratings)

## Project Structure

```
TasteEngine/
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
├── data/
│   ├── generate_data.py        # Synthetic dataset generator
│   ├── products.csv            # Product catalog
│   ├── users.csv               # User profiles
│   └── ratings.csv             # User-product ratings
├── recommender/
│   ├── collaborative.py        # Collaborative filtering implementations
│   ├── content_based.py        # Content-based recommendation engine
│   ├── knowledge_based.py      # Knowledge-based recommendation engine
│   ├── evaluation.py           # Evaluation metrics and comparison
│   └── explainer.py            # Recommendation explanation engine
├── utils/
│   ├── helpers.py              # Data loading and preprocessing utilities
│   └── similarity.py           # Similarity metrics (cosine, pearson, adjusted cosine, jaccard)
├── templates/
│   ├── index.html              # Landing page
│   ├── recommend.html          # Recommendation interface
│   └── evaluation.html         # Evaluation dashboard
├── static/
│   └── css/
│       └── style.css           # Application styles
└── plan/                       # Design documents
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

## Installation

```bash
git clone https://github.com/Abdallah4Z/TasteEngine.git
cd TasteEngine
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

### Generating Fresh Data

```bash
python data/generate_data.py
```

## Recommendation Approaches

| Approach | Methods | Description |
|---|---|---|
| Collaborative Filtering | User-Based, Item-Based, SVD, KNN, Slope One | Leverages user-item interaction patterns |
| Content-Based | TF-IDF, Feature Matching | Uses product attributes and user preferences |
| Knowledge-Based | Constraint, Rule, Utility | Applies domain rules and constraints |

## Dependencies

- Flask 3.0
- pandas 2.1
- numpy 1.26
- scikit-learn 1.3

## License

MIT
