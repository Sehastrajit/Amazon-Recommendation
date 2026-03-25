# 📚 Amazon Book Recommender

A production-grade hybrid book recommendation system trained on the Amazon Books dataset. Combines collaborative filtering (SVD) with NLP content similarity (TF-IDF + Sentence Embeddings) to deliver personalized recommendations that are both user-relevant and thematically consistent.

> **Hybrid model achieves +85.1% NDCG improvement over collaborative filtering alone.**

---

## How It Works

```
User ID
   │
   ▼
┌─────────────────────────────┐
│  Collaborative Filtering    │  SVD model predicts ratings for all unread books
│  (SVD — Surprise library)   │  → returns top 50 candidates
└─────────────────────────────┘
   │
   ▼
┌─────────────────────────────┐
│  NLP Content Model          │  Scores each candidate by similarity to
│  (TF-IDF / Embeddings)      │  user's top-5 most liked books
└─────────────────────────────┘
   │
   ▼
┌─────────────────────────────┐
│  Hybrid Combiner            │  Dynamic weighting based on user history size
│  CF × weight + NLP × weight │  → returns top 5 final recommendations
└─────────────────────────────┘
```

### Dynamic Weighting

| User History | CF Weight | NLP Weight |
|---|---|---|
| < 5 liked books | 80% | 20% |
| 5–19 liked books | 60% | 40% |
| 20+ liked books | 50% | 50% |

---

## Results

Evaluated across 190 users using a holdout strategy (hide last 3 rated books, check if model finds them).

| Metric | CF Only | Hybrid | Improvement |
|---|---|---|---|
| Precision@5 | 0.0021 | 0.0032 | **+50.0%** |
| Recall@5 | 0.0044 | 0.0061 | **+40.0%** |
| NDCG@5 | 0.0036 | 0.0067 | **+85.1%** |

> Absolute values are low because we recommend 5 books out of 11,839 — the hybrid is 8× better than random chance.

---

## Project Structure

```
amazon-recommender/
│
├── notebooks/
│   ├── 01_eda.py                      ← Exploratory data analysis
│   ├── 02_preprocess.py               ← Data cleaning & filtering
│   ├── 03_collaborative_filtering.py  ← SVD model training
│   ├── 04_nlp_model.py                ← TF-IDF NLP model
│   ├── 04b_embeddings_model.py        ← Sentence embeddings model
│   ├── 05_hybrid_model.py             ← Hybrid recommender
│   └── 06_evaluation.py               ← Metrics & comparison
│
├── src/
│   ├── data_loader.py
│   └── preprocessor.py
│
├── models/                            ← Saved trained models (not tracked in git)
│   ├── svd_model.pkl
│   ├── nlp_model.pkl
│   ├── embeddings_model.pkl
│   └── hybrid_model.pkl
│
├── data/processed/                    ← Processed data (not tracked in git)
│   ├── ratings_only.csv
│   └── train_data.csv
│
└── requirements.txt
```

---

## Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/Sehastrajit/Amazon-Recommendation.git
cd Amazon-Recommendation
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the Amazon Books dataset and place the files in `data/raw/`:
- `reviews_Books.json.gz`
- `meta_Books.json.gz`

### 5. Run the pipeline in order

```bash
cd notebooks
python 01_eda.py
python 02_preprocess.py
python 03_collaborative_filtering.py
python 04_nlp_model.py
python 04b_embeddings_model.py
python 05_hybrid_model.py
python 06_evaluation.py
```

---

## Dataset

- **Source:** Amazon Books Reviews (public dataset)
- **Raw size:** 500,000 reviews
- **After filtering:** 197,074 ratings across 16,203 users and 11,839 books
- **Filter applied:** Users with fewer than 5 ratings removed (reduces sparsity)

| Metric | Before Filter | After Filter |
|---|---|---|
| Users | 214,572 | 16,203 |
| Avg ratings/user | 1.3 | 12.2 |

---

## Models

### Collaborative Filtering (SVD)
- Library: `scikit-surprise`
- 50 latent factors, 20 epochs
- RMSE: **0.91** | MAE: **0.70**

### NLP — TF-IDF
- 10,000 features, bigrams, English stopwords removed
- Matrix: 11,839 × 10,000 (93.2% sparse)
- Best for: series detection, shared vocabulary

### NLP — Sentence Embeddings
- Model: `all-MiniLM-L6-v2`
- 384-dimensional dense vectors
- Best for: cross-author thematic similarity

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, numpy |
| Collaborative filtering | scikit-surprise |
| NLP vectorization | scikit-learn (TF-IDF) |
| Semantic embeddings | sentence-transformers |
| Similarity | scikit-learn (cosine similarity) |
| API | FastAPI |
| Containerization | Docker |
| Deployment | Hugging Face Spaces |

---

