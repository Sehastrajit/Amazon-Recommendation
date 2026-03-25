# 04_nlp_model.py
# Phase 4: NLP Content-Based Filtering
# Goal: recommend books based on what review text says, not just ratings

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
import os

# ============================================================
# STEP 1: Load data
# ============================================================
PROFILES_CACHE = "../models/book_profiles.pkl"
TFIDF_CACHE    = "../models/tfidf_matrix.pkl"
VECTORIZER_CACHE = "../models/tfidf_vectorizer.pkl"

if os.path.exists(PROFILES_CACHE) and os.path.exists(TFIDF_CACHE):
    # --- Fast path: skip CSV loading entirely ---
    print("Loading from cache (skipping CSV)...")
    book_profiles = pd.read_pickle(PROFILES_CACHE)
    tfidf_matrix  = pickle.load(open(TFIDF_CACHE, 'rb'))
    tfidf         = pickle.load(open(VECTORIZER_CACHE, 'rb'))
    print(f"Loaded {len(book_profiles):,} books from cache!")

else:
    # --- Slow path: runs only on first run ---
    print("Loading train_data.csv... (this may take 10-20 seconds)")
    df = pd.read_csv("data/processed/train_data.csv")
    print(f"Loaded {len(df):,} rows")

    # ============================================================
    # STEP 2: Aggregate all reviews per book into one text blob
    # ============================================================
    print("\nAggregating reviews per book...")
    df['combined_text'] = df['review_text'].fillna('') + ' ' + df['review_summary'].fillna('')

    book_profiles = df.groupby('product_id').agg(
        title       = ('title', 'first'),
        categories  = ('categories', 'first'),
        all_reviews = ('combined_text', ' '.join),
        avg_rating  = ('rating', 'mean'),
        num_ratings = ('rating', 'count')
    ).reset_index()

    print(f"Books after aggregation: {len(book_profiles):,}")

    # ============================================================
    # STEP 3: Clean the text
    # ============================================================
    print("\nCleaning review text...")

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    book_profiles['clean_reviews'] = book_profiles['all_reviews'].apply(clean_text)

    # ============================================================
    # STEP 4: Build TF-IDF matrix
    # ============================================================
    print("\nBuilding TF-IDF matrix...")

    tfidf = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        min_df=2,
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(book_profiles['clean_reviews'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # ============================================================
    # SAVE CACHE (so next run skips all the above)
    # ============================================================
    print("\nSaving cache for future runs...")
    book_profiles.to_pickle(PROFILES_CACHE)
    pickle.dump(tfidf_matrix, open(TFIDF_CACHE, 'wb'))
    pickle.dump(tfidf, open(VECTORIZER_CACHE, 'wb'))
    print("Saved to models/ folder!")

# ============================================================
# STEP 5: Compute cosine similarity (always recomputed, ~seconds)
# ============================================================
print("\nComputing cosine similarity matrix...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Similarity matrix shape: {cosine_sim.shape}")

# Build a lookup: book title → index
book_profiles = book_profiles.reset_index(drop=True)
title_to_idx = pd.Series(book_profiles.index, index=book_profiles['title'].str.lower())

# ============================================================
# STEP 6: Recommendation function
# ============================================================
def get_similar_books(book_title, n=5):
    book_title = book_title.lower()

    if book_title not in title_to_idx.index:
        print(f"\nBook '{book_title}' not found in dataset")
        return

    idx = title_to_idx[book_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]  # skip index 0 (the book itself)

    print(f"\nBooks similar to '{book_profiles.iloc[idx]['title']}':")
    print(f"{'Title':<45} {'Similarity':>10} {'Avg Rating':>10}")
    print("-" * 67)
    for i, score in sim_scores:
        title  = book_profiles.iloc[i]['title'][:44]
        rating = book_profiles.iloc[i]['avg_rating']
        print(f"{title:<45} {score:>10.3f} {rating:>10.2f}")
    
# DEBUG: search for partial title matches
def find_title(search_term):
    search_term = search_term.lower()
    matches = book_profiles[book_profiles['title'].str.lower().str.contains(search_term, na=False)]
    print(f"\nMatches for '{search_term}':")
    print(matches['title'].head(10).tolist())

find_title("da vinci")
find_title("twilight")
find_title("harry potter")


# ============================================================
# STEP 7: Save the NLP model for Phase 5 (Hybrid)
# ============================================================
print("\nSaving NLP model...")

nlp_model = {
    'book_profiles': book_profiles,   # book metadata + clean reviews
    'tfidf_matrix':  tfidf_matrix,    # the 11,839 × 10,000 matrix
    'tfidf':         tfidf,           # the fitted vectorizer
    'title_to_idx':  title_to_idx,    # title → index lookup
}

with open('../models/nlp_model.pkl', 'wb') as f:
    pickle.dump(nlp_model, f)

print("Saved to models/nlp_model.pkl")
print("\n✅ Phase 4 Complete!")
print(f"   → {len(book_profiles):,} books vectorized")
print(f"   → TF-IDF matrix: {tfidf_matrix.shape}")
print(f"   → Cosine similarity: {cosine_sim.shape}")
print(f"   → Model saved to models/nlp_model.pkl")



# ============================================================
# TEST IT
# ============================================================
get_similar_books("The Prophets (Perennial Classics)")
get_similar_books("Twilight (The Mediator, Book 6)")
get_similar_books("The Fifth Mountain")