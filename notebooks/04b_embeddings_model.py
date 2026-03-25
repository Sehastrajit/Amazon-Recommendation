# 04b_embeddings_model.py
# Phase 4 Part B: NLP with Sentence Embeddings
# Goal: same as TF-IDF but using semantic meaning instead of keyword matching

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# ============================================================
# STEP 1: Load book profiles (reuse cache from Part A)
# ============================================================
print("Loading book profiles from cache...")
book_profiles = pd.read_pickle("../models/book_profiles.pkl")
print(f"Loaded {len(book_profiles):,} books!")

# ============================================================
# STEP 2: Generate embeddings (with caching)
# ============================================================
EMBEDDINGS_CACHE = "../models/embeddings.pkl"

if os.path.exists(EMBEDDINGS_CACHE):
    print("\nLoading cached embeddings...")
    with open(EMBEDDINGS_CACHE, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded! Shape: {embeddings.shape}")

else:
    print("\nGenerating embeddings (first time — will take a few minutes)...")
    print("Downloading model on first run (~90MB, once only)...")

    # Load the model
    # all-MiniLM-L6-v2 is the sweet spot: fast, small, very accurate
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Truncate reviews to 512 chars — model has a token limit
    # and the first 512 chars already capture the most important signal
    texts = book_profiles['clean_reviews'].str[:512].tolist()

    # Generate embeddings in batches (shows a progress bar)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"\nEmbeddings shape: {embeddings.shape}")

    # Save cache
    with open(EMBEDDINGS_CACHE, 'wb') as f:
        pickle.dump(embeddings, f)
    print("Saved to models/embeddings.pkl!")

# ============================================================
# STEP 3: Compute cosine similarity
# ============================================================
print("\nComputing cosine similarity...")
cosine_sim = cosine_similarity(embeddings, embeddings)
print(f"Similarity matrix shape: {cosine_sim.shape}")

# Build title lookup
book_profiles = book_profiles.reset_index(drop=True)
title_to_idx = pd.Series(book_profiles.index, index=book_profiles['title'].str.lower())

# ============================================================
# STEP 4: Recommendation function
# ============================================================
def get_similar_books(book_title, n=5):
    book_title = book_title.lower()

    if book_title not in title_to_idx.index:
        print(f"\nBook '{book_title}' not found in dataset")
        return

    idx = title_to_idx[book_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]

    print(f"\nBooks similar to '{book_profiles.iloc[idx]['title']}':")
    print(f"{'Title':<45} {'Similarity':>10} {'Avg Rating':>10}")
    print("-" * 67)
    for i, score in sim_scores:
        title  = book_profiles.iloc[i]['title'][:44]
        rating = book_profiles.iloc[i]['avg_rating']
        print(f"{title:<45} {score:>10.3f} {rating:>10.2f}")

# ============================================================
# STEP 5: Test with same books as TF-IDF for direct comparison
# ============================================================
get_similar_books("The Prophets (Perennial Classics)")
get_similar_books("Twilight (The Mediator, Book 6)")
get_similar_books("The Fifth Mountain")

# ============================================================
# STEP 6: Save embeddings model for Phase 5
# ============================================================
print("\nSaving embeddings model...")
embeddings_model = {
    'book_profiles': book_profiles,
    'embeddings':    embeddings,
    'title_to_idx':  title_to_idx,
}
with open('../models/embeddings_model.pkl', 'wb') as f:
    pickle.dump(embeddings_model, f)

print("Saved to models/embeddings_model.pkl")
print("\n✅ Phase 4 Part B Complete!")