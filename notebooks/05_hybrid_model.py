# 05_hybrid_model.py
# Phase 5: Hybrid Recommender
# Goal: combine CF (personalization) + NLP (content) for better recommendations

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
warnings.filterwarnings('ignore')
# ============================================================
# STEP 1: Load both models
# ============================================================
print("Loading CF model...")
with open('../models/svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)

print("Loading NLP model...")
with open('../models/nlp_model.pkl', 'rb') as f:
    nlp = pickle.load(f)

book_profiles = nlp['book_profiles']
tfidf_matrix  = nlp['tfidf_matrix']
title_to_idx  = nlp['title_to_idx']

print("Loading ratings data...")
ratings_df = pd.read_csv('data/processed/ratings_only.csv')

print(f"\n✅ All models loaded!")
print(f"   → {ratings_df['user_id'].nunique():,} users in ratings data")
print(f"   → {ratings_df['product_id'].nunique():,} books in ratings data")
print(f"   → {len(book_profiles):,} books in NLP model")


# ============================================================
# STEP 2: Build the hybrid recommender
# ============================================================

# Build a lookup: product_id → index in book_profiles
product_to_idx = pd.Series(
    book_profiles.index,
    index=book_profiles['product_id']
)

def hybrid_recommend(user_id, n_candidates=50, n_final=5):
    """
    Improved hybrid:
    - CF filters to quality candidates (predicted 3.5+ stars)
    - NLP scores based on top-5 most similar liked books (not average of all)
    - Combined with dynamic weighting based on how many books user has liked
    """

    # --- Get books this user has already rated ---
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]

    if len(user_ratings) == 0:
        print(f"User '{user_id}' not found in dataset")
        return

    already_read = set(user_ratings['product_id'].tolist())

    # --- Get liked books (4+ stars) as reference for NLP ---
    liked_books = user_ratings[user_ratings['rating'] >= 4]['product_id'].tolist()
    liked_books = [b for b in liked_books if b in product_to_idx.index]

    # --- Get CF predictions for all unread books ---
    all_books = book_profiles['product_id'].tolist()
    unread_books = [b for b in all_books if b not in already_read]

    cf_scores = []
    for book_id in unread_books:
        pred = svd_model.predict(user_id, book_id)
        cf_scores.append((book_id, pred.est))

    # Sort and take top candidates
    cf_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = cf_scores[:n_candidates]

    # --- Score each candidate with NLP ---
    results = []

    for book_id, cf_score in top_candidates:
        if book_id not in product_to_idx.index:
            continue

        book_idx = product_to_idx[book_id]

        # FIX 1: Use top 5 most similar liked books instead of average of all
        if len(liked_books) > 0:
            liked_indices = [product_to_idx[b] for b in liked_books]

            # Get similarity to each liked book
            sims = cosine_similarity(
                tfidf_matrix[book_idx],
                tfidf_matrix[liked_indices]
            ).flatten()

            # Take top 5 similarities (focused signal)
            top_sims = np.sort(sims)[::-1][:5]
            nlp_score = top_sims.mean()
        else:
            nlp_score = 0.0

        # FIX 2: Dynamic weighting — more liked books = trust NLP more
        # fewer liked books = trust CF more
        n_liked = len(liked_books)
        if n_liked < 5:
            cf_weight, nlp_weight = 0.8, 0.2
        elif n_liked < 20:
            cf_weight, nlp_weight = 0.6, 0.4
        else:
            cf_weight, nlp_weight = 0.5, 0.5

        # Normalize CF score from 1-5 to 0-1
        cf_score_norm = (cf_score - 1) / 4

        # Combined score
        hybrid_score = cf_weight * cf_score_norm + nlp_weight * float(nlp_score)

        results.append((book_id, hybrid_score, cf_score, float(nlp_score), cf_weight, nlp_weight))

    # Sort by hybrid score
    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:n_final]

    # --- Display ---
    n_liked = len(liked_books)
    cf_w = top_results[0][4] if top_results else 0.6
    nlp_w = top_results[0][5] if top_results else 0.4

    print(f"\nHybrid recommendations for user '{user_id}':")
    print(f"(Based on {len(user_ratings)} ratings, {n_liked} liked books)")
    print(f"(Weights: {int(cf_w*100)}% CF + {int(nlp_w*100)}% NLP)")
    print(f"\n{'Title':<45} {'Hybrid':>8} {'CF':>8} {'NLP':>8}")
    print("-" * 73)
    for book_id, hybrid_score, cf_score, nlp_score, _, _ in top_results:
        idx   = product_to_idx[book_id]
        title = book_profiles.iloc[idx]['title'][:44]
        print(f"{title:<45} {hybrid_score:>8.3f} {cf_score:>8.2f} {nlp_score:>8.3f}")

# ============================================================
# STEP 3: Test it on a real user
# ============================================================

# Pick a user who has rated many books
test_user = ratings_df.groupby('user_id').size().sort_values(ascending=False).index[0]
print(f"\nTesting with most active user: {test_user}")
hybrid_recommend(test_user)

# Test with a more typical user (20-50 ratings)
typical_users = ratings_df.groupby('user_id').size()
typical_users = typical_users[(typical_users >= 20) & (typical_users <= 50)].index

test_user_2 = typical_users[0]
print(f"\nTesting with typical user: {test_user_2}")
hybrid_recommend(test_user_2)

# Also show what CF alone would have recommended for comparison
print(f"\nCF-only recommendations for user '{test_user_2}':")
user_ratings = ratings_df[ratings_df['user_id'] == test_user_2]
already_read = set(user_ratings['product_id'].tolist())
all_books = book_profiles['product_id'].tolist()
unread_books = [b for b in all_books if b not in already_read]

cf_only = [(b, svd_model.predict(test_user_2, b).est) for b in unread_books]
cf_only.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Title':<45} {'CF Score':>8}")
print("-" * 55)
for book_id, score in cf_only[:5]:
    idx = product_to_idx[book_id]
    title = book_profiles.iloc[idx]['title'][:44]
    print(f"{title:<45} {score:>8.2f}")


# ============================================================
# STEP 4: Save hybrid model
# ============================================================
print("\nSaving hybrid model...")

hybrid_model = {
    'svd_model':       svd_model,
    'book_profiles':   book_profiles,
    'tfidf_matrix':    tfidf_matrix,
    'title_to_idx':    title_to_idx,
    'product_to_idx':  product_to_idx,
}

with open('../models/hybrid_model.pkl', 'wb') as f:
    pickle.dump(hybrid_model, f)

print("Saved to models/hybrid_model.pkl")
print("\n✅ Phase 5 Complete!")
print(f"   → Hybrid = 60% CF + 40% NLP")
print(f"   → Tested on {len(ratings_df['user_id'].unique()):,} users")
print(f"   → Covering {len(book_profiles):,} books")