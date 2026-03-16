#Step 1 → Load 500,000 rows of reviews
#Step 2 → Load 100,000 rows of metadata
#Step 3 → Clean and combine them
#Step 4 → Filter out users with too few ratings
#Step 5 → Save the final dataset ready for model training

# notebooks/02_preprocess.py
# ─────────────────────────────────────────────────────────────
# Phase 2 — Preprocessing for Model Training
# Loads a larger sample and prepares it for collaborative filtering
# ─────────────────────────────────────────────────────────────

import sys
import os
import pandas as pd

# This lets us import from the src/ folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader  import load_reviews, load_metadata
from src.preprocessor import clean_reviews, clean_metadata, combine, save_processed

# ── Step 1: Load larger sample ────────────────────────────────
# We load 500k reviews and 100k metadata rows this time
# This will take a minute — completely normal
print("=" * 50)
print("PHASE 2 — PREPROCESSING FOR MODEL TRAINING")
print("=" * 50)

raw_reviews  = load_reviews(n_rows=500000)
raw_metadata = load_metadata(n_rows=100000)

# ── Step 2: Clean both ────────────────────────────────────────
clean_rev  = clean_reviews(raw_reviews)
clean_meta = clean_metadata(raw_metadata)

# ── Step 3: Combine ───────────────────────────────────────────
combined_df = combine(clean_rev, clean_meta)

print(f"\nCombined shape: {combined_df.shape}")
print(f"Columns: {combined_df.columns.tolist()}")

# ── Step 4: Filter users with too few ratings ─────────────────
# For collaborative filtering to work, users need enough history
# We keep only users who have rated at least 5 books

print("\n── Before filtering ─────────────────────────")
print(f"  Total rows        : {len(combined_df)}")
print(f"  Unique users      : {combined_df['user_id'].nunique()}")
print(f"  Unique books      : {combined_df['product_id'].nunique()}")

# Count how many books each user has rated
user_counts = combined_df.groupby('user_id')['product_id'].count()

# Keep only users who rated 5 or more books
users_to_keep = user_counts[user_counts >= 5].index

# Filter the dataframe to only those users
filtered_df = combined_df[combined_df['user_id'].isin(users_to_keep)]

print("\n── After filtering (min 5 ratings per user) ─")
print(f"  Total rows        : {len(filtered_df)}")
print(f"  Unique users      : {filtered_df['user_id'].nunique()}")
print(f"  Unique books      : {filtered_df['product_id'].nunique()}")

# Check the new average reviews per user
reviews_per_user = filtered_df.groupby('user_id')['product_id'].count()
print(f"\n  Avg ratings/user  : {reviews_per_user.mean():.1f}")
print(f"  Min ratings/user  : {reviews_per_user.min()}")
print(f"  Max ratings/user  : {reviews_per_user.max()}")

# ── Step 5: Save the final processed dataset ──────────────────
# This is the dataset we'll use for ALL model training
# We save two versions:
# 1. Full data with all columns  → for NLP model
# 2. Ratings only                → for collaborative filtering model

# Save full version
save_processed(filtered_df, "train_data.csv")

# Save ratings only version — just the three columns CF needs
ratings_df = filtered_df[['user_id', 'product_id', 'rating']]
save_processed(ratings_df, "ratings_only.csv")

print("\n── Final Dataset Summary ────────────────────")
print(f"  Rows              : {len(filtered_df)}")
print(f"  Unique users      : {filtered_df['user_id'].nunique()}")
print(f"  Unique books      : {filtered_df['product_id'].nunique()}")
print(f"  Avg ratings/user  : {filtered_df.groupby('user_id')['product_id'].count().mean():.1f}")
print(f"  Rating avg        : {filtered_df['rating'].mean():.2f}")
print(f"\n✅ Preprocessing complete — ready for model training!")