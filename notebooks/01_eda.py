# notebooks/01_eda.py
# ─────────────────────────────────────────────────────────────
# Phase 1 — Exploratory Data Analysis
# Run this file to explore and understand the raw data
# ─────────────────────────────────────────────────────────────

import sys
import os

# This line lets us import from the src/ folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader   import download_data, load_reviews, load_metadata
from src.preprocessor  import clean_reviews, clean_metadata, combine, save_processed

# ── Step 1: Download data if not already downloaded ───────────
download_data()

# ── Step 2: Load raw data ─────────────────────────────────────
raw_reviews  = load_reviews(n_rows=50000)
raw_metadata = load_metadata(n_rows=50000)

# ── Step 3: Clean both dataframes ────────────────────────────
clean_rev  = clean_reviews(raw_reviews)
clean_meta = clean_metadata(raw_metadata)

# ── Step 4: Combine into one dataframe ───────────────────────
combined_df = combine(clean_rev, clean_meta)

# ── Step 5: Explore ───────────────────────────────────────────
print("\n── Rating Distribution ──────────────────────")
print(combined_df['rating'].value_counts().sort_index())

five_star_pct = (combined_df['rating'] == 5.0).sum() / len(combined_df) * 100
print(f"\n{five_star_pct:.1f}% of reviews are 5 stars")

print("\n── Reviews Per User ─────────────────────────")
reviews_per_user = combined_df.groupby('user_id')['product_id'].count()
print(f"  Average : {reviews_per_user.mean():.1f}")
print(f"  Minimum : {reviews_per_user.min()}")
print(f"  Maximum : {reviews_per_user.max()}")

print("\n── Reviews Per Book ─────────────────────────")
reviews_per_book = combined_df.groupby('product_id')['user_id'].count()
print(f"  Average : {reviews_per_book.mean():.1f}")
print(f"  Minimum : {reviews_per_book.min()}")
print(f"  Maximum : {reviews_per_book.max()}")

print("\n── Top 5 Most Reviewed Books ────────────────")
top_books = (
    combined_df.groupby(['product_id', 'title'])['rating']
    .count()
    .reset_index()
    .rename(columns={'rating': 'review_count'})
    .sort_values('review_count', ascending=False)
    .head(5)
)
print(top_books.to_string(index=False))

# ── Step 6: Save cleaned data ─────────────────────────────────
save_processed(combined_df, "sample_clean.csv")