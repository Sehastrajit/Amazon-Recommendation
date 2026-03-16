# ─────────────────────────────────────────────
# PHASE 1 — EDA (Exploratory Data Analysis)
# ─────────────────────────────────────────────

# STEP 1: IMPORTS
# We load all the tools we need at the top
# Think of this like opening apps before using them

import pandas as pd        # loading and manipulating data
import numpy as np         # math operations
import matplotlib.pyplot as plt  # drawing charts
import gzip                # reading compressed .gz files
import json                # reading JSON format
import os                  # working with file paths
import urllib.request      # downloading files from the internet
import ast
print("✅ All imports successful!")


# ─────────────────────────────────────────────
# STEP 2: DOWNLOAD THE DATA
# ─────────────────────────────────────────────

# Create the data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Reviews file   →  WHO did WHAT (users + ratings + review text)
# Metadata file  →  WHAT is WHAT (book titles + genres + price)

# asin           →  the key that connects both
# Official download links from UCSD / Stanford
REVIEWS_URL  = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz"
META_URL     = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz"

REVIEWS_PATH = "data/reviews_books.json.gz"
META_PATH    = "data/meta_books.json.gz"

def download(url, path):
    """Download a file only if it doesn't already exist."""
    if os.path.exists(path):
        print(f"✅ Already exists, skipping: {path}")
        return
    print(f"⬇️  Downloading {path} ...")
    urllib.request.urlretrieve(url, path)
    print(f"✅ Saved to {path}")

download(REVIEWS_URL, REVIEWS_PATH)
download(META_URL, META_PATH)

print("\n✅ Both files ready!")

# ─────────────────────────────────────────────
# STEP 3: LOAD A SAMPLE OF THE REVIEWS DATA
# ─────────────────────────────────────────────

# We only load the first 50,000 rows to keep things fast
# The file is compressed (.gz) so we use gzip to open it

print("Loading reviews sample...")

reviews = []  # empty list — we'll fill it row by row

with gzip.open(REVIEWS_PATH, "rb") as f:
    for i, line in enumerate(f):
        if i >= 50000:          # stop after 50,000 rows
            break
        reviews.append(json.loads(line))   # each line is one review in JSON format

# Convert the list into a pandas DataFrame (like an Excel table in Python)
reviews_df = pd.DataFrame(reviews)

print(f"✅ Loaded {len(reviews_df)} reviews")
print("\nFirst 5 rows:")
print(reviews_df.head())

# ─────────────────────────────────────────────
# STEP 4: UNDERSTAND WHAT WE'RE WORKING WITH
# ─────────────────────────────────────────────

# This shows ALL column names
print("Column names:")
print(reviews_df.columns.tolist())

# This shows the shape — how many rows and columns
print(f"\nShape: {reviews_df.shape[0]} rows, {reviews_df.shape[1]} columns")

# This shows each column and what TYPE of data it holds
# object = text, float64 = decimal number, int64 = whole number
print("\nColumn types:")
print(reviews_df.dtypes)

# This shows how many values are MISSING in each column
# Missing data is a very common problem in real datasets
print("\nMissing values per column:")
print(reviews_df.isnull().sum())

# ─────────────────────────────────────────────
# STEP 5: CLEAN THE DATA
# ─────────────────────────────────────────────

# Keep only the columns we actually need
# We are throwing away: reviewerName, helpful, reviewTime
reviews_df = reviews_df[['reviewerID', 'asin', 'overall', 'reviewText', 'summary', 'unixReviewTime']]

print("✅ Kept only useful columns")
print(reviews_df.columns.tolist())

# Rename columns to cleaner names
# This makes the code easier to read and understand
reviews_df = reviews_df.rename(columns={
    'reviewerID'     : 'user_id',
    'asin'           : 'product_id',
    'overall'        : 'rating',
    'reviewText'     : 'review_text',
    'summary'        : 'review_summary',
    'unixReviewTime' : 'timestamp'
})

print("\n✅ Renamed columns to:")
print(reviews_df.columns.tolist())

# Show the first 3 rows now that it's clean
print("\nCleaned data sample:")
print(reviews_df.head(3))

# ─────────────────────────────────────────────
# STEP 6: LOAD AND CLEAN THE METADATA
# ─────────────────────────────────────────────

print("Loading metadata sample...")

metadata = []

with gzip.open(META_PATH, "rb") as f:
    for i, line in enumerate(f):
        if i >= 50000:
            break
        try:
            # decode bytes to string first, then parse as Python dict
            line_str = line.decode("utf-8")
            metadata.append(ast.literal_eval(line_str))
        except Exception:
            continue   # skip any line that still fails

metadata_df = pd.DataFrame(metadata)

print(f"✅ Loaded {len(metadata_df)} metadata rows")
print("\nAll columns in metadata:")
print(metadata_df.columns.tolist())

print("\nMissing values:")
print(metadata_df.isnull().sum())

# ─────────────────────────────────────────────
# STEP 7: CLEAN THE METADATA
# ─────────────────────────────────────────────

# Keep only useful columns
metadata_df = metadata_df[['asin', 'title', 'categories', 'price']]

print("✅ Kept only useful columns")

# Rename asin to product_id to match our reviews dataframe
metadata_df = metadata_df.rename(columns={
    'asin' : 'product_id'
})

# The categories column looks like this:
# [["Books", "Literature & Fiction", "Classics"]]
# It's a list inside a list — we just want the innermost values as a flat list
# Let's look at a sample first
print("\nSample of categories column (raw):")
print(metadata_df['categories'].head(3))

# Flatten categories — take the first list, join items as a string
# Example: [["Books", "Fiction"]] → "Books, Fiction"
def flatten_categories(cat):
    try:
        # cat is a list of lists — we take the first list and join it
        return ", ".join(cat[0])
    except:
        return ""   # return empty string if anything goes wrong

metadata_df['categories'] = metadata_df['categories'].apply(flatten_categories)

# Handle missing titles — drop rows where title is missing
# only 23 rows so we lose almost nothing
metadata_df = metadata_df.dropna(subset=['title'])

# Handle missing price — fill with 0.0 for now
metadata_df['price'] = metadata_df['price'].fillna(0.0)

# Convert price to a number — it's stored as text like "12.99"
# pd.to_numeric with errors='coerce' turns anything that's not a number into NaN
metadata_df['price'] = pd.to_numeric(metadata_df['price'], errors='coerce').fillna(0.0)

print("\n✅ Cleaned metadata sample:")
print(metadata_df.head(3))

print("\nMissing values after cleaning:")
print(metadata_df.isnull().sum())


# ─────────────────────────────────────────────
# STEP 8: JOIN REVIEWS AND METADATA
# ─────────────────────────────────────────────

# We merge both dataframes on 'product_id'
# This is exactly like a JOIN in SQL or VLOOKUP in Excel
# how='inner' means only keep rows where product_id exists in BOTH files

combined_df = pd.merge(
    reviews_df,
    metadata_df,
    on='product_id',
    how='inner'
)

print(f"✅ Reviews rows:  {len(reviews_df)}")
print(f"✅ Metadata rows: {len(metadata_df)}")
print(f"✅ Combined rows: {len(combined_df)}")

print("\nCombined dataframe columns:")
print(combined_df.columns.tolist())

print("\nSample of combined data:")
print(combined_df[['user_id', 'product_id', 'rating', 'title', 'categories']].head(5))

# ─────────────────────────────────────────────
# STEP 9: EXPLORE THE DATA
# ─────────────────────────────────────────────

# ── Rating Distribution ───────────────────────
# How many 1 star, 2 star, 3 star... reviews do we have?
print("Rating distribution:")
print(combined_df['rating'].value_counts().sort_index())

# What percentage of reviews are 5 stars?
five_star_pct = (combined_df['rating'] == 5.0).sum() / len(combined_df) * 100
print(f"\n{five_star_pct:.1f}% of reviews are 5 stars")

# ── Reviews Per User ──────────────────────────
# How many books has each user rated?
reviews_per_user = combined_df.groupby('user_id')['product_id'].count()
print(f"\nReviews per user:")
print(f"  Average : {reviews_per_user.mean():.1f}")
print(f"  Minimum : {reviews_per_user.min()}")
print(f"  Maximum : {reviews_per_user.max()}")

# ── Reviews Per Book ──────────────────────────
# How many reviews does each book have?
reviews_per_book = combined_df.groupby('product_id')['user_id'].count()
print(f"\nReviews per book:")
print(f"  Average : {reviews_per_book.mean():.1f}")
print(f"  Minimum : {reviews_per_book.min()}")
print(f"  Maximum : {reviews_per_book.max()}")

# ── Most Reviewed Books ───────────────────────
# Which books have the most reviews in our sample?
print("\nTop 5 most reviewed books:")
top_books = (
    combined_df.groupby(['product_id', 'title'])['rating']
    .count()
    .reset_index()
    .rename(columns={'rating': 'review_count'})
    .sort_values('review_count', ascending=False)
    .head(5)
)
print(top_books)