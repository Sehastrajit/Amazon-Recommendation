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