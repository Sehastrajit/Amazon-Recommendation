# src/data_loader.py
# ─────────────────────────────────────────────────────────────
# Responsible for loading raw data files
# This is imported and used by other files — not run directly
# ─────────────────────────────────────────────────────────────

import gzip
import json
import ast
import os
import urllib.request
import pandas as pd

# ── File paths ────────────────────────────────────────────────
REVIEWS_URL  = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz"
META_URL     = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz"
REVIEWS_PATH = "data/reviews_books.json.gz"
META_PATH    = "data/meta_books.json.gz"


def download_data():
    """Download both raw files if they don't already exist."""

    os.makedirs("data", exist_ok=True)

    for url, path in [(REVIEWS_URL, REVIEWS_PATH), (META_URL, META_PATH)]:
        if os.path.exists(path):
            print(f"✅ Already exists, skipping: {path}")
        else:
            print(f"⬇️  Downloading {path} ...")
            urllib.request.urlretrieve(url, path)
            print(f"✅ Saved: {path}")


def load_reviews(n_rows=50000):
    """
    Load the first n_rows from the reviews file.
    Returns a pandas DataFrame.
    """

    print(f"Loading {n_rows} reviews...")
    reviews = []

    with gzip.open(REVIEWS_PATH, "rb") as f:
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            try:
                reviews.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(reviews)
    print(f"✅ Loaded {len(df)} reviews")
    return df


def load_metadata(n_rows=50000):
    """
    Load the first n_rows from the metadata file.
    Returns a pandas DataFrame.
    """

    print(f"Loading {n_rows} metadata rows...")
    metadata = []

    with gzip.open(META_PATH, "rb") as f:
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            try:
                line_str = line.decode("utf-8")
                metadata.append(ast.literal_eval(line_str))
            except Exception:
                continue

    df = pd.DataFrame(metadata)
    print(f"✅ Loaded {len(df)} metadata rows")
    return df