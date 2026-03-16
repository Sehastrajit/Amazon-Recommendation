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