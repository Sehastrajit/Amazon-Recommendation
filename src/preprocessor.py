# src/preprocessor.py
# ─────────────────────────────────────────────────────────────
# Responsible for cleaning and combining raw dataframes
# This is imported and used by other files — not run directly
# ─────────────────────────────────────────────────────────────

import pandas as pd
import os


def clean_reviews(df):
    """
    Takes raw reviews dataframe.
    Keeps only useful columns, renames them, drops bad rows.
    Returns cleaned dataframe.
    """

    # Keep only columns we need
    df = df[['reviewerID', 'asin', 'overall',
             'reviewText', 'summary', 'unixReviewTime']]

    # Rename to cleaner names
    df = df.rename(columns={
        'reviewerID'     : 'user_id',
        'asin'           : 'product_id',
        'overall'        : 'rating',
        'reviewText'     : 'review_text',
        'summary'        : 'review_summary',
        'unixReviewTime' : 'timestamp'
    })

    # Drop rows where review text or rating is missing
    df = df.dropna(subset=['review_text', 'rating'])

    print(f"✅ Cleaned reviews: {len(df)} rows")
    return df


def clean_metadata(df):
    """
    Takes raw metadata dataframe.
    Keeps only useful columns, flattens categories, handles missing values.
    Returns cleaned dataframe.
    """

    # Keep only columns we need
    df = df[['asin', 'title', 'categories', 'price']]

    # Rename to match reviews dataframe
    df = df.rename(columns={'asin': 'product_id'})

    # Flatten categories from [["Books", "Fiction"]] to "Books, Fiction"
    def flatten_categories(cat):
        try:
            return ", ".join(cat[0])
        except Exception:
            return ""

    df['categories'] = df['categories'].apply(flatten_categories)

    # Drop rows with missing titles (only ~23 rows)
    df = df.dropna(subset=['title'])

    # Fill missing prices with 0.0
    df['price'] = pd.to_numeric(
        df['price'], errors='coerce'
    ).fillna(0.0)

    print(f"✅ Cleaned metadata: {len(df)} rows")
    return df


def combine(reviews_df, metadata_df):
    """
    Joins reviews and metadata on product_id.
    Returns combined dataframe.
    """

    combined = pd.merge(
        reviews_df,
        metadata_df,
        on='product_id',
        how='inner'
    )

    print(f"✅ Combined dataframe: {len(combined)} rows")
    return combined


def save_processed(df, filename="sample_clean.csv"):
    """Save the cleaned dataframe to data/processed/"""

    os.makedirs("data/processed", exist_ok=True)
    path = f"data/processed/{filename}"
    df.to_csv(path, index=False)
    print(f"✅ Saved to {path}")