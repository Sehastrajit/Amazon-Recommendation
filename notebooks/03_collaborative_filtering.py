# notebooks/03_collaborative_filtering.py
# ─────────────────────────────────────────────────────────────
# Phase 3 — Collaborative Filtering
# We train an SVD model on user ratings to recommend books
# ─────────────────────────────────────────────────────────────

import pandas as pd
import sys
import os
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
import pickle
# This lets us import from the src/ folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Step 1: Load the ratings data ─────────────────────────────
# We only need ratings_only.csv for collaborative filtering
# Remember this has just 3 columns: user_id, product_id, rating
# We don't need review text or titles for this model

print("Loading ratings data...")
ratings_df = pd.read_csv("data/processed/ratings_only.csv")

print(f"✅ Loaded {len(ratings_df)} ratings")
print(f"\nFirst 5 rows:")
print(ratings_df.head())

print(f"\nShape: {ratings_df.shape}")
print(f"\nBasic stats:")
print(ratings_df['rating'].describe())

# ── Step 2: Format data for Surprise ──────────────────────────

# Reader tells Surprise what scale our ratings are on
# Our ratings go from 1.0 to 5.0 so we tell it that
reader = Reader(rating_scale=(1, 5))

# Load our dataframe into Surprise's format
# It needs exactly 3 columns in this order: user, item, rating
# Our columns are already in that order: user_id, product_id, rating
data = Dataset.load_from_df(
    ratings_df[['user_id', 'product_id', 'rating']],
    reader
)

print("✅ Data formatted for Surprise")
print(f"   Total ratings: {len(ratings_df)}")

# ── Step 3: Split into train and test sets ────────────────────


# Split the data — 80% for training, 20% for testing
# random_state=42 means the split is always the same every time we run
# this is important for reproducibility — so results don't change each run
trainset, testset = train_test_split(data, test_size=0.20, random_state=42)

# trainset is in Surprise's internal format
# testset is a list of (user, book, real_rating) tuples
print("✅ Data split into train and test sets")
print(f"   Training ratings : {trainset.n_ratings}")
print(f"   Test ratings     : {len(testset)}")

# ── Step 4: Train the SVD model ───────────────────────────────

# Create the SVD model with our chosen settings
# n_factors   = number of latent factors (the hidden patterns)
# n_epochs    = how many times to loop through all the training data
# lr_all      = learning rate — how big each adjustment step is
# reg_all     = regularization — prevents model from memorising training data
model = SVD(
    n_factors=50,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

print("Training SVD model...")
print("(this will take a minute — it's looping through 157,659 ratings 20 times)")

# Train the model on our training set
# This is where gradient descent actually happens
model.fit(trainset)

print("✅ Model trained successfully!")

# ── Step 5: Evaluate the model ────────────────────────────────

# Ask the model to predict ratings for all 39,415 test ratings
# Remember the model never saw these during training
predictions = model.test(testset)

# RMSE — Root Mean Squared Error
# This measures how far off our predictions are on average
# Lower is better
rmse = accuracy.rmse(predictions)

# MAE — Mean Absolute Error
# Another way to measure accuracy — simpler to understand
# "on average our predictions are X stars away from the real rating"
mae = accuracy.mae(predictions)

print(f"\n── Model Accuracy ───────────────────────────")
print(f"   RMSE : {rmse:.4f}")

# ── Step 6: Generate recommendations for a real user ──────────
# part 1

# First let's pick a real user from our dataset
# We'll find a user who has rated many books so recommendations are meaningful
user_ratings_count = ratings_df.groupby('user_id')['rating'].count()
active_user = user_ratings_count.idxmax()  # user with most ratings

print(f"\n── Generating Recommendations ───────────────")
print(f"   Selected user    : {active_user}")
print(f"   Their rating count: {user_ratings_count[active_user]}")

# Find all books this user has NOT rated yet
all_books       = ratings_df['product_id'].unique()
rated_books     = ratings_df[ratings_df['user_id'] == active_user]['product_id'].values
unrated_books   = [b for b in all_books if b not in rated_books]

print(f"   Books rated      : {len(rated_books)}")
print(f"   Books not rated  : {len(unrated_books)}")

# Predict ratings for every unrated book
print(f"\n   Predicting ratings for {len(unrated_books)} unrated books...")
predictions_list = []

for book_id in unrated_books:
    predicted = model.predict(active_user, book_id)
    predictions_list.append((book_id, predicted.est))

# Sort by predicted rating — highest first
predictions_list.sort(key=lambda x: x[1], reverse=True)

# Take top 10
top_10 = predictions_list[:10]

# Load train_data to get book titles
train_df = pd.read_csv("data/processed/train_data.csv")
book_titles = train_df[['product_id', 'title']].drop_duplicates()

print(f"\n── Top 10 Recommended Books ─────────────────")
print(f"{'Title':<45} {'Predicted Rating':>16}")
print("-" * 63)

for book_id, predicted_rating in top_10:
    title_row = book_titles[book_titles['product_id'] == book_id]
    title = title_row['title'].values[0] if len(title_row) > 0 else "Unknown"
    # Trim long titles
    title = title[:43] + ".." if len(title) > 43 else title
    print(f"{title:<45} {predicted_rating:>16.2f}")

# part 2

# Find a realistic user — someone who rated between 20 and 50 books
# This is more realistic than the super reviewer with 2091 ratings
user_ratings_count = ratings_df.groupby('user_id')['rating'].count()
realistic_users = user_ratings_count[
    (user_ratings_count >= 20) & (user_ratings_count <= 50)
]

# Pick the first realistic user
active_user = realistic_users.index[0]

print(f"\n── Generating Recommendations ───────────────")
print(f"   Selected user     : {active_user}")
print(f"   Their rating count: {user_ratings_count[active_user]}")

# Show what books this user actually liked
# So we can see if recommendations make sense
print(f"\n── Books This User Rated Highly (4-5 stars) ─")
user_rated = ratings_df[ratings_df['user_id'] == active_user]
train_df   = pd.read_csv("data/processed/train_data.csv")
book_titles = train_df[['product_id', 'title']].drop_duplicates()

high_ratings = user_rated[user_rated['rating'] >= 4].merge(
    book_titles, on='product_id', how='left'
)
for _, row in high_ratings.head(5).iterrows():
    title = str(row['title'])[:50]
    print(f"   {row['rating']:.0f}⭐  {title}")

# Find all books this user has NOT rated yet
all_books     = ratings_df['product_id'].unique()
rated_books   = user_rated['product_id'].values
unrated_books = [b for b in all_books if b not in rated_books]

print(f"\n   Books rated       : {len(rated_books)}")
print(f"   Books not rated   : {len(unrated_books)}")

# Predict ratings for every unrated book
print(f"\n   Predicting ratings for {len(unrated_books)} unrated books...")
predictions_list = []

for book_id in unrated_books:
    predicted = model.predict(active_user, book_id)
    predictions_list.append((book_id, predicted.est))

# Sort by predicted rating — highest first
predictions_list.sort(key=lambda x: x[1], reverse=True)

# Take top 10
top_10 = predictions_list[:10]

print(f"\n── Top 10 Recommended Books ─────────────────")
print(f"{'Title':<45} {'Predicted Rating':>16}")
print("-" * 63)

for book_id, predicted_rating in top_10:
    title_row = book_titles[book_titles['product_id'] == book_id]
    title = title_row['title'].values[0] if len(title_row) > 0 else "Unknown"
    title = title[:43] + ".." if len(title) > 43 else title
    print(f"{title:<45} {predicted_rating:>16.2f}")

# ── Step 7: Save the trained model ───────────────────────────
os.makedirs("models", exist_ok=True)

with open("models/svd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved to models/svd_model.pkl")
print("   We can load this later without retraining")