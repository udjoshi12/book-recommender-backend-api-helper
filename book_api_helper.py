import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

# ---------------------- Load Dataset ----------------------
books = pd.read_csv("books.csv")
books = books.dropna(subset=['isbn13', 'original_title'])

ratings = pd.read_csv("ratings.csv")        # kept for parity even if not directly used
tags = pd.read_csv("tags.csv")
book_tags = pd.read_csv("book_tags.csv")

# ---------------------- Merge tags like your runtime code ----------------------
book_tags_merged = book_tags.merge(tags, on="tag_id")
top_tags = book_tags_merged.sort_values("count", ascending=False)
book_tags_grpd = top_tags.groupby("goodreads_book_id")["tag_name"].apply(lambda x: " ".join(x[:10])).reset_index()
books = books.merge(book_tags_grpd, how="left", left_on="book_id", right_on="goodreads_book_id")
books.rename(columns={"tag_name": "tags"}, inplace=True)
books["tags"] = books["tags"].fillna("")
books["combined_features"] = books["authors"] + " " + books["tags"]

# ---------------------- TF-IDF + cosine similarity ----------------------
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(books["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ---------------------- Build top-8 neighbor list per index ----------------------
TOP_N = 8
similar_books: dict[int, list[int]] = {}

# ensure we use default RangeIndex 0..n-1 so indices match
books = books.reset_index(drop=True)

for i in range(len(books)):
    row = cosine_sim[i]
    # argsort gives ascending order; take last TOP_N+1, drop self (largest), reverse
    top_idx = np.argsort(row)[-TOP_N-1:-1][::-1]
    similar_books[int(i)] = [int(j) for j in top_idx.tolist()]

# ---------------------- Save minimal, fast-to-load data ----------------------
with open("books_data.pkl", "wb") as f:
    pickle.dump(
        {
            "books": books,                 # pandas DataFrame (keeps your downstream columns)
            "similar_books": similar_books  # dict[int, list[int]]
        },
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print("✅ Precomputed file written: books_data.pkl")
