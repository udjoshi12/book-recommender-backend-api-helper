from urllib.parse import quote_plus
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
import pandas as pd
import pickle
from rapidfuzz import process
import httpx
import traceback

# ---------------------- MongoDB Setup ----------------------
username = quote_plus("sky-rocket")
password = quote_plus("@Udjoshi12")
MONGO_URI = f"mongodb+srv://{username}:{password}@beginning-of-the-end.iz3nbat.mongodb.net/?retryWrites=true&w=majority"
client = AsyncIOMotorClient(MONGO_URI)
db = client["Beginning-of-the-end"]
users_collection = db["users"]
books_collection = db["books"]

# ---------------------- Password Hashing ----------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def hash_password(password: str): 
    return pwd_context.hash(password)
def verify_password(plain_password: str, hashed_password: str): 
    return pwd_context.verify(plain_password, hashed_password)

# ---------------------- Load Precomputed Data ----------------------
with open("books_data.pkl", "rb") as f:
    data = pickle.load(f)

# books: pandas DataFrame; similar_books: dict[int, list[int]]
books: pd.DataFrame = data["books"]
similar_books = data["similar_books"]

# EXACT behavior of your fuzzy match helper: pick the dup with highest ratings_count
title_to_index_series = pd.Series(
    books.index,
    index=books["original_title"].str.strip().str.lower()
)  # keep as Series only for quick lookup of titles → row indices

# ---------------------- FastAPI Setup ----------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Models ----------------------
class RecommendationRequest(BaseModel):
    title: str

class BookOut(BaseModel):
    title: str
    author: Optional[str]
    average_rating: Optional[float]
    isbn13: Optional[str]
    image_url: Optional[str]

class BookOutFeedback(BaseModel):
    title: str
    author: Optional[str] = None
    average_rating: Optional[float] = None
    isbn13: Optional[str] = None
    image_url: Optional[str] = None
    likes: int = 0

class RecommendationResponse(BaseModel):
    recommendations: List[BookOut]
    message: Optional[str] = None

class Feedback(BaseModel):
    userId: str
    liked_books: List[str] = []
    disliked_books: List[str] = []

class UserSignup(BaseModel):
    name: str
    email: str
    password: str
    bio: Optional[str] = ""

class UserLogin(BaseModel):
    email: str
    password: str

# ---------------------- Helpers ----------------------
def get_best_match(title: str) -> int | None:
    # Use the set of titles we have as candidates
    candidate_titles = title_to_index_series.index.tolist()

    match = process.extractOne(title.lower(), candidate_titles)
    if not match or match[1] <= 85:
        return None

    matched_title = match[0]
    matching_rows = books[books['original_title'].str.lower() == matched_title.lower()]
    if matching_rows.empty:
        return None

    # Pick the row (index) with highest ratings_count among duplicates (your original behavior)
    best_row = matching_rows.sort_values(by='ratings_count', ascending=False).iloc[0]
    return int(best_row.name)

def get_user_collection():
    return db['users']

async def fetch_book_details_from_google(title: str):
    try:
        async with httpx.AsyncClient() as client:
            query = title.replace(" ", "+")
            url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{query}"
            response = await client.get(url)
            data = response.json()
            if data.get("totalItems", 0) == 0:
                return None
            item = data["items"][0]["volumeInfo"]
            return {
                "title": item.get("title", title),
                "author": ", ".join(item.get("authors", [])),
                "average_rating": item.get("averageRating"),
                "isbn13": next((id["identifier"] for id in item.get("industryIdentifiers", []) if id["type"] == "ISBN_13"), None),
                "image_url": item.get("imageLinks", {}).get("thumbnail")
            }
    except Exception as e:
        print("Google Books API Error:", e)
        return None

# ---------------------- Routes ----------------------
@app.post("/signup")
async def signup(user: UserSignup):
    try:
        users = get_user_collection()
        existing_user = await users.find_one({"userId": user.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists.")

        await users.insert_one({
            "userId": user.email,
            "name": user.name,
            "password": hash_password(user.password),
            "bio": user.bio,
            "liked_books": [],
            "disliked_books": []
        })
        return {"message": "User registered successfully"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/login")
async def login(user: UserLogin):
    try:
        existing_user = await users_collection.find_one({"userId": user.email})
        if not existing_user or not verify_password(user.password, existing_user["password"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        return {
            "message": "Login successful",
            "user": {
                "userId": existing_user["userId"],
                "name": existing_user["name"],
                "email": existing_user["userId"],
                "bio": existing_user.get("bio", "")
            }
        }
    except Exception as e:
        traceback.print_exc()
        return {"message": "Internal error occurred.", "user": ""}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_book(req: RecommendationRequest):
    try:
        idx = get_best_match(req.title)
        if idx is None:
            # Keep your original fallback using work_ratings_count
            popular_books = books.sort_values("work_ratings_count", ascending=False).head(8)
            recommendations = [
                {
                    "title": row["title"],
                    "author": row.get("authors", "Unknown"),
                    "average_rating": row.get("average_rating"),
                    "isbn13": str(int(row.get("isbn13"))) if pd.notnull(row.get("isbn13")) else None,
                    "image_url": row.get("image_url"),
                }
                for _, row in popular_books.iterrows()
            ]
            return {"recommendations": recommendations, "message": "Title not found. Showing popular books instead."}

        # Use precomputed neighbors (plain dict[int, list[int]])
        recommended_indices = similar_books[int(idx)]

        recommendations = []
        for i in recommended_indices:
            book = books.iloc[int(i)]
            recommendations.append({
                "title": book["title"],
                "author": book.get("authors", "Unknown"),
                "average_rating": book.get("average_rating"),
                "isbn13": str(int(book.get("isbn13"))) if pd.notnull(book.get("isbn13")) else None,
                "image_url": book.get("image_url"),
            })

        return {"recommendations": recommendations}

    except Exception as e:
        traceback.print_exc()
        return {"recommendations": [], "message": "An internal error occurred."}

@app.post("/feedback")
async def save_feedback(feedback: Feedback):
    try:
        existing_user = await users_collection.find_one({"userId": feedback.userId})
        already_liked = existing_user.get("liked_books", []) if existing_user else []
        new_likes = [book for book in feedback.liked_books if book not in already_liked]

        if existing_user:
            await users_collection.update_one(
                {"userId": feedback.userId},
                {"$addToSet": {"liked_books": {"$each": new_likes}}}
            )
        else:
            await users_collection.insert_one({
                "userId": feedback.userId,
                "liked_books": new_likes
            })

        for title in new_likes:
            matched_index = get_best_match(title)
            matched_title = books.iloc[int(matched_index)]["title"] if matched_index is not None else title
            existing_book = await books_collection.find_one({"title": matched_title})

            if existing_book:
                new_likes_count = max(existing_book.get("likes", 0) + 1, 0)
                await books_collection.update_one(
                    {"_id": existing_book["_id"]},
                    {"$set": {"likes": new_likes_count}}
                )
            else:
                details = await fetch_book_details_from_google(title)
                if details:
                    details["likes"] = 1
                    await books_collection.insert_one(details)

        return {"message": "Feedback saved and book info updated"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")

@app.get("/popular-books")
async def get_popular_books(limit: int = 10):
    try:
        cursor = books_collection.find({"likes": {"$gt": 0}}).sort("likes", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return {
            "recommendations": [
                {
                    "title": f"{doc.get('title', 'Untitled')} ({doc.get('likes', 0)} likes)",
                    "author": doc.get("author"),
                    "average_rating": doc.get("average_rating"),
                    "isbn13": doc.get("isbn13"),
                    "image_url": doc.get("image_url"),
                } for doc in docs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching popular books: {str(e)}")

@app.get("/top-rated-books")
def get_top_rated_books():
    try:
        top_books = books[books["ratings_count"] > 500].sort_values("average_rating", ascending=False).head(8)
        return top_books[["original_title", "authors", "book_id", "average_rating", "image_url", "isbn13", "title"]].to_dict(orient="records")
    except Exception as e:
        return {"Error": str(e)}

@app.get("/discover-books")
def get_random_books():
    try:
        books_sample = books[books["average_rating"] > 3.8].copy()
        books_sample = books_sample.dropna(subset=['isbn13'])
        books_sample['isbn13'] = books_sample['isbn13'].astype(str)
        sample_size = min(16, len(books_sample))
        books_sample = books_sample.sample(sample_size)
        return books_sample[["original_title", "authors", "book_id", "average_rating", "image_url", "isbn13", "title"]].to_dict(orient="records")
    except Exception as e:
        return {"Error": str(e)}

@app.get("/books-by-author")
def get_books_by_author(author: str = Query(..., description="Author name to search for")):
    try:
        same_author_books = books[books["authors"].str.contains(author, case=False, na=False)]
        if same_author_books.empty:
            return {"message": "No other books for this author"}
        return same_author_books[["original_title", "authors", "book_id", "average_rating", "image_url"]].to_dict(orient="records")
    except Exception as e:
        return {"Error": str(e)}
