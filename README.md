# cis655-working-files
current working files for book rec project

# app.py
```
from fastapi import FastAPI, HTTPException, Query, Depends, Path, Body
from pydantic import BaseModel
import psycopg2
import os
from typing import List, Optional
from google.cloud import pubsub_v1, bigquery
import json
from auth import router as auth_router, verify_token
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from datetime import datetime
from chatbot import ask_chatbot
from fastapi.security import OAuth2PasswordBearer


app = FastAPI()

# connection
from db import get_db_connection

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route for the homepage
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

# Route for auth.py
app.include_router(auth_router)

# Protect routes using OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# removed for now to hepl debug
# Use this dependency on any route that needs authentication
#def get_current_user(token: str = Depends(oauth2_scheme)):
    #return verify_token(token)

# Removed maturity_rating
class Book(BaseModel):
    title: Optional[str]
    authors: Optional[str]
    publisher: Optional[str]
    published_date: Optional[str]
    description: Optional[str]
    isbn_10: Optional[str]
    isbn_13: Optional[str]
    reading_mode_text: Optional[bool]
    reading_mode_image: Optional[bool]
    page_count: Optional[int]
    categories: Optional[str]
    image_small: Optional[str]
    image_large: Optional[str]
    language: Optional[str]
    sale_country: Optional[str]
    list_price_amount: Optional[float]
    list_price_currency: Optional[str]
    buy_link: Optional[str]
    web_reader_link: Optional[str]
    embeddable: Optional[bool]
    grade: Optional[str]


@app.get("/books", response_model=List[Book])
def get_books(
    title: str = Query(None),
    authors: str = Query(None),
    categories: str = Query(None),
    grade: str = Query(None),
    description: str = Query(None)
):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Build query with filters
        query = """
            SELECT title, authors, publisher, published_date, description,
                   isbn_10, isbn_13, reading_mode_text, reading_mode_image,
                   page_count, categories, image_small, image_large, language,
                   sale_country, list_price_amount, list_price_currency,
                   buy_link, web_reader_link, embeddable, grade
            FROM books
        """
        filters = []
        values = []

        if title:
            filters.append("LOWER(title) LIKE %s")
            values.append(f"%{title.lower()}%")
        if authors:
            filters.append("LOWER(authors) LIKE %s")
            values.append(f"%{authors.lower()}%")
        if categories:
            filters.append("LOWER(categories) LIKE %s")
            values.append(f"%{categories.lower()}%")
        if grade:
            filters.append("LOWER(grade) = %s")
            values.append(grade.lower())
        if description:
            filters.append("LOWER(description) LIKE %s")
            values.append(f"%{description.lower()}%")

        if filters:
            query += " WHERE " + " AND ".join(filters)

        cur.execute(query, values)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()

        return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# GET book by ISBN

@app.get("/books/isbn/{isbn}", response_model=Book)
def get_book_by_isbn(isbn: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT title, authors, publisher, published_date, description,
                   isbn_10, isbn_13, reading_mode_text, reading_mode_image,
                   page_count, categories, image_small, image_large, language,
                   sale_country, list_price_amount, list_price_currency,
                   buy_link, web_reader_link, embeddable, grade
            FROM books
            WHERE isbn_13 = %s OR isbn_10 = %s
        """, (isbn, isbn))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            columns = [desc[0] for desc in cur.description]
            return dict(zip(columns, row))
        else:
            raise HTTPException(status_code=404, detail="Book not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Initialize Pub/Sub publisher
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path("book-recommendations-456120", "new-book")

# POST create a new book
@app.post("/books", response_model=Book)
def create_book(book: Book):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Insert book into the database
        cur.execute("""
            INSERT INTO books (title, authors, publisher, published_date, description,
                               isbn_10, isbn_13, reading_mode_text, reading_mode_image,
                               page_count, categories, image_small, image_large, language,
                               sale_country, list_price_amount, list_price_currency,
                               buy_link, web_reader_link, embeddable, grade)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            book.title, book.authors, book.publisher, book.published_date, book.description,
            book.isbn_10, book.isbn_13, book.reading_mode_text, book.reading_mode_image,
            book.page_count, book.categories, book.image_small, book.image_large, book.language,
            book.sale_country, book.list_price_amount, book.list_price_currency,
            book.buy_link, book.web_reader_link, book.embeddable, book.grade
        ))
        conn.commit()
        cur.close()
        conn.close()

        # Publish to Pub/Sub (if description + ISBN exist)
        if book.description and book.isbn_10:
            payload = json.dumps({
                "isbn_10": book.isbn_10,
                "description": book.description
            }).encode("utf-8")
            publisher.publish(topic_path, payload)
            print(f" Published book {book.isbn_10} to Pub/Sub")

        return book

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# PUT update a book by isbn
@app.put("/books/{isbn}", response_model=Book)
def update_book(isbn: str, book: Book):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Update book in the database
        cur.execute("""
            UPDATE books
            SET title = %s, authors = %s, publisher = %s, published_date = %s, description = %s,
                isbn_10 = %s, isbn_13 = %s, reading_mode_text = %s, reading_mode_image = %s,
                page_count = %s, categories = %s, image_small = %s, image_large = %s, language = %s,
                sale_country = %s, list_price_amount = %s, list_price_currency = %s,
                buy_link = %s, web_reader_link = %s, embeddable = %s, grade = %s
            WHERE isbn_13 = %s OR isbn_10 = %s
            RETURNING title, authors, publisher, published_date, description,
                       isbn_10, isbn_13, reading_mode_text, reading_mode_image,
                       page_count, categories, image_small, image_large, language,
                       sale_country, list_price_amount, list_price_currency,
                       buy_link, web_reader_link, embeddable, grade;
        """, (book.title, book.authors, book.publisher, book.published_date, book.description,
              book.isbn_10, book.isbn_13, book.reading_mode_text, book.reading_mode_image,
              book.page_count, book.categories, book.image_small, book.image_large, book.language,
              book.sale_country, book.list_price_amount, book.list_price_currency,
              book.buy_link, book.web_reader_link, book.embeddable, book.grade,
              isbn, isbn))
        
        # Commit changes and return the updated data
        conn.commit()
        cur.close()
        conn.close()
        return book
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# DELETE a book by isbn

@app.delete("/books/{isbn}")
def delete_book(isbn: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Delete book from the database
        cur.execute("""
            DELETE FROM books
            WHERE isbn_13 = %s OR isbn_10 = %s;
        """, (isbn, isbn))
        
        conn.commit()
        cur.close()
        conn.close()
        return {"message": "Book deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# USER options

# GET all users
@app.get("/users")
def get_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    cursor.close()
    conn.close()
    return users

# GET user by ID
@app.get("/users/{user_id}")
def get_user(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if user:
        return user
    raise HTTPException(status_code=404, detail="User not found")

# POST create new user
@app.post("/users")
def create_user(username: str, email: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, email) VALUES (%s, %s) RETURNING user_id, username, email",
        (username, email)
    )
    user = cursor.fetchone()
    conn.commit()
    cursor.close()
    conn.close()
    return {
        "user_id": user[0],
        "username": user[1],
        "email": user[2]
    }

# PUT update user
@app.put("/users/{user_id}")
def update_user(user_id: int, username: str, email: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET username = %s, email = %s WHERE user_id = %s RETURNING *",
        (username, email, user_id)
    )
    user = cursor.fetchone()
    conn.commit()
    cursor.close()
    conn.close()
    if user:
        return user
    raise HTTPException(status_code=404, detail="User not found")

# DELETE user
@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE user_id = %s RETURNING *", (user_id,))
    user = cursor.fetchone()
    conn.commit()
    cursor.close()
    conn.close()
    if user:
        return {"message": "User deleted"}
    raise HTTPException(status_code=404, detail="User not found")


## USER_BOOKS 

class UserBook(BaseModel):
    user_id: int
    title: str
    author: str
    isbn_10: Optional[str] = None  
    rating: Optional[int] = None
    read_at: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[int] = None
    notes: Optional[str] = None


# get all user-book entries
@app.get("/user_books")
def get_all_user_books():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_books")
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    return records

# get books for specific user

@app.get("/user_books/{user_id}")
def get_user_books(user_id: int, status: Optional[str] = Query(None)):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT b.title, b.authors, b.image_small, b.categories, b.grade,
                   ub.rating, ub.status, ub.progress, ub.notes, ub.read_at
            FROM user_books ub
            JOIN books b
              ON LOWER(ub.title) = LOWER(b.title)
             AND LOWER(ub.author) = LOWER(b.authors)
            WHERE ub.user_id = %s
        """
        params = [user_id]

        if status:
            query += " AND ub.status = %s"
            params.append(status)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        cursor.close()
        conn.close()

        return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# add review using Pub/Sub

class Review(BaseModel):
    user_id: int
    isbn_10: Optional[str]
    title: str
    author: str
    rating: Optional[int]
    notes: Optional[str]
    status: Optional[str] = "completed"
    reviewed_at: Optional[str] = None


@app.post("/reviews")
def submit_review(entry: Review):
    try:
        publish_rating_event(entry.dict())
        return {"message": "Rating submitted to Pub/Sub successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to publish rating event: {str(e)}")


# update entry by id
@app.put("/user_books/{entry_id}")
def update_user_book(entry_id: int, user_book: UserBook):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE user_books
        SET user_id = %s, title = %s, author = %s, rating = %s,
            status = %s, progress = %s, notes = %s
        WHERE id = %s
        RETURNING *;
    """, (
        user_book.user_id,
        user_book.title,
        user_book.author,
        user_book.rating,
        user_book.status,
        user_book.progress,
        user_book.notes,
        entry_id
    ))
    updated_entry = cursor.fetchone()
    conn.commit()
    cursor.close()
    conn.close()
    if updated_entry:
        return updated_entry
    raise HTTPException(status_code=404, detail="Entry not found")

# delete entry by id
@app.delete("/user_books/{entry_id}")
def delete_user_book(entry_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM user_books WHERE id = %s RETURNING *", (entry_id,))
    deleted = cursor.fetchone()
    conn.commit()
    cursor.close()
    conn.close()
    if deleted:
        return {"message": "User-book entry deleted"}
    raise HTTPException(status_code=404, detail="Entry not found")


### SQL based recommendations

@app.get("/recommendations/for-user/{user_id}")
def recommendations(user_id: int, limit: int = 5):
    conn = get_db_connection()
    cur  = conn.cursor()


    cur.execute("SELECT grade FROM users WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    user_grade = row[0] if row and row[0] else None

    # If user.grade is NULL, derive grade(s) from their rated books
    if not user_grade:
        cur.execute("""
            SELECT DISTINCT b.grade
            FROM   user_books ub
            JOIN   books b
              ON  (ub.isbn_10 IS NOT NULL AND ub.isbn_10 = b.isbn_10)
               OR (LOWER(ub.title)  = LOWER(b.title)
               AND LOWER(ub.author) = LOWER(b.authors))
            WHERE  ub.user_id = %s
              AND  ub.rating IS NOT NULL
        """, (user_id,))
        grades = [g[0] for g in cur.fetchall() if g[0] is not None]
    else:
        grades = [user_grade]

   
    if grades:
        grade_filter_sql = "b.grade IN %s"
        grade_filter_val = (tuple(grades),)
    else:
        # no grade info – use all grades
        grade_filter_sql = "TRUE"
        grade_filter_val = tuple()

    cur.execute(f"""
        WITH candidate AS (
            SELECT b.isbn_10, b.title, b.authors, b.grade,
                   COALESCE(AVG(ub2.rating),0) AS avg_rating,
                   COUNT(ub2.rating)           AS num_ratings
            FROM books b
            LEFT JOIN user_books ub2 ON b.isbn_10 = ub2.isbn_10
            WHERE {grade_filter_sql}
              AND NOT EXISTS (
                SELECT 1 FROM user_books ub
                WHERE  ub.user_id = %s
                  AND (
                       (ub.isbn_10 IS NOT NULL AND ub.isbn_10 = b.isbn_10)
                    OR (LOWER(ub.title)  = LOWER(b.title)
                    AND LOWER(ub.author) = LOWER(b.authors))
                  )
              )
            GROUP BY b.isbn_10, b.title, b.authors, b.grade
        )
        SELECT * FROM candidate
        ORDER BY avg_rating DESC, num_ratings DESC
        LIMIT %s;
    """, grade_filter_val + (user_id, limit))

    rows = cur.fetchall()


    if not rows:
        cur.execute("""
            SELECT title, authors, grade
            FROM   books
            ORDER  BY RANDOM()
            LIMIT  %s;
        """, (limit,))
        rows = cur.fetchall()

    cols = [d[0] for d in cur.description]
    cur.close(); conn.close()
    return [dict(zip(cols, r)) for r in rows]


def publish_rating_event(entry: dict):
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path("book-recommendations-456120", "rate-book")
    payload = json.dumps(entry).encode("utf-8")
    publisher.publish(topic_path, payload)


## BigQuery endpoint


@app.get("/recommendations/by-metadata")
def recommend_by_metadata(
    isbn_10: Optional[str] = Query(None),
    title: Optional[str] = Query(None),
    authors: Optional[str] = Query(None),
    grade: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    limit: int = Query(10, description="Number of recommendations to return")
):
    try:
        client = bigquery.Client()

        filters = []
        if isbn_10:
            filters.append("LOWER(isbn_10) = LOWER(@isbn_10)")
        if title:
            filters.append("LOWER(REGEXP_REPLACE(title, r'[^a-zA-Z0-9]', '')) = LOWER(REGEXP_REPLACE(@title, r'[^a-zA-Z0-9]', ''))")
        if authors:
            filters.append("LOWER(REGEXP_REPLACE(authors, r'[^a-zA-Z0-9]', '')) = LOWER(REGEXP_REPLACE(@authors, r'[^a-zA-Z0-9]', ''))")
        if grade:
            filters.append("grade = @grade")
        if keyword:
            filters.append(
                """(
                    LOWER(title) LIKE CONCAT('%', LOWER(@keyword), '%') OR
                    LOWER(authors) LIKE CONCAT('%', LOWER(@keyword), '%') OR
                    LOWER(description) LIKE CONCAT('%', LOWER(@keyword), '%')
                )"""
            )

        if not filters:
            raise HTTPException(status_code=400, detail="At least one filter must be provided.")

        where_clause = " AND ".join(filters)

        query = f"""
        WITH target AS (
            SELECT embedding
            FROM `book-recommendations-456120.book_recs.book_embeddings`
            WHERE {where_clause}
            LIMIT 1
        ),
        scored AS (
            SELECT
                b2.isbn_10,
                b2.title,
                b2.authors,
                b2.grade,
                b2.image_small,
                b2.web_reader_link,
                b2.thumbs_up,
                b2.thumbs_down,
                (
                    SELECT SUM(e1 * e2)
                    FROM UNNEST(b2.embedding) AS e1 WITH OFFSET i
                    JOIN UNNEST(target.embedding) AS e2 WITH OFFSET j
                    ON i = j
                ) /
                (
                    SQRT((SELECT SUM(POW(val, 2)) FROM UNNEST(b2.embedding) AS val)) *
                    SQRT((SELECT SUM(POW(val2, 2)) FROM UNNEST(target.embedding) AS val2))
                ) AS similarity
            FROM `book-recommendations-456120.book_recs.book_embeddings` b2, target
            WHERE b2.embedding IS NOT NULL
        )
        SELECT *
        FROM scored
        ORDER BY similarity DESC
        LIMIT @limit
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("isbn_10", "STRING", isbn_10),
                bigquery.ScalarQueryParameter("title", "STRING", title),
                bigquery.ScalarQueryParameter("authors", "STRING", authors),
                bigquery.ScalarQueryParameter("grade", "STRING", grade),
                bigquery.ScalarQueryParameter("keyword", "STRING", keyword),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )

        job = client.query(query, job_config=job_config)
        results = job.result()
        return [dict(row) for row in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in recommendation query: {str(e)}")

## AI Chatbot

@app.post("/chat")
def chat_with_user(message: str = Body(..., embed=True)):
    try:
        response = ask_chatbot(message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


## Feedback endpoint

class Feedback(BaseModel):
    user_id: int
    isbn_10: str
    feedback: str  # 'like', 'dislike', 'interested'

@app.post("/feedback")
def submit_feedback(entry: Feedback):
    if entry.feedback not in ['like', 'dislike', 'interested']:
        raise HTTPException(status_code=400, detail="Invalid feedback type.")

    # Store feedback in PostgreSQL
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO recommendation_feedback (user_id, isbn_10, feedback)
            VALUES (%s, %s, %s)
        """, (entry.user_id, entry.isbn_10, entry.feedback))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"PostgreSQL error: {str(e)}")
    finally:
        cur.close()
        conn.close()

    # Update BigQuery thumbs
    try:
        if entry.feedback in ['like', 'dislike']:
            bq_client = bigquery.Client()
            field = "thumbs_up" if entry.feedback == "like" else "thumbs_down"

            query = f"""
                UPDATE `book-recommendations-456120.book_recs.book_embeddings`
                SET {field} = IFNULL({field}, 0) + 1
                WHERE isbn_10 = @isbn_10
            """

            job = bq_client.query(
                query,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("isbn_10", "STRING", entry.isbn_10)
                    ]
                )
            )
            job.result()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery error: {str(e)}")

    return {"message": "Feedback submitted successfully."}


    # User Review 


   # @app.post("/review")
    #def submit_review(entry: Review):
        #try:
            # If no timestamp is given, add it
            #if not entry.reviewed_at:
                #entry.reviewed_at = datetime.utcnow().isoformat()

            #publisher = pubsub_v1.PublisherClient()
            #topic_path = publisher.topic_path("book-recommendations-456120", "user-reviews")

            #payload = json.dumps(entry.dict()).encode("utf-8")
            #publisher.publish(topic_path, payload)

            #return {"message": "Review submitted successfully."}

        #except Exception as e:
            #raise HTTPException(status_code=500, detail=f"Error submitting review: {str(e)}")


   # @app.get("/secure")
   # def secure_endpoint(current_user: dict = Depends(get_current_user)):
      #  return {"message": "Hello", "user": current_user}
```

# auth.py
```
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from db import get_db_connection
import os

router = APIRouter()

# JWT Config
SECRET_KEY = os.environ.get("SECRET_KEY", "super-secret-key-for-dev")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Models
class User(BaseModel):
    username: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Helper Functions
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print("Decoded token payload:", payload) 
        return payload  
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# Register Endpoint
@router.post("/register")
def register(user: User):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM users WHERE username = %s OR email = %s", (user.username, user.email))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Username or email already exists")

        hashed_password = get_password_hash(user.password)
        cur.execute(
            "INSERT INTO users (username, email, hashed_password) VALUES (%s, %s, %s)",
            (user.username, user.email, hashed_password)
        )
        conn.commit()
        return {"message": "User registered successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

# Login Endpoint

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT user_id, username, email FROM users WHERE username = %s", (form_data.username,))
        user = cur.fetchone()
        if not user:
            raise HTTPException(status_code=400, detail="User not found")

        # Create token without checking password
        access_token = create_access_token(data={"sub": user[1], "user_id": user[0]})
        return {"access_token": access_token, "token_type": "bearer"}
    finally:
        cur.close()
        conn.close()
```


# index.html
```
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Book Recommendation App</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #f0f4f8;
      margin: 0;
      padding: 20px;
      color: #333;
    }

    h1 {
      color: #2c3e50;
      font-size: 2rem;
      margin-bottom: 10px;
    }

    .section {
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      padding: 20px;
      margin-bottom: 30px;
      max-width: 800px;
    }

    input, textarea, select, button {
      width: 100%;
      padding: 10px;
      margin-top: 10px;
      margin-bottom: 15px;
      font-size: 1rem;
      border-radius: 6px;
      border: 1px solid #ccc;
      box-sizing: border-box;
    }

    button {
      background-color: #4a90e2;
      color: white;
      border: none;
      cursor: pointer;
      transition: background 0.3s ease;
    }

    button:hover {
      background-color: #357ABD;
    }

    .book {
      display: flex;
      background: #fafafa;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 15px;
      margin-top: 10px;
      align-items: flex-start;
    }

    .book img {
      height: 100px;
      margin-right: 20px;
      border-radius: 4px;
    }

    .book strong {
      font-size: 1.1rem;
    }

    .book em {
      display: block;
      color: #666;
      margin-top: 5px;
    }

    textarea {
      resize: vertical;
    }
  </style>
</head>
<body>
  <h1>Book Recommender</h1>

  <!-- Book Review Section -->
  <div class="section">
    <h2>Leave a Review</h2>
    <input type="text" id="reviewTitle" placeholder="Search by Title">
    <button onclick="findBookForReview()">Search</button>
    <div id="reviewBookResult"></div>
  </div>

  <!-- My Books Section -->
  <div class="section">
    <h2>My Books</h2>
    <button onclick="loadMyBooks()">View My Books</button>
    <div id="myBooksList"></div>
  </div>  

  <!-- Metadata Recommendation Section -->
  <div class="section">
    <h2>Find Book Recommendations</h2>
    <input type="text" id="recTitle" placeholder="Title">
    <input type="text" id="recAuthors" placeholder="Author(s)">
    <input type="text" id="recIsbn" placeholder="ISBN-10">
    <input type="text" id="recKeyword" placeholder="Keyword">
    <select id="recGrade">
      <option value="">Select Grade</option>
      <option value="K">K</option><option value="1">1</option><option value="2">2</option>
      <option value="3">3</option><option value="4">4</option><option value="5">5</option>
      <option value="6">6</option><option value="7">7</option><option value="8">8</option>
    </select>
    <button onclick="getMetadataRecommendations()">Get Recommendations</button>
    <div id="recommendations"></div>
  </div>

  
  <script>
    const currentUserId = 1;

    function findBookForReview() {
      const title = document.getElementById('reviewTitle').value;
      if (!title) return alert("Enter a book title");

      fetch(`/books?title=${encodeURIComponent(title)}`)
      .then(res => res.json())
      .then(books => {
        const result = books[0];
        const div = document.getElementById("reviewBookResult");
        if (!result) return div.innerHTML = "<p>No book found.</p>";

        div.innerHTML = `
          <div class="book">
            <img src="${result.image_small || '/static/placeholder.png'}" alt="cover">
            <div>
              <strong>${result.title}</strong><br>
              <em>${result.authors}</em><br>
              <label for="reviewRating">Rating (1–5):</label>
              <input type="number" id="reviewRating" min="1" max="5"><br>
              <label for="reviewNotes">Notes:</label><br>
              <textarea id="reviewNotes" rows="3"></textarea><br>
              <button onclick="submitReview('${result.title}', '${result.authors}', '${result.isbn_10 || ''}')">Submit Review</button>
            </div>
          </div>
        `;
      })
      .catch(err => {
        console.error("Search error:", err);
        alert("Error fetching book");
      });
    }

    function submitReview(title, author, isbn_10) {
      const rating = parseInt(document.getElementById("reviewRating").value);
      const notes = document.getElementById("reviewNotes").value;

      if (!rating || rating < 1 || rating > 5) return alert("Rating must be 1–5");

      fetch("/reviews", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: currentUserId,
          title,
          author,
          isbn_10,
          rating,
          notes,
          status: "completed"
        })
      })
      .then(res => {
        if (!res.ok) throw new Error("Failed to submit");
        alert("Review submitted!");
        document.getElementById("reviewBookResult").innerHTML = "";
      })
      .catch(err => {
        console.error("Review error:", err);
        alert("Error submitting review");
      });
    }

    function rateBook(isbn_10, title, author, feedback) {
      fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: currentUserId, isbn_10, feedback })
      })
      .then(() => alert('Feedback submitted!'))
      .catch(error => {
        console.error('Feedback failed:', error);
      });
    }

    function loadMyBooks() {
      fetch(`/user_books/1`)
        .then(res => res.json())
        .then(books => {
          const container = document.getElementById("myBooksList");
          container.innerHTML = "";
          if (!books.length) return container.innerHTML = "<p>No books found.</p>";

          books.forEach(book => {
            const div = document.createElement("div");
            div.className = "book";
            div.innerHTML = `
              <img src="${book.image_small || '/static/placeholder.png'}" alt="cover">
              <div>
                <strong>${book.title}</strong><br>
                <em>${book.authors}</em><br>
                <p><b>Rating:</b> ${book.rating || "N/A"}</p>
                <p><b>Status:</b> ${book.status || "N/A"}</p>
                <p><b>Progress:</b> ${book.progress || "N/A"}%</p>
                <p><b>Notes:</b> ${book.notes || ""}</p>
              </div>
            `;
            container.appendChild(div);
          });
        })
        .catch(err => {
          console.error("Error loading books:", err);
          alert("Failed to load reviewed books.");
        });
    }


    function getMetadataRecommendations() {
      const title = document.getElementById('recTitle').value;
      const authors = document.getElementById('recAuthors').value;
      const isbn_10 = document.getElementById('recIsbn').value;
      const keyword = document.getElementById('recKeyword').value;
      const grade = document.getElementById('recGrade').value;

      const params = new URLSearchParams();
      if (title) params.append('title', title);
      if (authors) params.append('authors', authors);
      if (isbn_10) params.append('isbn_10', isbn_10);
      if (keyword) params.append('keyword', keyword);
      if (grade) params.append('grade', grade);

      fetch(`/recommendations/by-metadata?${params.toString()}`)
      .then(res => res.json())
      .then(books => {
        const resultsDiv = document.getElementById('recommendations');
        resultsDiv.innerHTML = '';
        books.forEach(book => {
          const div = document.createElement('div');
          div.className = 'book';
          div.innerHTML = `
            <img src="${book.image_small || '/static/placeholder.png'}" alt="cover">
            <div>
              <strong>${book.title}</strong><br>
              <em>${book.authors}</em><br>
              👍 ${book.thumbs_up || 0} 👎 ${book.thumbs_down || 0} ⭐ ${book.avg_rating?.toFixed(1) || "N/A"}<br>
              <button onclick="rateBook('${book.isbn_10}', '${book.title}', '${book.authors}', 'like')">👍</button>
              <button onclick="rateBook('${book.isbn_10}', '${book.title}', '${book.authors}', 'dislike')">👎</button><br>
              ${book.web_reader_link ? `<a href="${book.web_reader_link}" target="_blank">Read</a>` : ''}
            </div>
          `;
          resultsDiv.appendChild(div);
        });
      })
      .catch(error => {
        alert("Recommendation failed.");
        console.error("Recommendation error:", error);
      });
    }

  </script>
</body>
</html>
```

# generate_embeddings.py
```
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.cloud import bigquery
import psycopg2

# Constants
PROJECT_ID = "book-recommendations-456120"
REGION = "us-central1"
MODEL_ID = "text-embedding-005" 

# Initialize Vertex AI and BigQuery client
vertexai.init(project=PROJECT_ID, location=REGION)
model = TextEmbeddingModel.from_pretrained(MODEL_ID)
bq_client = bigquery.Client()

# DB connection
conn = psycopg2.connect(
    dbname="book_recommendations_db",
    user="postgres",
    password="pass",
    host="127.0.0.1",
    port="5432"
)
cur = conn.cursor()

# Query books with sufficient description
cur.execute("""
    SELECT isbn_10, title, authors, grade, description, image_small, web_reader_link
    FROM books
    WHERE description IS NOT NULL AND LENGTH(description) > 10
""")

books = cur.fetchall()

print(f"Found {len(books)} books to embed")

for isbn_10, title, authors, grade, description, image_small, web_reader_link in books:
    try:
        embedding = model.get_embeddings([description])[0].values

        row = {
            "isbn_10": isbn_10,
            "title": title,
            "authors": authors,
            "grade": grade,
            "description": description,
            "image_small": image_small,
            "web_reader_link": web_reader_link,
            "thumbs_up": 0,
            "thumbs_down": 0,
            "embedding": embedding
        } 


        errors = bq_client.insert_rows_json(
            "book-recommendations-456120.book_recs.book_embeddings",
            [row]
        )

        if errors:
            print(f"Failed to insert {title}: {errors}")
        else:
            print(f"Inserted: {title}")

    except Exception as e:
        print(f"Error processing {title}: {e}")

cur.close()
conn.close()
```

# fetch_books_dag.py
```
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import csv
import time
import os
from google.cloud import storage

API_KEY = "AIzaSyAF9e-dplvn7hy3ObmK60XV-cpht4pMeeY"
GRADE_QUERIES = {
    "K": "kindergarten books",
    "1": "grade 1 books",
    "2": "grade 2 books",
    "3": "grade 3 books",
    "4": "grade 4 books",
    "5": "grade 5 books",
    "6": "grade 6 books",
    "7": "grade 7 books",
    "8": "grade 8 books"
}

BUCKET_NAME = "k8-books-bucket"
EXISTING_FILE = "books.csv"
NEW_FILE = "k8_books_new.csv"


def download_existing_books():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(EXISTING_FILE)
    blob.download_to_filename(EXISTING_FILE)
    print(f"Downloaded {EXISTING_FILE} from {BUCKET_NAME}")

def upload_new_books():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(NEW_FILE)
    blob.upload_from_filename(NEW_FILE)
    print(f"Uploaded {NEW_FILE} to {BUCKET_NAME}")

def fetch_books():
    download_existing_books()

    existing_identifiers = set()
    with open(EXISTING_FILE, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            title = row.get("title", "").strip().lower()
            isbn_10 = row.get("isbn_10", "").strip()
            isbn_13 = row.get("isbn_13", "").strip()
            if title:
                existing_identifiers.add(title)
            if isbn_10:
                existing_identifiers.add(isbn_10)
            if isbn_13:
                existing_identifiers.add(isbn_13)

    def fetch_books_for_grade(query, grade):
        book_data = []
        url = (
            f"https://www.googleapis.com/books/v1/volumes?"
            f"q={query}&maxResults=40&printType=books&langRestrict=en&key={API_KEY}"
        )
        response = requests.get(url)
        books = response.json().get("items", [])

        for book in books:
            info = book.get("volumeInfo", {})
            sale = book.get("saleInfo", {})
            access = book.get("accessInfo", {})

            title = info.get("title", "").strip()
            title_key = title.lower()

            isbn_10 = ""
            isbn_13 = ""
            for identifier in info.get("industryIdentifiers", []):
                if identifier["type"] == "ISBN_10":
                    isbn_10 = identifier["identifier"]
                elif identifier["type"] == "ISBN_13":
                    isbn_13 = identifier["identifier"]

            if (
                title_key in existing_identifiers or
                (isbn_10 and isbn_10 in existing_identifiers) or
                (isbn_13 and isbn_13 in existing_identifiers)
            ):
                continue

            authors = ", ".join(info.get("authors", []))
            publisher = info.get("publisher", "")
            published_date = info.get("publishedDate", "")
            description = info.get("description", "")
            page_count = info.get("pageCount", "")
            categories = ", ".join(info.get("categories", []))
            language = info.get("language", "")
            reading_mode_text = info.get("readingModes", {}).get("text", "")
            reading_mode_image = info.get("readingModes", {}).get("image", "")
            image_small = info.get("imageLinks", {}).get("smallThumbnail", "")
            image_large = info.get("imageLinks", {}).get("thumbnail", "")
            country = sale.get("country", "")
            list_price_amount = sale.get("listPrice", {}).get("amount", "")
            list_price_currency = sale.get("listPrice", {}).get("currencyCode", "")
            buy_link = sale.get("buyLink", "")
            web_reader_link = access.get("webReaderLink", "")
            embeddable = access.get("embeddable", "")

            book_data.append([
                title, authors, publisher, published_date, description,
                isbn_10, isbn_13, reading_mode_text, reading_mode_image,
                page_count, categories, maturity_rating, image_small,
                image_large, language, country, list_price_amount,
                list_price_currency, buy_link, web_reader_link,
                embeddable, grade
            ])

            existing_identifiers.update([title_key, isbn_10, isbn_13])

        return book_data

    all_new_books = []
    for grade, query in GRADE_QUERIES.items():
        print(f"Searching for Grade {grade} books...")
        all_new_books.extend(fetch_books_for_grade(query, grade))
        time.sleep(10)  # Throttle to avoid hitting API limits

    headers = [
        "title", "authors", "publisher", "published_date", "description",
        "isbn_10", "isbn_13", "reading_mode_text", "reading_mode_image",
        "page_count", "categories", "image_small",
        "image_large", "language", "sale_country", "list_price_amount",
        "list_price_currency", "buy_link", "web_reader_link", "embeddable", "grade"
    ]

    with open(NEW_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(all_new_books)

    print(f"Saved {len(all_new_books)} books to {NEW_FILE}")
    upload_new_books()


# Define the DAG
with DAG(
    "fetch_books_pipeline",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)}
) as dag:
    fetch_books_task = PythonOperator(
        task_id="fetch_books",
        python_callable=fetch_books
    )
```

# dp.py
```
import psycopg2
import os

def get_db_connection():
    return psycopg2.connect(
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=f"/cloudsql/{os.environ['INSTANCE_CONNECTION_NAME']}" 
    )
```

# Dockerfile
```
# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app (code, templates, etc.)
COPY . .

# Expose port
EXPOSE 8080

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

# clean_books_csv.py
```
import csv
from google.cloud import storage

BUCKET_NAME = "k8-books-bucket"
SOURCE_FILE = "k8_books_new.csv"
TARGET_FILE = "k8_books_clean.csv"

# Final expected 22 columns (after dropping 'maturity_rating')
COLUMNS = [
    "title", "authors", "publisher", "published_date", "description",
    "isbn_10", "isbn_13", "reading_mode_text", "reading_mode_image", "page_count",
    "categories", "image_small", "image_large", "language",
    "sale_country", "list_price_amount", "list_price_currency",
    "buy_link", "web_reader_link", "embeddable", "grade"
]

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# Download the raw file from GCS
blob = bucket.blob(SOURCE_FILE)
blob.download_to_filename("temp_books.csv")

# Clean and save locally
with open("temp_books.csv", "r", encoding="utf-8") as infile, open("k8_books_clean.csv", "w", encoding="utf-8", newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    writer.writerow(COLUMNS)  # Write header

    for i, row in enumerate(reader):
        # Skip original header
        if i == 0 and "title" in row[0].lower():
            continue
        
        # Remove ID column if present (assume first column is ID if row is too long)
        if len(row) == 24:
            row = row[1:]  # Remove ID
        if len(row) == 23:
            del row[11]  # Remove 'maturity_rating'

        # Only keep rows that now match our column list and have a title
        if len(row) == 22 and row[0].strip():
            writer.writerow(row)

# Upload cleaned file to GCS
clean_blob = bucket.blob(TARGET_FILE)
clean_blob.upload_from_filename("k8_books_clean.csv")

print(" Cleaned file uploaded as", TARGET_FILE)
```

# chatbot.py
```
# chat bot
import vertexai
from vertexai.language_models import ChatModel

# Initialize Vertex AI client
vertexai.init(
    project="book-recommendations-456120",
    location="us-central1"
)

def ask_chatbot(prompt: str) -> str:
    try:
        print("Sending prompt to Vertex AI:", prompt)
        chat_model = ChatModel.from_pretrained("chat-bison@002")
        chat = chat_model.start_chat(
            context="You are a friendly book assistant. Recommend books based on what the user says. Use genres, topics, title, authors, and grade level if given.",
        )
        response = chat.send_message(prompt)
        print("Vertex AI responded with:", response.text)
        return response.text
    except Exception as e:
        print("Vertex AI chatbot error:", e)
        return "Sorry, I had trouble answering that."
```

# requirements.txt for book-api
```
annotated-types==0.7.0
anyio==4.9.0
bcrypt==4.3.0
blinker==1.9.0
cachetools==5.5.2
certifi==2025.1.31
charset-normalizer==3.4.1
click==8.1.8
Deprecated==1.2.18
docstring_parser==0.16
ecdsa==0.19.1
fastapi==0.115.12
Flask==3.1.0
Flask-SQLAlchemy==3.1.1
google-api-core==2.24.2
google-auth==2.39.0
google-cloud-aiplatform==1.89.0
google-cloud-bigquery==3.31.0
google-cloud-core==2.4.3
google-cloud-pubsub==2.29.0
google-cloud-resource-manager==1.14.2
google-cloud-storage==2.19.0
google-crc32c==1.7.1
google-resumable-media==2.7.2
googleapis-common-protos==1.70.0
greenlet==3.1.1
grpc-google-iam-v1==0.14.2
grpcio==1.71.0
grpcio-status==1.71.0
h11==0.14.0
idna==3.10
importlib_metadata==8.6.1
itsdangerous==2.2.0
Jinja2==3.1.6
MarkupSafe==3.0.2
numpy==2.2.4
opentelemetry-api==1.32.1
opentelemetry-sdk==1.32.1
opentelemetry-semantic-conventions==0.53b1
packaging==24.2
passlib==1.7.4
proto-plus==1.26.1
protobuf==5.29.4
psycopg2-binary==2.9.10
pyasn1==0.6.1
pyasn1_modules==0.4.2
pydantic==2.11.3
pydantic_core==2.33.1
python-dateutil==2.9.0.post0
python-dotenv==1.1.0
python-jose==3.3.0
python-multipart==0.0.20
requests==2.32.3
rsa==4.9.1
shapely==2.1.0
six==1.17.0
sniffio==1.3.1
SQLAlchemy==2.0.40
starlette==0.46.1
typing-inspection==0.4.0
typing_extensions==4.13.1
urllib3==2.4.0
uvicorn==0.34.0
Werkzeug==3.1.3
wrapt==1.17.2
zipp==3.21.0
google-cloud-sql-connector[pg8000]
```

# book_embeddings_schema.json
```
[
    {"name": "isbn_10", "type": "STRING"},
    {"name": "title", "type": "STRING"},
    {"name": "authors", "type": "STRING"},
    {"name": "description", "type": "STRING"},
    {"name": "grade", "type": "STRING"},
    {"name": "embedding", "type": "FLOAT64", "mode": "REPEATED"},
    {"name": "thumbs_up", "type": "INTEGER"},
    {"name": "thumbs_down", "type": "INTEGER"},
    {"name": "image_small", "type": "STRING"},
    {"name": "buy_link", "type": "STRING"},
    {"name": "web_reader_link", "type": "STRING"}
  ]
```


# redeploy checklist
```
docker build -t gcr.io/book-recommendations-456120/book-api .
docker push gcr.io/book-recommendations-456120/book-api

gcloud run deploy book-api \
  --image gcr.io/book-recommendations-456120/book-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --add-cloudsql-instances book-recommendations-456120:us-central1:book-recs-db \
  --set-env-vars DB_NAME=book_recommendations_db,DB_USER=postgres,DB_PASSWORD=pass,INSTANCE_CONNECTION_NAME=book-recommendations-456120:us-central1:book-recs-db \
  --port 8080
```

# append_books_from_gcs.py
```
import os
import csv
from google.cloud import storage
from google.cloud.sql.connector import Connector
import sqlalchemy
from flask import jsonify

# Environment variables
DB_NAME = os.environ['DB_NAME']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']
INSTANCE_CONNECTION_NAME = os.environ['INSTANCE_CONNECTION_NAME']

BUCKET_NAME = 'k8-books-bucket'
FILE_NAME = 'k8_books_clean.csv'

# Cloud SQL connection function
def getconn():
    connector = Connector()
    conn = connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pg8000",
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
    )
    return conn

# SQLAlchemy engine using connector
engine = sqlalchemy.create_engine(
    "postgresql+pg8000://",
    creator=getconn,
)

# Cloud Function entry point
def append_books(request):
    try:
        # Download CSV file from GCS to /tmp
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(FILE_NAME)
        blob.download_to_filename('/tmp/books.csv')

        new_count = 0
        with engine.connect() as conn:
            with open('/tmp/books.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    result = conn.execute(
                        sqlalchemy.text("""
                            SELECT 1 FROM books WHERE isbn_10 = :isbn_10 OR isbn_13 = :isbn_13 OR LOWER(title) = LOWER(:title)
                        """),
                        {"isbn_10": row['isbn_10'], "isbn_13": row['isbn_13'], "title": row['title']}
                    ).fetchone()

                    if not result:
                        conn.execute(
                            sqlalchemy.text("""
                                INSERT INTO books (
                                    title, authors, publisher, published_date, description,
                                    isbn_10, isbn_13, reading_mode_text, reading_mode_image,
                                    page_count, categories, image_small, image_large, language,
                                    sale_country, list_price_amount, list_price_currency,
                                    buy_link, web_reader_link, embeddable, grade
                                ) VALUES (
                                    :title, :authors, :publisher, :published_date, :description,
                                    :isbn_10, :isbn_13, :reading_mode_text, :reading_mode_image,
                                    :page_count, :categories, :image_small, :image_large, :language,
                                    :sale_country, :list_price_amount, :list_price_currency,
                                    :buy_link, :web_reader_link, :embeddable, :grade
                                )
                            """),
                            row
                        )
                        new_count += 1

            conn.commit()

        return jsonify({"message": f"{new_count} new books added."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

# main.py in book-api
```
from append_books_from_gcs import append_books
```

## rate book function

# requirements.txt in rate_book_function
```
psycopg2-binary
google-cloud-storage
google-cloud-pubsub
google-cloud-aiplatform
```

# main.py in rate_book_function
```
import base64
import json
import os
from google.cloud import storage
import psycopg2

DB_USER = "postgres"
DB_PASS = "pass"
DB_NAME = "book_recommendations_db"
DB_HOST = "/cloudsql/book-recommendations-456120:us-central1:book-recs-db"

def get_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST
    )

def entry_point(event, context):
    data = json.loads(base64.b64decode(event["data"]).decode("utf-8"))

    isbn_10 = data.get("isbn_10")
    title   = data["title"]
    author  = data["author"]
    user_id = data["user_id"]
    rating  = data.get("rating")
    status  = data.get("status")
    progress = data.get("progress")
    notes   = data.get("notes")

    conn = get_conn()
    cur = conn.cursor()

    try:
        # 1. Insert book if it doesn't exist
        cur.execute("""
            SELECT 1 FROM books WHERE isbn_10 = %s
        """, (isbn_10,))
        if not cur.fetchone():
            cur.execute("""
                INSERT INTO books (title, authors, isbn_10)
                VALUES (%s, %s, %s)
            """, (title, author, isbn_10))
            print(f" Added new book: {title}")

        # 2. Insert into user_books
        cur.execute("""
            INSERT INTO user_books (user_id, title, author, isbn_10, rating, status, progress, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (user_id, title, author, isbn_10, rating, status, progress, notes))
        print(f" Added rating: {rating} for user {user_id}")

        conn.commit()

    except Exception as e:
        conn.rollback()
        print(f" Error: {e}")
    finally:
        cur.close()
        conn.close()
```

## BigQuery SQL Query 
```
DECLARE target_isbn STRING DEFAULT "0375866116";
DECLARE target_title STRING DEFAULT NULL;
DECLARE target_authors STRING DEFAULT NULL;
DECLARE target_grade STRING DEFAULT NULL;
DECLARE target_keyword STRING DEFAULT NULL;

WITH target AS (
  SELECT embedding
  FROM `book-recommendations-456120.book_recs.book_embeddings`
  WHERE (
    (target_isbn IS NOT NULL AND isbn_10 = target_isbn) OR
    (target_title IS NOT NULL AND LOWER(title) = LOWER(target_title)) OR
    (target_authors IS NOT NULL AND LOWER(authors) = LOWER(target_authors)) OR
    (target_grade IS NOT NULL AND grade = target_grade) OR
    (target_keyword IS NOT NULL AND (
      LOWER(title) LIKE CONCAT('%', LOWER(target_keyword), '%') OR
      LOWER(authors) LIKE CONCAT('%', LOWER(target_keyword), '%') OR
      LOWER(description) LIKE CONCAT('%', LOWER(target_keyword), '%')
    ))
  )
  LIMIT 1
),

scored AS (
  SELECT
    b2.isbn_10,
    b2.title,
    b2.authors,
    b2.grade,
    b2.image_small,
    b2.web_reader_link,
    b2.thumbs_up,
    b2.thumbs_down,
    (
      SELECT SUM(e1 * e2)
      FROM UNNEST(b2.embedding) AS e1 WITH OFFSET i
      JOIN UNNEST(target.embedding) AS e2 WITH OFFSET j
      ON i = j
    ) /
    (
      SQRT((SELECT SUM(POW(val, 2)) FROM UNNEST(b2.embedding) AS val)) *
      SQRT((SELECT SUM(POW(val2, 2)) FROM UNNEST(target.embedding) AS val2))
    ) AS similarity
  FROM `book-recommendations-456120.book_recs.book_embeddings` b2, target
  WHERE b2.embedding IS NOT NULL
    AND (
      target_isbn IS NULL OR b2.isbn_10 != target_isbn
    )
)

SELECT *
FROM scored
ORDER BY similarity DESC
LIMIT 10;
```


## Google Books API Colab
# Create books.csv
```
import requests
import csv
import time

API_KEY = "key"

GRADE_QUERIES = {
    "K": "kindergarten books",
    "1": "grade 1 books",
    "2": "grade 2 books",
    "3": "grade 3 books",
    "4": "grade 4 books",
    "5": "grade 5 books",
    "6": "grade 6 books",
    "7": "grade 7 books",
    "8": "grade 8 books"
}

def fetch_books(query, grade, max_results=40):
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}&printType=books&langRestrict=en&key={API_KEY}"
    response = requests.get(url)
    books = response.json().get("items", [])
    book_data = []

    for book in books:
        info = book.get("volumeInfo", {})
        sale = book.get("saleInfo", {})
        access = book.get("accessInfo", {})

        title = info.get("title", "")
        authors = ", ".join(info.get("authors", []))
        publisher = info.get("publisher", "")
        published_date = info.get("publishedDate", "")
        description = info.get("description", "")
        page_count = info.get("pageCount", "")
        categories = ", ".join(info.get("categories", []))
        maturity_rating = info.get("maturityRating", "")
        language = info.get("language", "")
        reading_mode_text = info.get("readingModes", {}).get("text", "")
        reading_mode_image = info.get("readingModes", {}).get("image", "")

        image_small = info.get("imageLinks", {}).get("smallThumbnail", "")
        image_large = info.get("imageLinks", {}).get("thumbnail", "")

        # ISBNs
        isbn_10 = ""
        isbn_13 = ""
        for identifier in info.get("industryIdentifiers", []):
            if identifier["type"] == "ISBN_10":
                isbn_10 = identifier["identifier"]
            elif identifier["type"] == "ISBN_13":
                isbn_13 = identifier["identifier"]

        # Sale Info
        country = sale.get("country", "")
        list_price_amount = sale.get("listPrice", {}).get("amount", "")
        list_price_currency = sale.get("listPrice", {}).get("currencyCode", "")
        buy_link = sale.get("buyLink", "")

        # Access Info
        web_reader_link = access.get("webReaderLink", "")
        embeddable = access.get("embeddable", "")

        book_data.append([
            title, authors, publisher, published_date, description,
            isbn_10, isbn_13,
            reading_mode_text, reading_mode_image, page_count,
            categories, maturity_rating,
            image_small, image_large,
            language, country,
            list_price_amount, list_price_currency,
            buy_link, web_reader_link, embeddable,
            grade
        ])

    return book_data

def save_to_csv(data, filename="k8_books_40.csv"):
    headers = [
        "Title", "Authors", "Publisher", "Published Date", "Description",
        "ISBN_10", "ISBN_13",
        "ReadingMode_Text", "ReadingMode_Image", "PageCount",
        "Categories", "MaturityRating",
        "Image_Small", "Image_Large",
        "Language", "Sale_Country",
        "ListPrice_Amount", "ListPrice_Currency",
        "BuyLink", "WebReaderLink", "Embeddable",
        "Grade"
    ]
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

# Run the scraping process
all_books = []
for grade, query in GRADE_QUERIES.items():
    print(f"📚 Fetching books for Grade {grade}...")
    books = fetch_books(query, grade)
    all_books.extend(books)
    time.sleep(100)  # To avoid rate limits

save_to_csv(all_books)
print(" CSV saved as k8_books_40.csv")

```

# make books.csv bigger and bypass query limits
```
import requests
import csv
import time
import os

API_KEY = "key"

GRADE_QUERIES = {
    "K": "kindergarten books",
    "1": "grade 1 books",
    "2": "grade 2 books",
    "3": "grade 3 books",
    "4": "grade 4 books",
    "5": "grade 5 books",
    "6": "grade 6 books",
    "7": "grade 7 books",
    "8": "grade 8 books"
}

EXISTING_FILE = "books.csv"
NEW_FILE = "k8_books_new.csv"

# Step 1: Load existing identifiers (title + ISBNs)
existing_identifiers = set()
if os.path.exists(EXISTING_FILE):
    with open(EXISTING_FILE, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            title = row.get("title", "").strip().lower()
            isbn_10 = row.get("isbn_10", "").strip()
            isbn_13 = row.get("isbn_13", "").strip()
            if title:
                existing_identifiers.add(title)
            if isbn_10:
                existing_identifiers.add(isbn_10)
            if isbn_13:
                existing_identifiers.add(isbn_13)

def fetch_books(query, grade, start_index=0, max_results=40):
    url = (
        f"https://www.googleapis.com/books/v1/volumes?"
        f"q={query}&startIndex={start_index}&maxResults={max_results}&printType=books"
        f"&langRestrict=en&key={API_KEY}"
    )
    response = requests.get(url)
    books = response.json().get("items", [])
    book_data = []

    for book in books:
        info = book.get("volumeInfo", {})
        sale = book.get("saleInfo", {})
        access = book.get("accessInfo", {})

        title = info.get("title", "").strip()
        title_key = title.lower()

        isbn_10 = ""
        isbn_13 = ""
        for identifier in info.get("industryIdentifiers", []):
            if identifier["type"] == "ISBN_10":
                isbn_10 = identifier["identifier"]
            elif identifier["type"] == "ISBN_13":
                isbn_13 = identifier["identifier"]

        # Duplicate check
        if (
            title_key in existing_identifiers or
            (isbn_10 and isbn_10 in existing_identifiers) or
            (isbn_13 and isbn_13 in existing_identifiers)
        ):
            continue

        authors = ", ".join(info.get("authors", []))
        publisher = info.get("publisher", "")
        published_date = info.get("publishedDate", "")
        description = info.get("description", "")
        page_count = info.get("pageCount", "")
        categories = ", ".join(info.get("categories", []))
        maturity_rating = info.get("maturityRating", "")
        language = info.get("language", "")
        reading_mode_text = info.get("readingModes", {}).get("text", "")
        reading_mode_image = info.get("readingModes", {}).get("image", "")
        image_small = info.get("imageLinks", {}).get("smallThumbnail", "")
        image_large = info.get("imageLinks", {}).get("thumbnail", "")
        country = sale.get("country", "")
        list_price_amount = sale.get("listPrice", {}).get("amount", "")
        list_price_currency = sale.get("listPrice", {}).get("currencyCode", "")
        buy_link = sale.get("buyLink", "")
        web_reader_link = access.get("webReaderLink", "")
        embeddable = access.get("embeddable", "")

        book_data.append([
            title, authors, publisher, published_date, description,
            isbn_10, isbn_13,
            reading_mode_text, reading_mode_image, page_count,
            categories, maturity_rating,
            image_small, image_large,
            language, country,
            list_price_amount, list_price_currency,
            buy_link, web_reader_link, embeddable,
            grade
        ])

        # Add identifiers to prevent future duplicates
        if title_key:
            existing_identifiers.add(title_key)
        if isbn_10:
            existing_identifiers.add(isbn_10)
        if isbn_13:
            existing_identifiers.add(isbn_13)

    return book_data

def save_to_csv(data, filename):
    headers = [
        "title", "authors", "publisher", "published_date", "description",
        "isbn_10", "isbn_13",
        "reading_mode_text", "reading_mode_image", "page_count",
        "categories", "maturity_rating",
        "image_small", "image_large",
        "language", "sale_country",
        "list_price_amount", "list_price_currency",
        "buy_link", "web_reader_link", "embeddable",
        "grade"
    ]
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

# Step 2: Fetch books per grade level
all_new_books = []
for grade, query in GRADE_QUERIES.items():
    print(f"Searching for more Grade {grade} books...")
    all_new_books.extend(fetch_books(query, grade))
    time.sleep(60)  # Respect API rate limits

save_to_csv(all_new_books, NEW_FILE)
print(f"New books saved to {NEW_FILE}")

```
