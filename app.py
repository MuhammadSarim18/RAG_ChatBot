import os
import sys
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import numpy as np
import requests
import psycopg2
from psycopg2 import pool
from pgvector.psycopg2 import register_vector
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq, APIError

# Configuration
DB_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag_kb")
MODEL_NAME = "all-mpnet-base-v2"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
SIMILARITY_THRESHOLD = 0.3
TOP_K = 5

app_state: Dict[str, Any] = {
    "db_pool": None,
    "model": None,
    "groq_client": None
}


def get_db_pool() -> pool.ThreadedConnectionPool:
    """Lazy-initializes the PostgreSQL connection pool."""
    if app_state["db_pool"] is None:
        print(f"Connecting to Postgres at {DB_URL}...")
        try:
            # Set connect_timeout so it never hangs indefinitely
            db_pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=DB_URL,
                connect_timeout=10
            )
            test_conn = db_pool.getconn()
            register_vector(test_conn)
            db_pool.putconn(test_conn)
            app_state["db_pool"] = db_pool
            print("Database connection pool initialized.")
        except Exception as e:
            print(f"Failed to connect to database: {e}", file=sys.stderr)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database connection error: {str(e)}"
            )
    return app_state["db_pool"]


def get_embedding(text: str) -> np.ndarray:
    """
    Computes 768-dim embedding for text.
    Uses Hugging Face Serverless API if HF_TOKEN is provided (recommended on Render free tier to use 0 MB RAM),
    otherwise falls back to local SentenceTransformer.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            api_url = f"https://router.huggingface.co/hf-inference/models/sentence-transformers/{MODEL_NAME}"
            headers = {"Authorization": f"Bearer {hf_token}"}
            resp = requests.post(api_url, headers=headers, json={"inputs": text}, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                # Handle pipeline response format
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], list):
                        data = data[0]
                    return np.array(data, dtype=np.float32)
            else:
                print(f"HF API returned {resp.status_code}: {resp.text}, falling back to local model.")
        except Exception as e:
            print(f"HF API call failed ({e}), falling back to local model.")

    # Local fallback
    if app_state["model"] is None:
        print(f"Loading local SentenceTransformer model '{MODEL_NAME}'...")
        from sentence_transformers import SentenceTransformer
        app_state["model"] = SentenceTransformer(MODEL_NAME)
        print("Local model loaded.")

    return app_state["model"].encode(text, convert_to_numpy=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Non-blocking lifespan manager.
    Allows uvicorn to bind the port immediately so Render detects the service as live instantly.
    """
    print("Application starting up... Port will open immediately.")

    # Initialize Groq client
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        app_state["groq_client"] = Groq(api_key=groq_api_key)
        print("Groq client initialized.")
    else:
        print("WARNING: GROQ_API_KEY environment variable is not set.")

    yield

    # Cleanup resources on shutdown
    print("Shutting down resources...")
    if app_state["db_pool"]:
        app_state["db_pool"].closeall()
        print("Database connection pool closed.")


# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot Backend",
    description="FastAPI backend with pgvector cosine similarity retrieval and Groq LLM generation",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "RAG Chatbot API is running!",
        "interactive_docs": "/docs",
        "endpoint": "POST /api/chat"
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


# Request & Response schemas
class ChatRequest(BaseModel):
    text: str = Field(..., description="User query / question")


class SourceItem(BaseModel):
    url: str
    title: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


def query_nearest_chunks(query_text: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Embeds query text and retrieves top_k nearest chunks using pgvector's <=> operator.
    Calculates similarity as (1 - distance).
    """
    query_vector = get_embedding(query_text)
    db_pool = get_db_pool()

    conn = db_pool.getconn()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            sql = """
                SELECT source_url, title, content, (embedding <=> %s) AS distance
                FROM chunks
                ORDER BY distance ASC
                LIMIT %s;
            """
            cur.execute(sql, (query_vector, top_k))
            rows = cur.fetchall()

            results = []
            for row in rows:
                source_url, title, content, distance = row
                similarity = 1.0 - float(distance)
                results.append({
                    "url": source_url or "",
                    "title": title or "",
                    "content": content or "",
                    "distance": float(distance),
                    "similarity": similarity
                })
            return results
    finally:
        db_pool.putconn(conn)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_query = request.text.strip()
    if not user_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'text' field cannot be empty."
        )

    # 1. Retrieve 5 nearest chunks
    chunks = query_nearest_chunks(user_query, top_k=TOP_K)

    # 2. Check best similarity score
    best_score = max((chunk["similarity"] for chunk in chunks), default=0.0)

    # 3. Fallback if best score < 0.3
    if not chunks or best_score < SIMILARITY_THRESHOLD:
        return ChatResponse(
            answer="I'm sorry, but there isn't enough relevant information in the knowledge base to answer your question.",
            sources=[]
        )

    # 4. Construct numbered context for prompt
    context_blocks = []
    sources: List[SourceItem] = []

    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[{i}] Title: {chunk['title']}\nSource URL: {chunk['url']}\nContent: {chunk['content']}"
        )
        sources.append(SourceItem(url=chunk["url"], title=chunk["title"]))

    joined_context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an accurate, helpful assistant for Times Consultant study abroad services.\n"
        "Guidelines:\n"
        "1. Answer the question using ONLY the provided numbered context below.\n"
        "2. Cite your sources inline using [1], [2], etc., corresponding to the context item number.\n"
        "3. If the context does not contain the necessary information, state that there is not enough information to answer."
    )

    user_prompt = (
        f"Context:\n{joined_context}\n\n"
        f"Question: {user_query}\n\n"
        f"Answer:"
    )

    # 5. Call Groq API
    client: Optional[Groq] = app_state.get("groq_client")
    if client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GROQ_API_KEY environment variable is not set. Please set your GROQ_API_KEY in Render settings."
            )
        client = Groq(api_key=api_key)
        app_state["groq_client"] = client

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        answer_text = response.choices[0].message.content
    except APIError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Groq API error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error calling Groq LLM: {str(e)}"
        )

    return ChatResponse(
        answer=answer_text,
        sources=sources
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
