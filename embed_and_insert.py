import argparse
import json
import os
import sys
from typing import List, Dict, Any
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DEFAULT_DB_URL = "postgresql://rag:rag@localhost:5432/rag_kb"
DEFAULT_MODEL_NAME = "all-mpnet-base-v2"
DEFAULT_CHUNKS_FILE = "chunks.json"
DEFAULT_BATCH_SIZE = 32


def load_chunks(file_path: str) -> List[Dict[str, Any]]:
    """Loads chunks from the specified JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Chunks file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from '{file_path}'.")
    return chunks


def embed_and_insert_chunks(
    chunks_file: str = DEFAULT_CHUNKS_FILE,
    db_url: str = DEFAULT_DB_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """
    Loads chunks, computes 768-dim embeddings in batches using sentence-transformers,
    bulk-inserts them into Postgres via psycopg2 and pgvector, and creates an HNSW index.
    """
    # 1. Load data
    chunks = load_chunks(chunks_file)
    if not chunks:
        print("No chunks to process.")
        return

    # 2. Connect to Postgres and register pgvector
    print(f"Connecting to database at {db_url}...")
    conn = psycopg2.connect(db_url)
    register_vector(conn)
    cur = conn.cursor()

    try:
        # 3. Load embedding model
        print(f"Loading sentence-transformers model '{model_name}'...")
        model = SentenceTransformer(model_name)
        embedding_dim = model.get_sentence_embedding_dimension()
        print(f"Model loaded successfully. Embedding dimension: {embedding_dim}")

        insert_sql = """
            INSERT INTO chunks (source_url, title, content, embedding)
            VALUES %s
        """

        total_chunks = len(chunks)
        print(f"Processing and inserting {total_chunks} chunks in batches of {batch_size}...")

        # 4. Process in batches
        for start_idx in range(0, total_chunks, batch_size):
            end_idx = min(start_idx + batch_size, total_chunks)
            batch = chunks[start_idx:end_idx]

            # Extract contents to embed
            contents = [item.get("content", "") for item in batch]

            # Compute embeddings
            embeddings = model.encode(
                contents,
                batch_size=len(contents),
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Build row tuples: (source_url, title, content, embedding)
            records = []
            for item, emb in zip(batch, embeddings):
                source_url = item.get("source_url") or item.get("url", "")
                title = item.get("title", "")
                content = item.get("content", "")
                records.append((source_url, title, content, emb))

            # Bulk insert
            execute_values(cur, insert_sql, records, template="(%s, %s, %s, %s)")
            conn.commit()

            print(f"Inserted chunks {start_idx + 1} to {end_idx} of {total_chunks}.")

        print("All chunks successfully inserted.")

        # 5. Create HNSW index using vector_cosine_ops if not exists
        print("Creating HNSW index on embedding column (vector_cosine_ops) if not exists...")
        hnsw_index_sql = """
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
            ON chunks USING hnsw (embedding vector_cosine_ops);
        """
        cur.execute(hnsw_index_sql)
        conn.commit()
        print("HNSW index created / verified successfully.")

    except Exception as e:
        conn.rollback()
        print(f"Error occurred during embed and insert: {e}", file=sys.stderr)
        raise
    finally:
        cur.close()
        conn.close()
        print("Database connection closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks with sentence-transformers and insert into Postgres pgvector.")
    parser.add_argument("--chunks-file", default=DEFAULT_CHUNKS_FILE, help="Path to chunks.json")
    parser.add_argument("--db-url", default=DEFAULT_DB_URL, help="PostgreSQL connection string")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for embedding and insertion")

    args = parser.parse_args()
    embed_and_insert_chunks(
        chunks_file=args.chunks_file,
        db_url=args.db_url,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

