-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table to store document chunks and embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    source_url TEXT,
    title TEXT,
    content TEXT,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
