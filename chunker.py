import json
import os
import re
from typing import List, Dict, Any


def estimate_tokens(text: str) -> int:
    """
    Approximates tokens as word count divided by 0.75.
    (i.e. tokens ≈ words / 0.75)
    """
    words = len(text.split())
    if words == 0:
        return 0
    return round(words / 0.75)


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences on sentence boundaries (. ! ?) and newlines,
    ensuring sentences are kept intact and no sentence ends prematurely.
    """
    sentences = []
    # Split by lines first to preserve headings and paragraph structure
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split on punctuation (. ! ?) followed by whitespace, keeping delimiters attached
        parts = re.split(r'(?<=[.!?])\s+', line)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def chunk_page_text(text: str, target_tokens: int = 500, overlap_tokens: int = 80) -> List[str]:
    """
    Chunks text into medium-sized segments:
    - Target: ~500 tokens per chunk.
    - Overlap: ~80 tokens carried into the subsequent chunk.
    - Preserves sentence boundaries so no chunk cuts off mid-sentence.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    sentence_token_counts = [max(1, estimate_tokens(s)) for s in sentences]
    total_sentences = len(sentences)

    chunks = []
    i = 0

    while i < total_sentences:
        current_chunk_sentences = []
        current_tokens = 0
        start_idx = i

        # Accumulate sentences until reaching target_tokens
        while i < total_sentences:
            sentence = sentences[i]
            tokens = sentence_token_counts[i]

            # If adding the next sentence would overshoot and chunk already has content, break
            if current_chunk_sentences and (current_tokens + tokens > target_tokens):
                break

            current_chunk_sentences.append(sentence)
            current_tokens += tokens
            i += 1

        if current_chunk_sentences:
            chunk_content = " ".join(current_chunk_sentences)
            chunks.append(chunk_content)

        # Reached the end of text
        if i >= total_sentences:
            break

        # Calculate overlap carried into next chunk (~80 tokens)
        overlap_accum = 0
        overlap_idx = i - 1

        while overlap_idx > start_idx:
            prev_tokens = sentence_token_counts[overlap_idx]
            if overlap_accum + prev_tokens > overlap_tokens and overlap_accum > 0:
                break
            overlap_accum += prev_tokens
            overlap_idx -= 1

        # Advance start of next chunk, ensuring forward progress of at least 1 sentence
        i = max(start_idx + 1, overlap_idx + 1)

    return chunks


def create_chunks_from_scraped_data(
    input_file: str = "scraped.json",
    output_file: str = "chunks.json",
    target_tokens: int = 500,
    overlap_tokens: int = 80
) -> List[Dict[str, Any]]:
    """
    Loads scraped pages from JSON, splits each page's text into ~500 token chunks
    with ~80 tokens overlap split strictly on sentence boundaries, and saves the
    result as chunks.json (list of {url, title, content}).
    """
    # Fallback to scraper.json if scraped.json is not found
    if not os.path.exists(input_file) and os.path.exists("scraper.json"):
        input_file = "scraper.json"

    with open(input_file, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []
    for page in pages:
        url = page.get("url", "")
        title = page.get("title", "")
        text = page.get("text", "")

        page_chunks = chunk_page_text(text, target_tokens=target_tokens, overlap_tokens=overlap_tokens)

        for content in page_chunks:
            all_chunks.append({
                "url": url,
                "title": title,
                "content": content
            })

    # Save output to chunks.json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(pages)} pages into {len(all_chunks)} chunks.")
    print(f"Saved chunks to '{output_file}'.")

    return all_chunks


if __name__ == "__main__":
    create_chunks_from_scraped_data()
