import argparse
import csv
import json
import os
import sys
import time
from typing import List, Dict, Any
import requests

DEFAULT_ENDPOINT = "http://localhost:8000/api/chat"
DEFAULT_OUTPUT_CSV = "evaluation_results.csv"

# Default test questions representing a mix of domain-specific topics, edge cases, and out-of-domain queries
DEFAULT_QUESTIONS = [
    # In-domain: Visa & Study Destinations
    "What are the requirements for a German student visa from Pakistan?",
    "How much does it cost to study in Hungary and what are the living expenses?",
    "Can international students work part-time while studying in the UK?",
    "What services does Times Consultant provide for student accommodation?",
    "What language tests does Times Consultant offer preparation for?",
    "What are the steps involved in the university application and admission process?",
    "How early should I apply before an intake starts?",
    "What documents are needed for study abroad applications?",
    "Tell me about upcoming Times Consultant study abroad expo events.",
    "What is the difference between a visit visa and a student visa?",
    
    # Out-of-domain: Should trigger low similarity fallback (< 0.3 threshold)
    "How do I bake chocolate chip cookies?",
    "Explain how quantum computers use qubits for superposition.",
    "What is the capital of Mars and who is its president?"
]


def load_questions(file_path: str = None) -> List[str]:
    """Loads questions from a text or JSON file if provided, otherwise returns defaults."""
    if not file_path:
        return DEFAULT_QUESTIONS

    if not os.path.exists(file_path):
        print(f"Warning: File '{file_path}' not found. Using default question set.")
        return DEFAULT_QUESTIONS

    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [q if isinstance(q, str) else q.get("question", "") for q in data]
            return DEFAULT_QUESTIONS
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            return lines if lines else DEFAULT_QUESTIONS


def format_sources(sources: List[Dict[str, str]]) -> str:
    """Formats the sources list into a readable string for a CSV cell."""
    if not sources:
        return "None"
    
    formatted = []
    for i, s in enumerate(sources, 1):
        title = s.get("title", "").strip() or "Untitled"
        url = s.get("url", "").strip()
        formatted.append(f"[{i}] {title}\n    {url}")
    return "\n\n".join(formatted)


def evaluate_and_export(
    endpoint: str = DEFAULT_ENDPOINT,
    output_file: str = DEFAULT_OUTPUT_CSV,
    questions_file: str = None,
    delay: float = 0.5
):
    """
    Sends each question to /api/chat and writes results to CSV with grading columns.
    """
    questions = load_questions(questions_file)
    print(f"Loaded {len(questions)} test questions.")
    print(f"Target endpoint: {endpoint}")
    print(f"Output CSV: {output_file}\n")

    results = []

    for idx, question in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] Sending: '{question}'...")
        
        start_time = time.time()
        try:
            resp = requests.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json={"text": question},
                timeout=60
            )
            elapsed = round(time.time() - start_time, 2)

            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                sources = data.get("sources", [])
                status_note = f"OK ({elapsed}s)"
                print(f"       -> Received answer with {len(sources)} sources ({elapsed}s)")
            else:
                answer = f"ERROR: HTTP {resp.status_code} - {resp.text}"
                sources = []
                status_note = f"HTTP {resp.status_code}"
                print(f"       -> Error {resp.status_code}: {resp.text}")

        except requests.exceptions.RequestException as e:
            answer = f"REQUEST FAILED: {str(e)}"
            sources = []
            status_note = "Network / Timeout Error"
            print(f"       -> Request exception: {e}")

        results.append({
            "id": idx,
            "question": question,
            "answer": answer,
            "sources_count": len(sources),
            "sources": format_sources(sources),
            "grade_accuracy (1-5)": "",     # Empty for manual grading
            "grade_citation (1-5)": "",     # Empty for manual grading
            "manual_notes": "",            # Empty for manual notes
            "api_status": status_note
        })

        time.sleep(delay)

    # Write results to CSV
    fieldnames = [
        "id",
        "question",
        "answer",
        "sources_count",
        "sources",
        "grade_accuracy (1-5)",
        "grade_citation (1-5)",
        "manual_notes",
        "api_status"
    ]

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCompleted! Successfully wrote {len(results)} rows to '{output_file}'.")
    print(f"You can now open '{output_file}' in Excel / Google Sheets to grade each row.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG chat API with test questions and export CSV for grading.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Chat endpoint URL (default: http://localhost:8000/api/chat)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="Output CSV path (default: evaluation_results.csv)")
    parser.add_argument("--questions-file", default=None, help="Optional text file with questions (1 per line) or JSON")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay in seconds between requests")

    args = parser.parse_args()
    evaluate_and_export(
        endpoint=args.endpoint,
        output_file=args.output,
        questions_file=args.questions_file,
        delay=args.delay
    )

