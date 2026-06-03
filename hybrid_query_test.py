import json
import os
import sys
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from hybrid_search import build_chunk_index, load_chunk_index, hybrid_search


DATA_DIR = Path("data")
MATCH_INDEX_PATH = Path("rag_index_demo.faiss")
MATCH_METADATA_PATH = Path("rag_metadata_demo.json")
# chunked over-level index
CHUNK_INDEX_PATH = Path("rag_chunk_index.faiss")
CHUNK_METADATA_PATH = Path("rag_chunk_metadata.json")
OUTPUT_PATH = Path("hybrid_query_results.jsonl")
MODEL_ID = "EleutherAI/gpt-neo-125M"


def load_or_build_rag(data_dir: Path, index_path: Path, metadata_path: Path):
    if index_path.exists() and metadata_path.exists():
        print(f"Loading existing RAG index from {index_path}")
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, pd.DataFrame(metadata)

    print("Building new RAG index from data...")
    files = sorted(data_dir.glob("*.json"))
    rows = []
    summaries = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            match = json.load(f)
        info = match.get("info", {})
        teams = info.get("teams", [])
        venue = info.get("venue", "Unknown")
        winner = info.get("outcome", {}).get("winner", "Unknown")
        date = info.get("dates", ["Unknown"])[0]
        row = {
            "file_path": str(path),
            "team1": teams[0] if len(teams) > 0 else "Unknown",
            "team2": teams[1] if len(teams) > 1 else "Unknown",
            "venue": venue,
            "winner": winner,
            "date": date,
        }
        rows.append(row)
        summaries.append(f"{row['team1']} vs {row['team2']} at {venue} on {date}, winner {winner}.")

    df = pd.DataFrame(rows)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(summaries, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return index, df


def route_query(query: str) -> str:
    text = query.lower()
    story_keywords = [
        "describe",
        "story",
        "summar",
        "narrat",
        "highlight",
        "tense",
        "dramatic",
        "final over",
        "last over",
        "moment",
        "memorable",
        "how did",
        "what happened",
    ]
    stats_keywords = [
        "who",
        "when",
        "won",
        "score",
        "team",
        "player",
        "venue",
        "date",
        "runs",
        "how many",
        "captain",
        "win",
        "loss",
        "toss",
    ]
    if any(keyword in text for keyword in story_keywords):
        return "STORY"
    if any(keyword in text for keyword in stats_keywords):
        return "STATS"
    return "STATS"


def query_stats(df: pd.DataFrame, query: str) -> str:
    stopwords = {
        "who",
        "what",
        "where",
        "when",
        "how",
        "which",
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "between",
        "match",
        "team",
        "teams",
        "was",
        "were",
        "did",
        "by",
        "for",
        "to",
        "is",
        "are",
        "will",
        "won",
        "lost",
        "played",
        "score",
        "describe",
        "what",
        "who",
        "how",
        "did",
        "was",
        "were",
    }
    query_tokens = [
        token
        for token in re.findall(r"\w+", query.lower())
        if token not in stopwords
    ]
    def row_text_contains(row):
        row_text = " ".join(row.astype(str).str.lower())
        return all(token in row_text for token in query_tokens)

    mask = df.apply(row_text_contains, axis=1)
    if mask.any():
        top = df[mask].iloc[0]
        return (
            f"Teams: {top['team1']} vs {top['team2']} · Venue: {top.get('venue','Unknown')} · "
            f"Date: {top.get('date','Unknown')} · Winner: {top.get('winner','Unknown')}"
        )
    return "No matching match stats found."


def query_story(index, summaries, embedder, query: str) -> str:
    emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    distances, hits = index.search(emb, k=min(3, len(summaries)))
    top = [summaries[i] for i in hits[0] if i != -1]
    return "\n".join(top) if top else "No story context available."


def assemble_prompt(question: str, context: str) -> str:
    return (
        "You are a helpful cricket assistant. Use ONLY the context below to answer. "
        "If context contains no matching data, reply 'No data available'."
        f"\nContext: {context}"
        f"\nQuestion: {question}"
        "\nAnswer:" 
    )


def normalize_text(text: str) -> str:
    return text.strip().replace("\n", " ").strip()


def make_answer(pipe, prompt: str) -> str:
    out = pipe(
        prompt,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
    )
    text = out[0].get("generated_text", "")
    text = normalize_text(text)
    if prompt in text:
        text = text.replace(prompt, "").strip()
    # return first sentence to avoid repetition
    if "." in text:
        return text.split(".")[0].strip() + "."
    return text


def answer_from_context(question: str, context: str) -> str:
    """Deterministically answer simple STATS questions from the formatted context."""
    q = question.lower()
    if context.startswith("No matching"):
        return "No data available."

    # extract fields
    venue = re.search(r"Venue:\s*(.*?)\s*·", context)
    date = re.search(r"Date:\s*(.*?)\s*·", context)
    winner = re.search(r"Winner:\s*(.*)$", context)
    teams = re.search(r"Teams:\s*(.*?)\s*·", context)

    venue = venue.group(1) if venue else None
    date = date.group(1) if date else None
    winner = winner.group(1) if winner else None
    teams = teams.group(1) if teams else None

    if any(k in q for k in ["who won", "who was the winner", "winner", "won the match", "who won"]):
        return winner or "No data available."
    if any(k in q for k in ["where", "venue", "played"]):
        return venue or "No data available."
    if any(k in q for k in ["when", "date", "year", "what date"]):
        return date or "No data available."
    if any(k in q for k in ["who played", "which teams", "teams", "vs"]):
        return teams or "No data available."
    return ""


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # build or load delivery-level chunked BM25 + FAISS indices
    try:
        BM25, INDEX, META, EMBEDDER = load_chunk_index(str(CHUNK_INDEX_PATH), str(CHUNK_METADATA_PATH))
        print("Loaded existing chunked hybrid index")
    except Exception:
        print("Building chunked hybrid index from data...")
        BM25, INDEX, META, EMBEDDER = build_chunk_index(str(DATA_DIR), str(CHUNK_INDEX_PATH), str(CHUNK_METADATA_PATH))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
        device=-1,
    )

    questions = [
        "Who won the match between Mumbai Indians and Pune Warriors?",
        "Describe a tense final over of a cricket match.",
        "Where was the match between Mumbai Indians and Pune Warriors played?"
    ]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for question in questions:
            category = route_query(question)
            if category == "STATS":
                # use match-level metadata for STATS
                with open(MATCH_METADATA_PATH, 'r', encoding='utf-8') as mf:
                    match_meta = json.load(mf)
                context = query_stats(pd.DataFrame(match_meta), question)
                answer = answer_from_context(question, context)
                if not answer or answer == "No data available.":
                    prompt = assemble_prompt(question, context)
                    answer = make_answer(pipe, prompt)
            else:
                # perform hybrid retrieval (BM25 + dense) at delivery level
                hits = hybrid_search(question, BM25, INDEX, EMBEDDER, META, top_k=3, alpha=0.4)
                context = "\n".join([h.get("text", "") for h in hits])
                prompt = assemble_prompt(question, context)
                answer = make_answer(pipe, prompt)
            result = {
                "question": question,
                "category": category,
                "context": context,
                "answer": answer,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"Results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
