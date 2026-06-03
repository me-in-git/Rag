import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from hybrid_search import build_chunk_index, load_chunk_index, hybrid_search


class DeliveryRequest(BaseModel):
    delivery: Dict


class QueryRequest(BaseModel):
    query: str


def select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def select_dtype(device: str):
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(model_dir: str, device: str):
    dtype = select_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
    )
    if device == "cpu":
        model = model.to("cpu")
    return tokenizer, model


def create_pipeline(tokenizer, model, max_new_tokens: int = 200):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
        device=0 if torch.cuda.is_available() else -1,
    )


def load_match_metadata(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    rows = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            match = json.load(f)
        info = match.get("info", {})
        teams = info.get("teams", [])
        venue = info.get("venue", "Unknown")
        winner = match.get("outcome", {}).get("winner", "Unknown")
        date = info.get("dates", ["Unknown"])[0]
        rows.append(
            {
                "file_path": file_path,
                "team1": teams[0] if len(teams) > 0 else "Unknown",
                "team2": teams[1] if len(teams) > 1 else "Unknown",
                "venue": venue,
                "winner": winner,
                "date": date,
            }
        )
    return pd.DataFrame(rows)


def commentary_prompt(delivery: dict) -> str:
    batter = delivery.get("batter", "Unknown")
    bowler = delivery.get("bowler", "Unknown")
    runs = delivery.get("runs", {}).get("batter", 0)
    extras = delivery.get("extras", {})
    wicket = delivery.get("wickets", [])
    return (
        "You are an expert cricket commentator. Write an exciting single-sentence commentary for the following delivery:\n"
        f"Bowler: {bowler}\n"
        f"Batter: {batter}\n"
        f"Runs Scored: {runs}\n"
        f"Extras: {extras}\n"
        f"Wicket: {wicket}\n"
        "Commentary:"
    )


def normalize_text(text: str) -> str:
    return text.strip().replace("\n", " ").strip()


ALIASES = {
    "mumbai indians": ["mi", "mumbai"],
    "pune warriors": ["pune", "pwi"],
    "chennai super kings": ["csk", "chennai", "super kings"],
    "rajasthan royals": ["rr", "rajasthan", "rajasthan royals"],
    "kolkata knight riders": ["kkr", "kolkata", "knight riders"],
    "sunrisers hyderabad": ["srh", "hyderabad", "sunrisers"],
    "delhi capitals": ["dc", "delhi", "capitals"],
    "royal challengers bangalore": ["rcb", "bangalore", "bengaluru", "royal challengers", "royal challengers bengaluru"],
    "lucknow super giants": ["lsg", "lucknow", "super giants"],
    "india": ["ind", "india"],
    "australia": ["aus", "australia"],
    "england": ["eng", "england"],
    "south africa": ["sa", "south africa"],
    "new zealand": ["nz", "new zealand"],
    "pakistan": ["pak", "pakistan"],
    "west indies": ["wi", "west indies"],
    "sri lanka": ["sl", "sri lanka"],
}


def _build_alias_phrases() -> dict:
    alias_map = {}
    for canonical, aliases in ALIASES.items():
        alias_map[tuple(canonical.split())] = canonical
        for alias in aliases:
            alias_map[tuple(alias.split())] = canonical
    return alias_map


ALIAS_MAP = _build_alias_phrases()
MAX_PHRASE_LEN = max(len(phrase) for phrase in ALIAS_MAP) if ALIAS_MAP else 0


def normalize_team_name(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return normalized
    alias_map = ALIAS_MAP
    tokens = re.findall(r"\w+", normalized)
    phrase = tuple(tokens)
    if phrase in alias_map:
        return alias_map[phrase]
    for canonical, aliases in ALIASES.items():
        if canonical == normalized or any(alias == normalized for alias in aliases):
            return canonical
    for canonical, aliases in ALIASES.items():
        if canonical in normalized or any(alias in normalized for alias in aliases):
            return canonical
    return normalized


def normalize_query_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    alias_map = ALIAS_MAP
    tokens = re.findall(r"\w+", normalized)
    if not tokens:
        return normalized
    max_phrase_len = MAX_PHRASE_LEN
    output_tokens = []
    i = 0
    while i < len(tokens):
        match_len = 0
        canonical = None
        for size in range(max_phrase_len, 0, -1):
            if i + size > len(tokens):
                continue
            phrase = tuple(tokens[i : i + size])
            if phrase in alias_map:
                match_len = size
                canonical = alias_map[phrase]
                break
        if match_len and canonical:
            output_tokens.append(canonical)
            i += match_len
        else:
            output_tokens.append(tokens[i])
            i += 1
    deduped = []
    for token in output_tokens:
        if not deduped or token != deduped[-1]:
            deduped.append(token)
    return " ".join(deduped)


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
        "play",
        "score",
        "game",
        "vs",
        "versus",
        "fixture",
        "face",
        "faced",
    }
    normalized_query = normalize_query_text(query)
    query_tokens = [
        token
        for token in re.findall(r"\w+", normalized_query)
        if token not in stopwords
    ]

    def row_matches(row):
        row_text = " ".join(row.astype(str).str.lower())
        row_text = normalize_query_text(row_text)
        return all(token in row_text for token in query_tokens)

    normalized_query = normalize_query_text(query)
    mask = df.apply(row_matches, axis=1)
    if mask.any():
        row = df[mask].iloc[0]
        return (
            f"Teams: {row['team1']} vs {row['team2']} · Venue: {row['venue']} · "
            f"Date: {row['date']} · Winner: {row['winner']}"
        )
    return "No matching match stats found."


def answer_from_context(question: str, context: str) -> str:
    if context.startswith("No matching"):
        return "No data available."

    venue = re.search(r"Venue:\s*(.*?)\s*·", context)
    date = re.search(r"Date:\s*(.*?)\s*·", context)
    winner = re.search(r"Winner:\s*(.*?)$", context)
    teams = re.search(r"Teams:\s*(.*?)\s*·", context)

    venue = venue.group(1) if venue else None
    date = date.group(1) if date else None
    winner = winner.group(1) if winner else None
    teams = teams.group(1) if teams else None

    q = question.lower()
    if any(k in q for k in ["who won", "winner", "won the match"]):
        return winner or "No data available."
    if any(k in q for k in ["where", "venue", "played"]):
        return venue or "No data available."
    if any(k in q for k in ["when", "date", "year"]):
        return date or "No data available."
    if any(k in q for k in ["who played", "which teams", "teams", "vs"]):
        return teams or "No data available."
    return "No data available."


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
    stat_keywords = [
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
    if any(keyword in text for keyword in stat_keywords):
        return "STATS"
    return "STATS"


def assemble_prompt(question: str, context: str) -> str:
    return (
        "You are a helpful cricket assistant. Use ONLY the context below to answer the question."
        f"\nContext: {context}"
        f"\nQuestion: {question}"
        "\nAnswer:"
    )


def load_chunked_hybrid(data_dir: str, embedder_name: str = "all-MiniLM-L6-v2"):
    index_path = os.path.join(data_dir, "rag_chunk_index.faiss")
    meta_path = os.path.join(data_dir, "rag_chunk_metadata.json")
    try:
        return load_chunk_index(index_path, meta_path, embedder_name)
    except Exception:
        return build_chunk_index(data_dir, index_path, meta_path, embedder_name)


parser = argparse.ArgumentParser(description="Run the cricket API service")
parser.add_argument("--model-dir", required=True, help="Path to the fine-tuned Gemma model directory")
parser.add_argument("--data-dir", default="data", help="Raw match JSON directory for RAG retrieval")
parser.add_argument("--embedder", default="all-MiniLM-L6-v2", help="Sentence transformer embedder")
args = parser.parse_args()

app = FastAPI(title="Gemma Cricket Service")

DEVICE = select_device()
TOKENIZER, MODEL = load_model(args.model_dir, DEVICE)
PIPE = create_pipeline(TOKENIZER, MODEL, max_new_tokens=200)

MATCHES_DF = load_match_metadata(args.data_dir)
CHUNK_BM25, CHUNK_INDEX, CHUNK_META, CHUNK_EMBEDDER = load_chunked_hybrid(args.data_dir, args.embedder)


@app.get("/")
def root():
    return {
        "service": "Gemma Cricket API",
        "device": DEVICE,
        "model_dir": args.model_dir,
        "chunked_hybrid_ready": CHUNK_INDEX is not None,
    }


@app.post("/commentary")
def commentary(request: DeliveryRequest):
    if not request.delivery:
        raise HTTPException(status_code=400, detail="Delivery object is required.")
    prompt = commentary_prompt(request.delivery)
    commentary_text = PIPE(prompt, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"]
    return {"prompt": prompt, "commentary": normalize_text(commentary_text)}


@app.post("/query")
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    category = route_query(request.query)
    if category == "STATS":
        context = query_stats(MATCHES_DF, request.query)
        answer = answer_from_context(request.query, context)
        if answer == "No data available.":
            prompt = assemble_prompt(request.query, context)
            answer = PIPE(prompt, do_sample=True, temperature=0.3, top_p=0.9)[0]["generated_text"]
    else:
        context = "No story context available."
        if CHUNK_BM25 is not None and CHUNK_INDEX is not None and CHUNK_EMBEDDER is not None:
            hits = hybrid_search(request.query, CHUNK_BM25, CHUNK_INDEX, CHUNK_EMBEDDER, CHUNK_META, top_k=3, alpha=0.4)
            context = "\n\n".join([h.get("text", "") for h in hits])
        prompt = assemble_prompt(request.query, context)
        answer = PIPE(prompt, do_sample=True, temperature=0.4, top_p=0.9)[0]["generated_text"]
    return {"query": request.query, "category": category, "context": context, "answer": normalize_text(answer)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
