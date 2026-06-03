import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def _delivery_to_text(match: dict, inning_idx: int, over_idx: int, delivery_idx: int, delivery: dict) -> str:
    teams = match.get("info", {}).get("teams", [])
    team_text = " vs ".join(teams) if teams else "Unknown vs Unknown"
    venue = match.get("info", {}).get("venue", "Unknown")
    date = match.get("info", {}).get("dates", ["Unknown"])[0]
    batter = delivery.get("batter") or delivery.get("batsman") or "Unknown"
    bowler = delivery.get("bowler") or delivery.get("bowling") or "Unknown"
    runs = delivery.get("runs", {}).get("batter", delivery.get("runs", {}).get("batsman", 0))
    desc = f"{team_text} | {venue} | {date} | Inning {inning_idx} Over {over_idx} Delivery {delivery_idx}: {bowler} to {batter}, runs {runs}."
    # include any event text if present
    if "wickets" in delivery and delivery.get("wickets"):
        desc += " Wicket: " + json.dumps(delivery.get("wickets"))
    return desc


def build_chunk_index(data_dir: str, index_path: str, metadata_path: str, embedder_name: str = "all-MiniLM-L6-v2") -> Tuple[BM25Okapi, faiss.IndexFlatIP, List[dict], SentenceTransformer]:
    files = sorted(Path(data_dir).glob("*.json"))
    chunks: List[str] = []
    meta: List[dict] = []

    # Chunk at OVER level: aggregate all deliveries within an over into one chunk
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            match = json.load(f)
        innings = match.get("innings", [])
        for i_idx, inning in enumerate(innings):
            overs = inning.get("overs", [])
            for o_idx, over in enumerate(overs):
                deliveries = over.get("deliveries", [])
                if not deliveries:
                    continue
                # build over-level text by concatenating delivery summaries
                delivery_texts = []
                for d_idx, delivery in enumerate(deliveries):
                    dt = _delivery_to_text(match, i_idx + 1, o_idx + 1, d_idx + 1, delivery)
                    delivery_texts.append(dt)
                over_text = " ".join(delivery_texts)
                chunks.append(over_text)
                meta.append({
                    "file_path": str(path),
                    "inning": i_idx + 1,
                    "over": o_idx + 1,
                    "text": over_text,
                })

    if not chunks:
        raise RuntimeError("No delivery chunks found in data directory")

    # BM25
    tokenized = [c.split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    # dense embeddings + faiss
    embedder = SentenceTransformer(embedder_name)
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # persist metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    faiss.write_index(index, index_path)

    return bm25, index, meta, embedder


def load_chunk_index(index_path: str, metadata_path: str, embedder_name: str = "all-MiniLM-L6-v2"):
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Index or metadata not found")
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    chunks = [m["text"] for m in meta]
    tokenized = [c.split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    embedder = SentenceTransformer(embedder_name)
    return bm25, index, meta, embedder


def hybrid_search(query: str, bm25: BM25Okapi, index: faiss.IndexFlatIP, embedder: SentenceTransformer, meta: List[dict], top_k: int = 5, alpha: float = 0.5) -> List[dict]:
    # BM25 retrieval (raw term-frequency style scores)
    q_tokens = query.split()
    bm25_scores = bm25.get_scores(q_tokens)
    # pick a candidate pool from BM25 (wider than top_k)
    bm25_candidate_idx = np.argsort(bm25_scores)[::-1][: top_k * 2]

    # dense retrieval
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    distances, hits = index.search(q_emb, k=min(len(meta), top_k * 2))
    dense_hits = hits[0]
    dense_scores = distances[0]

    # Normalize BM25 scores to 0-1 over the selected pool
    bm25_pool_scores = np.array([float(bm25_scores[i]) for i in bm25_candidate_idx])
    bm25_range = bm25_pool_scores.max() - bm25_pool_scores.min() + 1e-9
    bm25_norm = (bm25_pool_scores - bm25_pool_scores.min()) / bm25_range

    bm25_map = {int(idx): float(score) for idx, score in zip(bm25_candidate_idx, bm25_norm)}

    # Normalize dense scores (they are inner products on normalized vectors -> cosine in [-1,1])
    dense_valid = [(int(idx), float(s)) for idx, s in zip(dense_hits, dense_scores) if idx != -1]
    dense_scores_arr = np.array([s for _, s in dense_valid]) if dense_valid else np.array([])
    if dense_scores_arr.size and dense_scores_arr.max() > dense_scores_arr.min():
        dense_norm = (dense_scores_arr - dense_scores_arr.min()) / (dense_scores_arr.max() - dense_scores_arr.min())
    else:
        dense_norm = np.zeros_like(dense_scores_arr)

    dense_map = {idx: float(score) for (idx, _), score in zip(dense_valid, dense_norm)}

    # Combine candidate indices
    candidates = {}
    for idx, s in bm25_map.items():
        candidates[idx] = candidates.get(idx, 0.0) + (1.0 - alpha) * s
    for idx, s in dense_map.items():
        candidates[idx] = candidates.get(idx, 0.0) + alpha * s

    # select top_k by combined normalized score
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = [meta[idx] for idx, _ in ranked]
    return results
