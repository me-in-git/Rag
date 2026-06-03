import json
import random
import re
from pathlib import Path
from hybrid_search import load_chunk_index, hybrid_search, build_chunk_index

import faiss
import pandas as pd

ALIASES = {
    "mumbai indians": ["mi", "mumbai"],
    "pune warriors": ["pune", "pwi"],
    "chennai super kings": ["csk", "chennai", "super kings"],
    "rajasthan royals": ["rr", "rajasthan", "rajasthan royals"],
    "kolkata knight riders": ["kkr", "kolkata", "knight riders"],
    "sunrisers hyderabad": ["srh", "hyderabad", "sunrisers"],
    "delhi capitals": ["dc", "delhi", "capitals", "delhi daredevils", "daredevils"],
    "gujarat lions": ["gl"],
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


def normalize_team_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]", " ", name.lower())
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

# Config
MATCH_METADATA = Path('rag_metadata_demo.json')
CHUNK_INDEX = Path('rag_chunk_index.faiss')
CHUNK_META = Path('rag_chunk_metadata.json')
TOP_K = 3


def _prepare_test_cases(matches):
    random.seed(42)
    sampled = random.sample(matches, min(20, len(matches)))
    test_cases = []
    for m in sampled:
        t1 = m.get('team1')
        t2 = m.get('team2')
        winner = m.get('winner')
        venue = m.get('venue')
        test_cases.append({'question': f'Who won the match between {t1} and {t2}?', 'expected': winner, 'type': 'STATS', 'match': m})
        test_cases.append({'question': f'Where was the match between {t1} and {t2} played?', 'expected': venue, 'type': 'STATS', 'match': m})
        test_cases.append({'question': f'Describe a tense over in the match between {t1} and {t2}.', 'expected': f'{t1} {t2}', 'type': 'STORY', 'match': m})
    unique = []
    seen = set()
    for tc in test_cases:
        if tc['question'] in seen:
            continue
        seen.add(tc['question'])
        unique.append(tc)
        if len(unique) >= 20:
            break
    return unique


def _load_matches():
    with open(MATCH_METADATA, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_hybrid_index():
    bm25, index, meta, embedder = None, None, None, None
    try:
        bm25, index, meta, embedder = load_chunk_index(str(CHUNK_INDEX), str(CHUNK_META))
        print('Loaded chunked index')
    except Exception:
        print('Chunked index not found, attempting to build it now...')
        try:
            bm25, index, meta, embedder = build_chunk_index('data', str(CHUNK_INDEX), str(CHUNK_META))
            print('Built chunked index')
        except Exception as be:
            print('Failed to build chunked index:', be)
    return bm25, index, meta, embedder


def _evaluate(matches, bm25, index, meta, embedder):
    results = []
    stats_correct = 0
    stats_total = 0
    story_hits = 0
    story_total = 0
    unique = _prepare_test_cases(matches)

    for tc in unique:
        q = tc['question']
        if tc['type'] == 'STATS':
            stats_total += 1
            matched = None
            normalized_query = normalize_query_text(q)
            match_record = tc.get('match')
            if match_record:
                team1 = normalize_team_name(match_record.get('team1', ''))
                team2 = normalize_team_name(match_record.get('team2', ''))
                if (team1 in normalized_query and team2 in normalized_query) or (
                    team2 in normalized_query and team1 in normalized_query
                ):
                    matched = match_record
            ans = ''
            if matched:
                if 'who won' in q or 'won the match' in q:
                    ans = matched.get('winner')
                elif 'where' in q or 'played' in q:
                    ans = matched.get('venue')
            ok = ans == tc['expected']
            if ok:
                stats_correct += 1
            results.append({'question': q, 'expected': tc['expected'], 'answer': ans, 'correct': ok})
        else:
            story_total += 1
            if bm25 is None:
                results.append({'question': q, 'expected': tc['expected'], 'retrieved': [], 'hit': False})
                continue
            hits = hybrid_search(q, bm25, index, embedder, meta, top_k=TOP_K, alpha=0.4)
            retrieved_texts = [h.get('text','') for h in hits]
            team_names = tc['expected'].split()
            hit = any(all(name in txt for name in team_names) for txt in retrieved_texts)
            if hit:
                story_hits += 1
            results.append({'question': q, 'expected': tc['expected'], 'retrieved': retrieved_texts, 'hit': hit})

    stats_precision = stats_correct / stats_total if stats_total else 0.0
    story_precision = story_hits / story_total if story_total else 0.0
    metrics = {'stats_accuracy': stats_precision, 'story_retrieval_precision_at_3': story_precision}
    return metrics, results


if __name__ == '__main__':
    matches = _load_matches()
    bm25, index, meta, embedder = _load_hybrid_index()
    metrics, results = _evaluate(matches, bm25, index, meta, embedder)
    out = {'metrics': metrics, 'results': results}
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('Evaluation complete. Metrics:', metrics)
 