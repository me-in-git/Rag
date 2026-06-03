import json
import re
from pathlib import Path
from typing import List, Dict

import pandas as pd
from hybrid_search import load_chunk_index, hybrid_search
from evaluate_rag import normalize_query_text, normalize_team_name, ALIASES

DATA_DIR = Path('.')
MATCH_METADATA = DATA_DIR / 'rag_metadata_demo.json'
CHUNK_INDEX = DATA_DIR / 'rag_chunk_index.faiss'
CHUNK_META = DATA_DIR / 'rag_chunk_metadata.json'
TOP_K = 3

MANUAL_QUESTIONS = [
    {
        'question': 'Which team won at Wankhede when MI faced SRH?',
        'expected': 'Mumbai Indians',
        'type': 'STATS',
    },
    {
        'question': 'Where did the Gujarat Lions vs Sunrisers Hyderabad match take place?',
        'expected': 'Rajiv Gandhi International Stadium, Uppal',
        'type': 'STATS',
    },
    {
        'question': 'Which team won the match between Delhi Capitals and Royal Challengers Bangalore?',
        'expected': 'Royal Challengers Bangalore',
        'type': 'STATS',
    },
    {
        'question': 'What was the venue for the Chennai Super Kings versus Mumbai Indians game?',
        'expected': 'Wankhede Stadium',
        'type': 'STATS',
    },
    {
        'question': 'Who won the match between Kolkata Knight Riders and Royal Challengers Bangalore?',
        'expected': 'Kolkata Knight Riders',
        'type': 'STATS',
    },
    {
        'question': 'Describe a tense over in the match between Mumbai Indians and Sunrisers Hyderabad.',
        'expected': 'Mumbai Indians Sunrisers Hyderabad',
        'type': 'STORY',
    },
    {
        'question': 'Tell me about a dramatic over from the Delhi Capitals versus Royal Challengers Bangalore game.',
        'expected': 'Delhi Capitals Royal Challengers Bangalore',
        'type': 'STORY',
    },
    {
        'question': 'Give a tense over in the match between Kolkata Knight Riders and Royal Challengers Bangalore.',
        'expected': 'Kolkata Knight Riders Royal Challengers Bangalore',
        'type': 'STORY',
    },
    {
        'question': 'What was a dramatic over in the Chennai Super Kings vs Mumbai Indians match?',
        'expected': 'Chennai Super Kings Mumbai Indians',
        'type': 'STORY',
    },
    {
        'question': 'Describe a tense over in the match between Lucknow Super Giants and Punjab Kings.',
        'expected': 'Lucknow Super Giants Punjab Kings',
        'type': 'STORY',
    },
]

STOPWORDS = {
    'who', 'what', 'where', 'when', 'how', 'which', 'the', 'a', 'an', 'and', 'or',
    'of', 'in', 'on', 'at', 'between', 'match', 'team', 'teams', 'was', 'were',
    'did', 'by', 'for', 'to', 'is', 'are', 'won', 'lost', 'played', 'score',
    'game', 'vs', 'versus', 'face', 'faced', 'play', 'played', 'fixture',
}


def load_matches() -> pd.DataFrame:
    with open(MATCH_METADATA, 'r', encoding='utf-8') as f:
        matches = json.load(f)
    return pd.DataFrame(matches)


def row_matches(row: pd.Series, normalized_query: str, query_tokens: List[str]) -> bool:
    row_text = ' '.join(str(val).lower() for val in row[['team1', 'team2', 'venue', 'winner', 'date']].tolist())
    row_text = normalize_query_text(row_text)
    return all(token in row_text for token in query_tokens)


def query_stats(df: pd.DataFrame, query: str) -> str:
    normalized_query = normalize_query_text(query)
    query_tokens = [token for token in re.findall(r"\w+", normalized_query) if token not in STOPWORDS]

    matched_rows = []
    team_names = [team for team in ALIASES if team in normalized_query]
    if len(team_names) >= 2:
        for _, row in df.iterrows():
            team1 = normalize_team_name(row['team1'])
            team2 = normalize_team_name(row['team2'])
            if (team1 in team_names and team2 in team_names) or (team2 in team_names and team1 in team_names):
                matched_rows.append(row)
        if matched_rows:
            row = matched_rows[0]
            return f"Teams: {row['team1']} vs {row['team2']} · Venue: {row['venue']} · Date: {row['date']} · Winner: {row['winner']}"

    mask = df.apply(lambda row: row_matches(row, normalized_query, query_tokens), axis=1)
    if mask.any():
        row = df[mask].iloc[0]
        return f"Teams: {row['team1']} vs {row['team2']} · Venue: {row['venue']} · Date: {row['date']} · Winner: {row['winner']}"
    return 'No matching match stats found.'


def answer_from_context(question: str, context: str) -> str:
    if context.startswith('No matching'):
        return 'No data available.'
    venue = re.search(r'Venue:\s*(.*?)\s*·', context)
    date = re.search(r'Date:\s*(.*?)\s*·', context)
    winner = re.search(r'Winner:\s*(.*?)$', context)
    teams = re.search(r'Teams:\s*(.*?)\s*·', context)
    venue = venue.group(1) if venue else None
    date = date.group(1) if date else None
    winner = winner.group(1) if winner else None
    teams = teams.group(1) if teams else None
    q = question.lower()
    if any(k in q for k in ['who won', 'winner', 'won the match', 'which team won', 'which team won at', 'which team won the match']):
        return winner or 'No data available.'
    if any(k in q for k in ['where', 'venue', 'played', 'took place', 'what was the venue']):
        return venue or 'No data available.'
    if any(k in q for k in ['when', 'date', 'year']):
        return date or 'No data available.'
    return 'No data available.'


def normalize_text(text: str) -> str:
    return text.strip().replace('\n', ' ').strip()


def load_hybrid_index():
    return load_chunk_index(str(CHUNK_INDEX), str(CHUNK_META))


def run_manual_eval():
    df = load_matches()
    bm25, index, meta, embedder = load_hybrid_index()
    results: List[Dict] = []
    stats_correct = 0
    stats_total = 0
    story_hits = 0
    story_total = 0

    for item in MANUAL_QUESTIONS:
        q = item['question']
        expected = item['expected']
        if item['type'] == 'STATS':
            stats_total += 1
            context = query_stats(df, q)
            answer = answer_from_context(q, context)
            ok = answer == expected
            if ok:
                stats_correct += 1
            results.append({
                'question': q,
                'category': 'STATS',
                'expected': expected,
                'answer': answer,
                'context': context,
                'correct': ok,
            })
        else:
            story_total += 1
            if bm25 is None:
                results.append({
                    'question': q,
                    'category': 'STORY',
                    'expected': expected,
                    'retrieved': [],
                    'hit': False,
                })
                continue
            hits = hybrid_search(q, bm25, index, embedder, meta, top_k=TOP_K, alpha=0.4)
            retrieved_texts = [h.get('text', '') for h in hits]
            expected_team_tokens = expected.split()
            hit = any(all(token in txt for token in expected_team_tokens) for txt in retrieved_texts)
            if hit:
                story_hits += 1
            results.append({
                'question': q,
                'category': 'STORY',
                'expected': expected,
                'retrieved': retrieved_texts,
                'hit': hit,
            })

    metrics = {
        'stats_accuracy': stats_correct / stats_total if stats_total else 0.0,
        'story_retrieval_precision_at_3': story_hits / story_total if story_total else 0.0,
    }
    out = {'metrics': metrics, 'results': results}
    with open('manual_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print('Manual evaluation complete.')
    print('Metrics:', metrics)
    for r in results:
        print(r)


if __name__ == '__main__':
    run_manual_eval()
