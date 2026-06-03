# RAG Project

A polished end-to-end cricket commentary system built with data augmentation, fine-tuning, and hybrid retrieval.

## Highlights

- Data augmentation pipeline for cricket commentary generation
- Fine-tuning support using `peft` and `trl` for efficient instruction tuning
- Hybrid RAG retrieval layer with `faiss` and `sentence-transformers`
- Production-ready FastAPI deployment service
- Containerized app via `Dockerfile`
- Test coverage for pipeline utilities and notebook validation

## Project structure

- `pipeline.py` — CLI for augmentation, fine-tuning, and RAG index creation
- `app.py` — FastAPI service for commentary generation and hybrid cricket QA
- `run_all.py` — Demo runner for a minimal end-to-end proof-of-concept
- `requirements.txt` — Python dependencies
- `Dockerfile` — Container build recipe
- `tests/test_pipeline.py` — Basic unit tests for the pipeline

## New features added

- Delivery/over-level chunking for RAG (over-level chunks improve story retrieval granularity).
- True hybrid retrieval combining BM25 (`rank_bm25`) and FAISS dense vectors with score fusion.
- Deterministic STATS resolver for factual match queries and a generative STORY pathway.
- Evaluation harness: `evaluate_rag.py` — samples 20 Q/A and reports STATS accuracy and STORY retrieval precision@3.
- `hybrid_search.py` — build/load chunked BM25+FAISS indices and hybrid search API.

## Quick start

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Run the demo pipeline:

```bash
python run_all.py
```

This will generate a small sample dataset and build a FAISS retrieval index.
If `google/gemma-2-2b-it` is unavailable, the demo falls back to `distilgpt2` so the pipeline can still be validated locally.

### Example outputs (from local demo run)

```
Q: Who won the match between Mumbai Indians and Pune Warriors?
A: Mumbai Indians

Q: Where was the match between Mumbai Indians and Pune Warriors played?
A: Wankhede Stadium

Q: Describe a tense final over of a cricket match.
A: The final over of a cricket match is a tense one.
```


### Evaluation

- Run the evaluation harness:

```bash
python evaluate_rag.py
```

This produces `evaluation_results.json` containing `stats_accuracy` and `story_retrieval_precision_at_3`.



The evaluation harness now builds the chunked over-level index automatically if it is missing, so the metrics are reproducible with `python evaluate_rag.py`.



3. Fine-tune the model:

```bash
python pipeline.py finetune \
  --model-id google/gemma-2-2b-it \
  --data-file data/rich_commentary_train.jsonl \
  --output-dir models/gemma-2-cricket
```

4. Start the API server:

```bash
python app.py --model-dir models/gemma-2-cricket --data-dir data
```

Visit `http://127.0.0.1:8000/docs` for Swagger UI documentation.

## API endpoints

- `POST /commentary` — generate ball-by-ball commentary from a delivery JSON
- `POST /query` — ask a cricket question with hybrid RAG context

## Tests

Run the test suite with:

```bash
python -m pytest tests/test_pipeline.py
```

## Docker

Build and run the container:

```bash
docker build -t gemma-cricket .
docker run --rm -p 8000:8000 gemma-cricket
```

Mount `data/` and `models/` if you want to use local artifacts inside the container.
