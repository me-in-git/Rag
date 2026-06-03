import argparse
import glob
import json
import os
import random
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import SFTConfig, SFTTrainer


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def select_dtype(device: str):
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_generation_pipeline(model_id: str, device: str, max_new_tokens: int = 150):
    dtype = select_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
    )

    if device == "cpu":
        model = model.to("cpu")

    text_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        device=0 if device == "cuda" else -1,
    )
    return tokenizer, text_pipe


def delivery_commentary_prompt(delivery: dict) -> str:
    batter = delivery.get("batter", "Unknown")
    bowler = delivery.get("bowler", "Unknown")
    runs = delivery.get("runs", {}).get("batter", 0)
    extras = delivery.get("extras", {})
    wicket = delivery.get("wickets", [])

    return (
        f"You are an expert cricket commentator. Write an exciting single-sentence commentary for the following delivery:\n"
        f"Bowler: {bowler}\n"
        f"Batter: {batter}\n"
        f"Runs Scored: {runs}\n"
        f"Extras: {extras}\n"
        f"Wicket: {wicket}\n"
        f"Commentary:"
    )


def generate_rich_commentary(model_id: str, data_dir: str, output_file: str, num_samples: int = 500):
    device = select_device()
    tokenizer, pipe = load_generation_pipeline(model_id, device)

    files = glob.glob(os.path.join(data_dir, "*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    random.shuffle(files)
    files = files[: min(len(files), 200)]

    examples = []
    count = 0

    for file_path in tqdm(files, desc="Generating samples"):
        if count >= num_samples:
            break

        with open(file_path, "r", encoding="utf-8") as f:
            match = json.load(f)

        innings = match.get("innings", [])
        if not innings:
            continue

        inning = random.choice(innings)
        over = random.choice(inning.get("overs", [])) if inning.get("overs") else None
        if not over:
            continue

        delivery = random.choice(over.get("deliveries", [])) if over.get("deliveries") else None
        if not delivery:
            continue

        prompt = delivery_commentary_prompt(delivery)
        text = pipe(prompt, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"]
        output = text[len(prompt) :] if text.startswith(prompt) else text
        output = output.strip()

        examples.append(
            {
                "instruction": "Write exciting cricket commentary for this ball.",
                "input": json.dumps(delivery, ensure_ascii=False),
                "output": output,
            }
        )
        count += 1

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in examples:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} examples to {output_file}")


def format_sft_text(example: dict) -> str:
    return (
        "<start_of_turn>user\n"
        f"{example['instruction']}\n"
        "Input:\n"
        f"{example['input']}<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{example['output']}<end_of_turn>"
    )


def fine_tune(model_id: str, data_file: str, output_dir: str, epochs: int = 3, batch_size: int = 4, lr: float = 2e-4):
    device = select_device()
    dtype = select_dtype(device)

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Could not find data file: {data_file}")

    dataset = load_dataset("json", data_files=data_file, split="train")
    dataset = dataset.train_test_split(test_size=0.1)

    def convert(example):
        return {"text": format_sft_text(example)}

    dataset["train"] = dataset["train"].map(convert, remove_columns=dataset["train"].column_names)
    dataset["test"] = dataset["test"].map(convert, remove_columns=dataset["test"].column_names)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=device == "cuda",
        bf16=False,
        report_to="none",
        dataset_text_field="text",
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def build_rag(data_dir: str, index_path: str, metadata_path: str, embedder_name: str = "all-MiniLM-L6-v2"):
    files = glob.glob(os.path.join(data_dir, "*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    rows = []
    summaries = []

    for path in tqdm(files, desc="Reading matches"):
        with open(path, "r", encoding="utf-8") as f:
            match = json.load(f)

        info = match.get("info", {})
        teams = info.get("teams", [])
        venue = info.get("venue", "Unknown")
        winner = info.get("outcome", {}).get("winner", "Unknown")
        player_of_match = info.get("player_of_match", ["Unknown"])
        dates = info.get("dates", ["Unknown"])

        row = {
            "file_path": path,
            "team1": teams[0] if len(teams) > 0 else "Unknown",
            "team2": teams[1] if len(teams) > 1 else "Unknown",
            "venue": venue,
            "winner": winner,
            "player_of_match": player_of_match[0] if player_of_match else "Unknown",
            "date": dates[0] if dates else "Unknown",
        }

        summary = f"{row['team1']} vs {row['team2']} at {row['venue']} on {row['date']}: winner {row['winner']}."
        rows.append(row)
        summaries.append(summary)

    df = pd.DataFrame(rows)
    embedder = SentenceTransformer(embedder_name)
    embeddings = embedder.encode(summaries, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"RAG index saved to {index_path}")
    print(f"Metadata saved to {metadata_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma cricket pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_augment = subparsers.add_parser("augment", help="Generate augmented commentary training data")
    parser_augment.add_argument("--data-dir", required=True)
    parser_augment.add_argument("--output-file", required=True)
    parser_augment.add_argument("--model-id", required=True)
    parser_augment.add_argument("--num-samples", type=int, default=500)

    parser_finetune = subparsers.add_parser("finetune", help="Fine-tune Gemma on generated commentary data")
    parser_finetune.add_argument("--model-id", required=True)
    parser_finetune.add_argument("--data-file", required=True)
    parser_finetune.add_argument("--output-dir", required=True)
    parser_finetune.add_argument("--epochs", type=int, default=3)
    parser_finetune.add_argument("--batch-size", type=int, default=4)
    parser_finetune.add_argument("--lr", type=float, default=2e-4)

    parser_rag = subparsers.add_parser("build_rag", help="Build a FAISS vector index for hybrid RAG")
    parser_rag.add_argument("--data-dir", required=True)
    parser_rag.add_argument("--index-path", required=True)
    parser_rag.add_argument("--metadata-path", required=True)
    parser_rag.add_argument("--embedder", default="all-MiniLM-L6-v2")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "augment":
        generate_rich_commentary(args.model_id, args.data_dir, args.output_file, args.num_samples)
    elif args.command == "finetune":
        fine_tune(args.model_id, args.data_file, args.output_dir, args.epochs, args.batch_size, args.lr)
    elif args.command == "build_rag":
        build_rag(args.data_dir, args.index_path, args.metadata_path, args.embedder)


if __name__ == "__main__":
    main()
