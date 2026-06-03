"""Run a minimal end-to-end demonstration of the Gemma cricket project."""

from pipeline import generate_rich_commentary, build_rag


def main():
    demo_config = {
        "model_id": "google/gemma-2-2b-it",
        "fallback_model_id": "distilgpt2",
        "data_dir": "data",
        "output_file": "data/rich_commentary_demo.jsonl",
        "num_samples": 2,
        "index_path": "rag_index_demo.faiss",
        "metadata_path": "rag_metadata_demo.json",
    }

    print("=== Gemma Cricket Demo ===")
    print("1) Generating a small sample dataset")

    try:
        generate_rich_commentary(
            model_id=demo_config["model_id"],
            data_dir=demo_config["data_dir"],
            output_file=demo_config["output_file"],
            num_samples=demo_config["num_samples"],
        )
    except Exception as exc:
        print(f"⚠️ Could not load model {demo_config['model_id']}: {exc}")
        print(f"▶ Falling back to demo model: {demo_config['fallback_model_id']}")
        generate_rich_commentary(
            model_id=demo_config["fallback_model_id"],
            data_dir=demo_config["data_dir"],
            output_file=demo_config["output_file"],
            num_samples=demo_config["num_samples"],
        )

    print("\n2) Building a FAISS hybrid RAG index")
    build_rag(
        data_dir=demo_config["data_dir"],
        index_path=demo_config["index_path"],
        metadata_path=demo_config["metadata_path"],
    )

    print("\n✅ Demo complete. See README.md for fine-tuning and deployment steps.")


if __name__ == "__main__":
    main()
