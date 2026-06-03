import json
import os
from pathlib import Path

from pipeline import format_sft_text, delivery_commentary_prompt


def test_delivery_commentary_prompt():
    delivery = {
        "batter": "Virat Kohli",
        "bowler": "James Anderson",
        "runs": {"batter": 4},
        "extras": {},
        "wickets": [],
    }
    prompt = delivery_commentary_prompt(delivery)
    assert "Batter: Virat Kohli" in prompt
    assert "Bowler: James Anderson" in prompt
    assert "Runs Scored: 4" in prompt


def test_format_sft_text():
    example = {
        "instruction": "Write commentary.",
        "input": '{"batter": "Kohli"}',
        "output": "A strong cover drive.",
    }
    text = format_sft_text(example)
    assert "<start_of_turn>user" in text
    assert "<start_of_turn>model" in text
    assert "A strong cover drive." in text


def test_notebook_demo_cell_headers():
    # Ensure the notebooks still include a pipeline reference for maintainability.
    notebook_path = Path("01_data_augmentation.ipynb")
    assert notebook_path.exists()
    with notebook_path.open("r", encoding="utf-8") as f:
        content = f.read()
    assert "pipeline.py" in content
