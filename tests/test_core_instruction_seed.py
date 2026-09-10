from pathlib import Path

from app.ai.training_data import load_examples


def test_core_instruction_seed_is_valid() -> None:
    path = Path("data/raw/core_instruction_seed.jsonl")
    examples = load_examples(path)
    assert len(examples) >= 8
    categories = {example.category for example in examples}
    assert {"identity", "education", "honesty", "planning", "safety", "development", "research"} <= categories
