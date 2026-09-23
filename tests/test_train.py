from pathlib import Path

import json
import torch

from app.ai.tokenizer import BPETokenizer
from app.ai.train import train


def test_bpe_tokenizer_round_trip(tmp_path: Path) -> None:
    text = "Namaskara! Hello. Kannada: ನಮಸ್ಕಾರ. Hindi: नमस्ते."
    tokenizer = BPETokenizer.train(text, vocab_size=128, min_frequency=1)

    encoded = tokenizer.encode(text, add_special_tokens=True)
    decoded = tokenizer.decode(encoded)

    assert tokenizer.vocab_size <= 128
    assert tokenizer.stoi["<bos>"] in encoded
    assert tokenizer.stoi["<eos>"] in encoded
    assert decoded == text

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)

    assert loaded.decode(encoded) == text
    assert loaded.stoi["<bos>"] == tokenizer.stoi["<bos>"]


def test_train_writes_checkpoint_and_history(tmp_path: Path) -> None:
    corpus = (
        "Indoone builds local AI for useful assistance.\n\n"
        "The assistant should be clear, honest, and respectful.\n\n"
        "The model learns from curated text and is evaluated for quality and safety.\n\n"
        "Authorized tools may be used only when the user has permission.\n"
    )
    corpus_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    output_dir = tmp_path / "model"
    corpus_path.write_text(corpus, encoding="utf-8")
    validation_path.write_text(corpus, encoding="utf-8")

    loss = train(
        corpus_path=corpus_path,
        output_dir=output_dir,
        steps=2,
        seed=42,
        validation_path=validation_path,
        batch_size=1,
        checkpoint_interval=1,
        learning_rate=1e-3,
    )

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert (output_dir / "indoone-small.pt").exists()
    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "training_history.json").exists()
    assert (output_dir / "metadata.json").exists()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["source_fingerprint"]) == 64
    assert len(metadata["validation_fingerprint"]) == 64
    assert metadata["instruction_data_enabled"] is False
    assert metadata["training_text_characters"] == len(corpus)

    checkpoint = torch.load(
        output_dir / "indoone-small.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["model_state"]
    assert checkpoint["config"]["block_size"] >= 2



def test_weighted_instruction_pools_prioritize_curated_examples(tmp_path: Path) -> None:
    curated = tmp_path / "curated_instructions.jsonl"
    generated = tmp_path / "generated_multilingual_examples.jsonl"
    capability = tmp_path / "indoone_phone_contacts_examples.jsonl"

    curated.write_text(
        json.dumps({"instruction": "Curated one", "response": "Useful curated response", "category": "general"}) + "\n"
        + json.dumps({"instruction": "Curated two", "response": "Another curated response", "category": "general"}) + "\n",
        encoding="utf-8",
    )
    generated.write_text(
        "".join(
            json.dumps({"instruction": f"Generated {i}", "response": f"Generated response {i}", "category": "math"}) + "\n"
            for i in range(20)
        ),
        encoding="utf-8",
    )
    capability.write_text(
        json.dumps({"instruction": "Call contact", "response": "Ask for confirmation before calling.", "category": "tools"}) + "\n",
        encoding="utf-8",
    )

    from app.ai.train import _merge_instruction_sets_weighted

    examples, _, weights, policy = _merge_instruction_sets_weighted(
        [curated, generated, capability]
    )

    assert len(examples) == 23
    assert abs(sum(weights) - 1.0) < 1e-6
    assert policy[curated.name]["target_probability"] == 0.70
    assert policy[generated.name]["target_probability"] == 0.25
    assert policy[capability.name]["target_probability"] == 0.05
    assert abs(sum(weights[:2]) - 0.70) < 1e-6
    assert abs(sum(weights[2:22]) - 0.25) < 1e-6
    assert abs(weights[-1] - 0.05) < 1e-6
