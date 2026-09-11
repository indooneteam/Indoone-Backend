from pathlib import Path

import json
import torch

from app.ai.train import train


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
