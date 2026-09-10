from pathlib import Path

import torch

from app.ai.evaluate import evaluate_checkpoint
from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer


def test_evaluate_checkpoint_returns_loss_and_perplexity(tmp_path: Path) -> None:
    corpus = "Indoone builds local AI. " * 20
    tokenizer = BPETokenizer.train(corpus, vocab_size=64, min_frequency=1)
    model = IndooneTransformer(
        vocab_size=tokenizer.vocab_size,
        block_size=8,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    )

    checkpoint = tmp_path / "indoone-small.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    corpus_path = tmp_path / "test.txt"
    tokenizer.save(tokenizer_path)
    corpus_path.write_text(corpus, encoding="utf-8")
    torch.save(
        {"config": model.config(), "model_state": model.state_dict()},
        checkpoint,
    )

    metrics = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        tokenizer_path=tokenizer_path,
        corpus_path=corpus_path,
        batch_size=2,
    )

    assert metrics["model_version"] == "indoone-gpt-v1"
    assert metrics["corpus_tokens"] > 9
    assert metrics["evaluation_batches"] > 0
    assert isinstance(metrics["loss"], float)
    assert metrics["loss"] >= 0.0
    assert isinstance(metrics["perplexity"], float)
    assert metrics["perplexity"] >= 1.0
