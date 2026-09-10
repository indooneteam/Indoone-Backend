from pathlib import Path

import torch

from app.ai.local_engine import LocalAIEngine
from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer


def test_local_engine_loads_checkpoint_with_model_version(tmp_path: Path) -> None:
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
    tokenizer.save(tokenizer_path)
    torch.save(
        {"config": model.config(), "model_state": model.state_dict()},
        checkpoint,
    )

    engine = LocalAIEngine(checkpoint=checkpoint, tokenizer_path=tokenizer_path)

    assert engine.ready
    assert engine.model is not None
    assert engine.tokenizer is not None
