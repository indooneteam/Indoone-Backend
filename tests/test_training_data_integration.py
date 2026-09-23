from pathlib import Path

import torch

from app.ai.tokenizer import BPETokenizer
from app.ai.train import _instruction_batchify, train
from app.ai.training_data import TrainingExample


def test_train_accepts_instruction_dataset(tmp_path: Path) -> None:
    corpus = tmp_path / "train.txt"
    corpus.write_text(
        "This is a small language-model corpus with enough repeated content for a smoke test. "
        * 20,
        encoding="utf-8",
    )
    instructions = tmp_path / "instructions.jsonl"
    instructions.write_text(
        '{"instruction":"Say hello","response":"Hello from Indoone","category":"greeting"}\n',
        encoding="utf-8",
    )

    output = tmp_path / "model"
    loss = train(
        corpus,
        output,
        steps=1,
        seed=7,
        validation_path=None,
        batch_size=2,
        checkpoint_interval=1,
        learning_rate=3e-4,
        instruction_path=instructions,
    )

    assert loss >= 0
    assert (output / "instruction_corpus.txt").exists()
    assert (output / "indoone-small.pt").exists()
    assert (output / "tokenizer.json").exists()


def test_instruction_batch_only_scores_response_tokens() -> None:
    tokenizer = BPETokenizer.train(
        "<instruction>\\nSay hello\\n</instruction>\\n<response>\\nHello\\n</response>\\n<eos>",
        vocab_size=64,
        min_frequency=1,
    )
    x, y = _instruction_batchify(
        [TrainingExample("Say hello", "Hello", "greeting")],
        tokenizer,
        block_size=64,
        batch_size=1,
        device="cpu",
        generator=torch.Generator().manual_seed(1),
    )
    assert x.shape == y.shape
    assert (y == -100).any()
    assert (y != -100).any()
    assert y[0, -1].item() == tokenizer.stoi["<eos>"]
