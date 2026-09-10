from pathlib import Path

from app.ai.train import train


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
