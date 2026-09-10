from pathlib import Path

import scripts.train_and_evaluate as pipeline


def test_train_and_evaluate_orchestrates_steps(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "corpus.txt"
    processed = tmp_path / "processed"
    model = tmp_path / "model"
    source.write_text("seed corpus\n", encoding="utf-8")

    calls: list[str] = []

    def fake_prepare_dataset(**kwargs):
        calls.append("prepare")
        return {"documents": 3, "train": 1, "validation": 1, "test": 1, "total_characters": 120}

    def fake_train(**kwargs):
        calls.append("train")
        return 0.25

    def fake_evaluate_checkpoint(**kwargs):
        calls.append("evaluate")
        return {
            "model_version": "indoone-gpt-v1",
            "corpus_tokens": 10,
            "evaluation_batches": 1,
            "loss": 1.5,
            "perplexity": 4.48,
        }

    monkeypatch.setattr(pipeline, "prepare_dataset", fake_prepare_dataset)
    monkeypatch.setattr(pipeline, "train", fake_train)
    monkeypatch.setattr(pipeline, "evaluate_checkpoint", fake_evaluate_checkpoint)

    report = pipeline.train_and_evaluate(
        source=source,
        processed_dir=processed,
        model_dir=model,
        steps=2,
        seed=42,
        batch_size=2,
        checkpoint_interval=1,
        learning_rate=1e-3,
    )

    assert calls == ["prepare", "train", "evaluate"]
    assert report["training"]["final_train_loss"] == 0.25
    assert report["evaluation"]["perplexity"] == 4.48
    assert (model / "evaluation_report.json").exists()
