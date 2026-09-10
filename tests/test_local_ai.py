import torch
import pytest

from app.ai.model import MODEL_VERSION, IndooneTransformer
from app.ai.tokenizer import BPETokenizer
from app.ai.train import train


def test_bpe_tokenizer_round_trip(tmp_path) -> None:
    text = "Hello Indoone! Hello Indoone!"
    tokenizer = BPETokenizer.train(text, vocab_size=64, min_frequency=2)
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.vocab_size <= 64
    assert tokenizer.encode("?") == [tokenizer.stoi["<unk>"]]

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.encode(text) == ids
    assert loaded.decode(ids) == text


def test_bpe_special_tokens_round_trip() -> None:
    tokenizer = BPETokenizer.train("Hello Indoone", vocab_size=64, min_frequency=1)
    ids = tokenizer.encode("Hello", add_special_tokens=True)
    assert ids[0] == tokenizer.stoi["<bos>"]
    assert ids[-1] == tokenizer.stoi["<eos>"]
    assert tokenizer.decode(ids) == "Hello"


def test_transformer_forward_shapes_and_config() -> None:
    model = IndooneTransformer(
        vocab_size=32,
        block_size=16,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    )
    x = torch.randint(0, 32, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 32)
    assert loss is not None
    assert torch.isfinite(loss)
    assert model.config()["model_version"] == MODEL_VERSION
    assert model.config()["block_size"] == 16


def test_transformer_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n_embd must be divisible by n_head"):
        IndooneTransformer(vocab_size=32, n_embd=30, n_head=4)

    model = IndooneTransformer(vocab_size=32, block_size=8, n_embd=32, n_head=4, n_layer=2)
    with pytest.raises(ValueError, match="input sequence must not be empty"):
        model(torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="input sequence exceeds model block size"):
        model(torch.randint(0, 32, (1, 9)))
    with pytest.raises(ValueError, match="targets must match input tensor shape"):
        model(torch.randint(0, 32, (1, 4)), torch.randint(0, 32, (1, 3)))


def test_training_pipeline_writes_checkpoint_and_history(tmp_path) -> None:
    corpus = ("Indoone builds its own AI training pipeline. " * 20).strip()
    validation = ("Indoone evaluates its own model. " * 10).strip()
    corpus_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    output_dir = tmp_path / "model"
    corpus_path.write_text(corpus, encoding="utf-8")
    validation_path.write_text(validation, encoding="utf-8")

    final_loss = train(
        corpus_path=corpus_path,
        output_dir=output_dir,
        steps=2,
        seed=7,
        validation_path=validation_path,
        batch_size=2,
        checkpoint_interval=1,
    )

    assert torch.isfinite(torch.tensor(final_loss))
    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "indoone-small.pt").exists()
    assert (output_dir / "training_history.json").exists()
    assert (output_dir / "metadata.json").exists()
