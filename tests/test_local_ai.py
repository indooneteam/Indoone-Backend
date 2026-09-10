import torch

from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer


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


def test_transformer_forward_shapes() -> None:
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
