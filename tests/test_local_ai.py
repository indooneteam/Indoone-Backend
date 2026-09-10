import torch

from app.ai.model import IndooneTransformer
from app.ai.tokenizer import CharacterTokenizer


def test_tokenizer_round_trip() -> None:
    tokenizer = CharacterTokenizer.from_text("Hello Indoone!")
    ids = tokenizer.encode("Hello Indoone!")
    assert tokenizer.decode(ids) == "Hello Indoone!"


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
