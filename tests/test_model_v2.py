import torch

from app.ai.model import IndooneTransformer, MODEL_VERSION


def test_model_v2_config_and_forward_shape() -> None:
    model = IndooneTransformer(vocab_size=512)
    assert MODEL_VERSION == "indoone-gpt-v2"
    assert model.block_size == 512
    assert model.n_embd == 384
    assert model.n_head == 8
    assert model.n_layer == 10

    tokens = torch.randint(0, 512, (2, 32))
    logits, loss = model(tokens, tokens)

    assert logits.shape == (2, 32, 512)
    assert loss is not None
    assert torch.isfinite(loss)


def test_model_rejects_sequence_beyond_context() -> None:
    model = IndooneTransformer(vocab_size=128, block_size=16)
    tokens = torch.randint(0, 128, (1, 17))
    try:
        model(tokens)
    except ValueError as exc:
        assert "block size" in str(exc)
    else:
        raise AssertionError("expected block-size validation")
