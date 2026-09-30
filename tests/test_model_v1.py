import torch

from app.ai.model import IndooneTransformer, MODEL_VERSION


def test_model_v1_config_and_forward_shape() -> None:
    model = IndooneTransformer(vocab_size=512)
    assert MODEL_VERSION == "indoone-gpt-v1"
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


def test_email_adapter_is_identity_when_disabled() -> None:
    base = IndooneTransformer(vocab_size=256)
    adapted = IndooneTransformer(
        vocab_size=256,
        email_adapter_dim=8,
        email_adapter_layers=2,
    )

    missing, unexpected = adapted.load_state_dict(
        base.state_dict(),
        strict=False,
    )
    assert unexpected == []
    assert missing
    assert all("email_adapter_" in key for key in missing)

    base.eval()
    adapted.eval()

    tokens = torch.randint(0, 256, (2, 24))
    base_logits, _ = base(tokens)
    disabled_logits, _ = adapted(
        tokens,
        use_email_adapter=False,
    )

    assert torch.equal(base_logits, disabled_logits)


def test_email_adapter_starts_as_identity() -> None:
    model = IndooneTransformer(
        vocab_size=128,
        email_adapter_dim=8,
        email_adapter_layers=2,
    )
    model.eval()

    tokens = torch.randint(0, 128, (1, 16))

    disabled, _ = model(
        tokens,
        use_email_adapter=False,
    )
    enabled, _ = model(
        tokens,
        use_email_adapter=True,
    )

    assert torch.equal(disabled, enabled)
