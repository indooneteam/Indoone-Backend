from __future__ import annotations

from pathlib import Path

import pytest

from app.ai.local_engine import LocalAIEngine
from app.ai.service import LocalAIService


def test_local_engine_rejects_empty_message(tmp_path: Path) -> None:
    engine = LocalAIEngine(
        checkpoint=tmp_path / "missing.pt",
        tokenizer_path=tmp_path / "missing.json",
    )

    assert engine.ready is False
    with pytest.raises(RuntimeError, match="not trained yet"):
        import asyncio

        asyncio.run(engine.generate("Hello"))


def test_local_service_rejects_blank_message() -> None:
    service = LocalAIService()
    with pytest.raises(ValueError, match="message cannot be empty"):
        import asyncio

        asyncio.run(service.generate("   "))
