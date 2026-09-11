import asyncio

import app.ai.service as service
from app.ai.knowledge import KnowledgeDocument, LocalKnowledgeBase


def test_service_includes_retrieved_knowledge(monkeypatch) -> None:
    captured: list[str] = []

    class FakeEngine:
        async def generate(self, message: str) -> str:
            captured.append(message)
            return "ok"

    monkeypatch.setattr(service, "_runtime", None)
    monkeypatch.setattr(
        service,
        "_knowledge_base",
        LocalKnowledgeBase(
            [KnowledgeDocument("doc", "Indoone Architecture", "Indoone uses a local Transformer model.")]
        ),
    )
    monkeypatch.setattr(service, "_fallback_engine", FakeEngine())

    reply = asyncio.run(service.generate_reply("What model does Indoone use?"))

    assert reply == "ok"
    assert "Indoone Architecture" in captured[0]
    assert "local Transformer model" in captured[0]
