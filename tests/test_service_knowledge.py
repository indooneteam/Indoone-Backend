import asyncio

import app.ai.service as service
from app.ai.knowledge import KnowledgeDocument, LocalKnowledgeBase


def test_service_includes_retrieved_knowledge(monkeypatch) -> None:
    monkeypatch.setattr(service, "_runtime", None)
    monkeypatch.setattr(
        service,
        "_knowledge_base",
        LocalKnowledgeBase(
            [KnowledgeDocument("doc", "Indoone Architecture", "Indoone uses a local Transformer model.")]
        ),
    )
    reply = asyncio.run(service.generate_reply("What model does Indoone use?"))

    assert reply == "Indoone uses a local Transformer model."
    assert reply == "Indoone uses a local Transformer model."
