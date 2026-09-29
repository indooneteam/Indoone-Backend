from pathlib import Path

from app.ai.knowledge import LocalKnowledgeBase
from app.ai.service import _knowledge_fallback_sentence


def test_knowledge_fallback_returns_matching_sentence(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "facts.txt").write_text(
        "The capital city of India is New Delhi.\n"
        "Day and night happen because Earth rotates on its axis.\n",
        encoding="utf-8",
    )

    knowledge_base = LocalKnowledgeBase.from_directory(knowledge_dir)
    hits = knowledge_base.search("What is the capital city of India?", limit=3)

    answer = _knowledge_fallback_sentence(
        "What is the capital city of India?",
        hits,
    )

    assert answer == "The capital city of India is New Delhi."


def test_knowledge_fallback_preserves_matching_script(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "facts.txt").write_text(
        "The capital city of India is New Delhi.\n"
        "ಭಾರತದ ರಾಜಧಾನಿ ನವದೆಹಲಿ.\n",
        encoding="utf-8",
    )

    knowledge_base = LocalKnowledgeBase.from_directory(knowledge_dir)
    hits = knowledge_base.search("ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು?", limit=3)

    answer = _knowledge_fallback_sentence(
        "ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು?",
        hits,
    )

    assert answer == "ಭಾರತದ ರಾಜಧಾನಿ ನವದೆಹಲಿ."
