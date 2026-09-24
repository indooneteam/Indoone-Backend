from pathlib import Path

from app.ai.instruction_retrieval import InstructionRetriever


def _retriever() -> InstructionRetriever:
    root = Path(__file__).resolve().parents[1]
    return InstructionRetriever(
        (
            root / "data/raw/indoone_instructions.jsonl",
            root / "data/raw/core_instruction_seed.jsonl",
            root / "data/raw/indoone_phone_contacts_examples.jsonl",
        )
    )


def test_retrieval_matches_account_permission_behavior() -> None:
    match = _retriever().retrieve(
        "How should Indoone handle a request to use an account without user permission?"
    )
    assert match is not None
    assert "permission" in match.example.response.casefold()
    assert "authorized" in match.example.response.casefold()


def test_retrieval_rejects_unrelated_text() -> None:
    assert _retriever().retrieve("Explain a bicycle repair shop schedule.") is None
