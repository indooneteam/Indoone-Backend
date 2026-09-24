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


def test_retrieval_matches_connected_action_behavior() -> None:
    match = _retriever().retrieve("What actions did you take with connected services?")
    assert match is not None
    assert "permission" in match.example.response.casefold()


def test_retrieval_rejects_unrelated_text() -> None:
    assert _retriever().retrieve("Explain a bicycle repair shop schedule.") is None

def test_retrieval_matches_short_permission_prompt() -> None:
    match = _retriever().retrieve(
        "Answer concisely: can an assistant use an account without permission?"
    )
    assert match is not None
    assert "permission" in match.example.response.casefold()


def test_retrieval_matches_connected_service_actions() -> None:
    match = _retriever().retrieve(
        "What actions did you take with connected services?"
    )
    assert match is not None
    assert "permission" in match.example.response.casefold()

def test_retrieval_private_database_response_mentions_permission() -> None:
    match = _retriever().retrieve(
        "Tell me about a private company database you can access right now."
    )
    assert match is not None
    assert "permission" in match.example.response.casefold()
