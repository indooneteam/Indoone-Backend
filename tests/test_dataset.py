from scripts.prepare_dataset import fingerprint, normalize_text, prepare_documents, split_documents


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text(" Hello\r\n\r\n world  ") == "Hello\n\nworld"


def test_split_documents_uses_blank_lines() -> None:
    assert split_documents("one\n\ntwo\n\nthree") == ["one", "two", "three"]


def test_prepare_documents_filters_and_deduplicates() -> None:
    docs = [
        "tiny",
        "This is a sufficiently long document for the dataset pipeline.",
        "this is a sufficiently long document for the dataset pipeline.",
        "Another sufficiently long document that should remain unique.",
    ]
    result = prepare_documents(docs)
    assert len(result) == 2


def test_fingerprint_is_deterministic() -> None:
    assert fingerprint("Hello") == fingerprint("hello ")
