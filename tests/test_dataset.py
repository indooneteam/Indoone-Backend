import pytest

from scripts.prepare_dataset import fingerprint, normalize_text, prepare_documents, split_documents, validate_split_quality


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


def test_validate_split_quality_rejects_cross_split_duplicates() -> None:
    document = "A sufficiently long document that must not appear in two splits."
    with pytest.raises(ValueError, match="duplicate document across dataset splits"):
        validate_split_quality([document], [document], ["Another unique sufficiently long document."])


def test_validate_split_quality_reports_split_counts() -> None:
    train = ["A sufficiently long training document for the dataset."]
    validation = ["A sufficiently long validation document for the dataset."]
    test = ["A sufficiently long test document for the dataset."]
    result = validate_split_quality(train, validation, test)
    assert result["train_documents"] == 1
    assert result["validation_documents"] == 1
    assert result["test_documents"] == 1
    assert result["total_characters"] == sum(map(len, train + validation + test))
