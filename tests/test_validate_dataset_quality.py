from scripts.validate_dataset_quality import validate_rows


def test_quality_gate_rejects_duplicates_and_placeholders() -> None:
    rows = [
        {"instruction": "A", "response": "A useful answer here.", "category": "reasoning"},
        {"instruction": "A", "response": "A useful answer here.", "category": "reasoning"},
        {"instruction": "B", "response": "TODO", "category": "coding"},
    ]

    report = validate_rows(rows)

    assert report["ready"] is False
    assert report["duplicates"] == 1
    assert report["placeholder_responses"] == 1


def test_quality_gate_accepts_clean_rows() -> None:
    rows = [
        {"instruction": "Explain gravity simply", "response": "Gravity attracts masses toward one another.", "category": "education"},
        {"instruction": "Translate good morning", "response": "ಶುಭೋದಯ, ನಿಮಗೆ ಶುಭೋದಯವಾಗಲಿ.", "category": "translation"},
        {"instruction": "Write a Python loop", "response": "Use a for loop to iterate over the items.", "category": "coding"},
    ]

    report = validate_rows(rows)

    assert report["ready"] is True
    assert report["duplicates"] == 0
    assert report["invalid_rows"] == 0


def test_quality_gate_rejects_missing_required_fields() -> None:
    report = validate_rows([
        {"instruction": "Only instruction", "response": "", "category": "education"},
    ])

    assert report["ready"] is False
    assert report["invalid_rows"] == 1
