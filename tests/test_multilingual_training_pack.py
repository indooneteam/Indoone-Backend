import json

from scripts import build_multilingual_training_pack as pack


def test_multilingual_pack_covers_all_target_languages(tmp_path, monkeypatch):
    instruction_path = tmp_path / "instructions.jsonl"
    multilingual_path = tmp_path / "multilingual.jsonl"
    corpus_path = tmp_path / "corpus.txt"
    monkeypatch.setattr(pack, "INSTRUCTION_PATH", instruction_path)
    monkeypatch.setattr(pack, "MULTILINGUAL_PATH", multilingual_path)
    monkeypatch.setattr(pack, "CORPUS_PATH", corpus_path)

    report = pack.build()
    assert report["generated_rows"] == 14_400
    assert report["multilingual_rows"] >= 14_400
    assert report["corpus_chars"] >= 1_000_000

    rows = [json.loads(line) for line in multilingual_path.read_text(encoding="utf-8").splitlines()]
    languages = {row["language"] for row in rows if row.get("source") == pack.SOURCE}
    assert languages == set(pack.LANGUAGES)


def test_multilingual_pack_is_idempotent(tmp_path, monkeypatch):
    instruction_path = tmp_path / "instructions.jsonl"
    multilingual_path = tmp_path / "multilingual.jsonl"
    corpus_path = tmp_path / "corpus.txt"
    monkeypatch.setattr(pack, "INSTRUCTION_PATH", instruction_path)
    monkeypatch.setattr(pack, "MULTILINGUAL_PATH", multilingual_path)
    monkeypatch.setattr(pack, "CORPUS_PATH", corpus_path)

    first = pack.build()
    first_instructions = instruction_path.read_text(encoding="utf-8")
    first_multilingual = multilingual_path.read_text(encoding="utf-8")

    second = pack.build()
    assert first == second
    assert instruction_path.read_text(encoding="utf-8") == first_instructions
    assert multilingual_path.read_text(encoding="utf-8") == first_multilingual
