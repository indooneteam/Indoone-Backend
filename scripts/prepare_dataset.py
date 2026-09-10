from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_documents(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]


def fingerprint(text: str) -> str:
    normalized = normalize_text(text).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def prepare_documents(documents: list[str], min_chars: int = 40) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for document in documents:
        value = normalize_text(document)
        if len(value) < min_chars:
            continue
        digest = fingerprint(value)
        if digest in seen:
            continue
        seen.add(digest)
        cleaned.append(value)
    return cleaned


def validate_split_quality(train: list[str], validation: list[str], test: list[str]) -> dict[str, int]:
    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    seen: dict[str, str] = {}
    for split_name, documents in splits.items():
        for document in documents:
            digest = fingerprint(document)
            previous = seen.get(digest)
            if previous is not None:
                raise ValueError(f"duplicate document across dataset splits: {previous} and {split_name}")
            seen[digest] = split_name

    return {
        "train_documents": len(train),
        "validation_documents": len(validation),
        "test_documents": len(test),
        "total_characters": sum(len(document) for documents in splits.values() for document in documents),
    }


def write_split(documents: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(documents) + "\n", encoding="utf-8")


def prepare_dataset(source: Path, output_dir: Path, train_ratio: float, validation_ratio: float) -> dict[str, int]:
    documents = prepare_documents(split_documents(source.read_text(encoding="utf-8")))
    if len(documents) < 3:
        raise ValueError("dataset needs at least 3 usable unique documents")

    train_end = max(1, int(len(documents) * train_ratio))
    validation_end = train_end + max(1, int(len(documents) * validation_ratio))
    if validation_end >= len(documents):
        validation_end = len(documents) - 1

    train = documents[:train_end]
    validation = documents[train_end:validation_end]
    test = documents[validation_end:]
    quality = validate_split_quality(train, validation, test)

    write_split(train, output_dir / "train.txt")
    write_split(validation, output_dir / "validation.txt")
    write_split(test, output_dir / "test.txt")
    return {
        "documents": len(documents),
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "total_characters": quality["total_characters"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, deduplicate, split, and validate an Indoone text dataset")
    parser.add_argument("--source", type=Path, default=Path("data/raw/indoone_corpus.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1 or not 0 < args.validation_ratio < 1:
        raise ValueError("split ratios must be between 0 and 1")
    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")
    stats = prepare_dataset(args.source, args.output_dir, args.train_ratio, args.validation_ratio)
    print(stats)


if __name__ == "__main__":
    main()
