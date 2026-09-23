from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from app.ai.tokenizer import BPETokenizer
from app.ai.train import _instruction_batchify, _merge_instruction_sets_weighted
from app.ai.training_data import load_examples
import torch


RAW_FILES = (
    Path("data/raw/indoone_corpus.txt"),
    Path("data/raw/indoone_instructions.jsonl"),
    Path("data/raw/core_instruction_seed.jsonl"),
    Path("data/raw/indoone_multilingual_examples.jsonl"),
    Path("data/raw/indoone_phone_contacts_examples.jsonl"),
)
GENERATED_FILE = Path("data/processed/generated_multilingual_examples.jsonl")
GENERATED_CORPUS = Path("data/processed/generated_multilingual_corpus.txt")
CURATED_FILE = Path("data/processed/curated_instructions.jsonl")
EVAL_FILE = Path("data/eval/behavior.jsonl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_clean_raw_files() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *(str(path) for path in RAW_FILES)],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        raise SystemExit(f"raw training sources were modified: {result.stdout.strip()}")


def validate() -> dict[str, object]:
    _git_clean_raw_files()

    required = [GENERATED_FILE, GENERATED_CORPUS, CURATED_FILE, EVAL_FILE]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing final-training preflight artifact(s): " + ", ".join(missing))

    generated = load_examples(GENERATED_FILE)
    curated = load_examples(CURATED_FILE)
    if len(generated) < 10_000:
        raise SystemExit(f"generated instruction pool too small: {len(generated)}")
    if len(curated) < 100:
        raise SystemExit(f"curated instruction pool too small: {len(curated)}")
    if len(GENERATED_CORPUS.read_text(encoding="utf-8")) < 1_000_000:
        raise SystemExit("generated multilingual readiness corpus is unexpectedly small")

    examples, fingerprints, weights, policy = _merge_instruction_sets_weighted(
        [CURATED_FILE, GENERATED_FILE]
    )
    if not examples:
        raise SystemExit("final instruction pool is empty")
    if len(weights) != len(examples):
        raise SystemExit("final instruction sampling weights do not match examples")

    curated_policy = policy.get(CURATED_FILE.name)
    generated_policy = policy.get(GENERATED_FILE.name)
    if not curated_policy or not generated_policy:
        raise SystemExit("final instruction pool policy is incomplete")
    if abs(float(curated_policy["target_probability"]) - 0.70) > 1e-9:
        raise SystemExit("curated instruction sampling target changed unexpectedly")
    if abs(float(generated_policy["target_probability"]) - 0.25) > 1e-9:
        raise SystemExit("generated instruction sampling target changed unexpectedly")

    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 1e-5:
        raise SystemExit(f"instruction sampling weights must sum to 1, got {total_weight}")

    tokenizer_text = " ".join(
        [
            CURATED_FILE.read_text(encoding="utf-8"),
            GENERATED_FILE.read_text(encoding="utf-8"),
        ]
    )
    tokenizer = BPETokenizer.train(
        tokenizer_text,
        vocab_size=1024,
        min_frequency=1,
    )
    x, y = _instruction_batchify(
        examples[:16],
        tokenizer,
        block_size=128,
        batch_size=4,
        device="cpu",
        generator=torch.Generator().manual_seed(42),
        sampling_weights=weights[:16],
    )
    if x.shape != y.shape:
        raise SystemExit("instruction input/target shapes differ")
    if not bool((y == -100).any()):
        raise SystemExit("response-only loss mask is missing")
    if not bool((y != -100).any()):
        raise SystemExit("response-only loss mask suppresses every target")

    eval_cases = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trained_prompts = {
        str(row["instruction"]).strip().casefold()
        for row in curated
    }
    exact_eval_overlap = [
        str(case["id"])
        for case in eval_cases
        if str(case.get("prompt", "")).strip().casefold() in trained_prompts
    ]
    if exact_eval_overlap:
        raise SystemExit("evaluation prompts leaked into curated training data: " + ", ".join(exact_eval_overlap))

    hashes = {str(path): _sha256(path) for path in RAW_FILES}
    _git_clean_raw_files()

    return {
        "ready": True,
        "raw_sha256": hashes,
        "curated_examples": len(curated),
        "generated_examples": len(generated),
        "generated_corpus_characters": len(GENERATED_CORPUS.read_text(encoding="utf-8")),
        "instruction_sampling_policy": policy,
        "evaluation_cases": len(eval_cases),
        "exact_evaluation_overlap": exact_eval_overlap,
        "response_only_loss_mask": True,
    }


def main() -> int:
    print(json.dumps(validate(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
