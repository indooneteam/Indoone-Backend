from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn

from app.ai.behavior_eval import load_cases, score_case, summarize_gate
from app.ai.inference import LocalModelRuntime
from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer
from app.ai.training.evaluate import evaluate_checkpoint
from app.ai.training.model_registry import ModelRecord, active_record, promote_candidate
from app.ai.training.train import (
    _instruction_batchify,
    _prepare_instruction_examples,
    evaluate_instruction_loss,
)
from app.ai.training.training_data import TrainingExample, load_examples
from app.ai.training.train import _file_fingerprint


DEFAULT_MODEL_DIR = Path("models/indoone-small")
DEFAULT_CANDIDATE_DIR = Path("models/indoone-email-capability-candidate")
DEFAULT_SAFETY_TRAIN = Path("data/raw/indoone_email_safety_examples.jsonl")
DEFAULT_ACTION_TRAIN = Path("data/raw/indoone_email_actions_examples.jsonl")
DEFAULT_SAFETY_EXTRA = Path("data/raw/indoone_email_safety_additional_generated.jsonl")
DEFAULT_ACTION_EXTRA = Path("data/raw/indoone_email_actions_additional_generated.jsonl")
DEFAULT_EXPANSION_GENERATOR = Path("scripts/generate_email_training_expansion.py")
DEFAULT_SAFETY_VALIDATION = Path("data/eval/email_safety_validation.jsonl")
DEFAULT_SAFETY_TEST = Path("data/eval/email_safety_test.jsonl")
DEFAULT_ANCHOR = Path("data/raw/core_instruction_seed.jsonl")
DEFAULT_GENERAL_EVAL = Path("data/raw/indoone_corpus.txt")
DEFAULT_REGISTRY = Path("models/registry.json")

BENCHMARK_STEPS = 20
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
EVAL_INTERVAL = 50
TRAIN_BUDGET_SECONDS = 55 * 60
HARD_MAX_RUNTIME_SECONDS = 60 * 60
SAFETY_WEIGHT = 0.80
ACTION_WEIGHT = 0.10
ANCHOR_WEIGHT = 0.10
MAX_PLANNED_STEPS = 4000
MIN_EMAIL_ACCURACY = 0.75
MIN_EMAIL_ACCURACY_GAIN = 0.10


def _sha256_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _load_model(model_dir: Path) -> tuple[IndooneTransformer, BPETokenizer]:
    checkpoint = torch.load(
        model_dir / "indoone-small.pt",
        map_location="cpu",
        weights_only=False,
    )
    tokenizer = BPETokenizer.load(model_dir / "tokenizer.json")
    config = dict(checkpoint["config"])
    config.pop("model_version", None)
    model = IndooneTransformer(vocab_size=tokenizer.vocab_size, **config)
    model.load_state_dict(checkpoint["model_state"])
    return model, tokenizer


def _merge_pool(
    safety: list[TrainingExample],
    actions: list[TrainingExample],
    anchors: list[TrainingExample],
) -> tuple[list[TrainingExample], list[float]]:
    pools = [
        (safety, SAFETY_WEIGHT),
        (actions, ACTION_WEIGHT),
        (anchors, ANCHOR_WEIGHT),
    ]
    merged: list[TrainingExample] = []
    weights: list[float] = []
    seen: set[tuple[str, str]] = set()

    for pool, target in pools:
        unique: list[TrainingExample] = []
        for example in pool:
            key = (example.instruction.casefold(), example.response.casefold())
            if key in seen:
                continue
            seen.add(key)
            unique.append(example)
        if not unique:
            continue
        merged.extend(unique)
        weights.extend([target / len(unique)] * len(unique))

    if not merged:
        raise ValueError("email incremental training pool is empty")

    total = sum(weights)
    return merged, [value / total for value in weights]


def _benchmark(
    model_dir: Path,
    pool: list[TrainingExample],
    sampling_weights: list[float],
    device: str,
) -> float:
    model, tokenizer = _load_model(model_dir)
    model = model.to(device)
    prepared = _prepare_instruction_examples(pool, tokenizer, model.block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(4242)

    started = time.monotonic()
    model.train()
    for _ in range(BENCHMARK_STEPS):
        x, y = _instruction_batchify(
            pool,
            tokenizer,
            model.block_size,
            BATCH_SIZE,
            "cpu",
            generator,
            sampling_weights=sampling_weights,
            prepared_examples=prepared,
        )
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    elapsed = time.monotonic() - started
    speed = BENCHMARK_STEPS / elapsed if elapsed > 0 else 0.0
    print(
        json.dumps(
            {
                "benchmark_steps": BENCHMARK_STEPS,
                "elapsed_seconds": round(elapsed, 2),
                "steps_per_second": round(speed, 4),
                "training_budget_seconds": TRAIN_BUDGET_SECONDS,
                "planned_steps": min(
                    MAX_PLANNED_STEPS,
                    max(1, int(speed * TRAIN_BUDGET_SECONDS * 0.95)),
                ),
            },
            indent=2,
        ),
        flush=True,
    )
    return speed


def _train(
    model: IndooneTransformer,
    tokenizer: BPETokenizer,
    pool: list[TrainingExample],
    sampling_weights: list[float],
    validation: list[TrainingExample],
    steps: int,
) -> dict[str, object]:
    model = model.to("cpu")
    prepared = _prepare_instruction_examples(pool, tokenizer, model.block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(4242)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    started = time.monotonic()

    model.train()
    for step in range(1, steps + 1):
        elapsed = time.monotonic() - started
        if elapsed >= TRAIN_BUDGET_SECONDS:
            print("training budget reached; stopping at last completed step", flush=True)
            break

        x, y = _instruction_batchify(
            pool,
            tokenizer,
            model.block_size,
            BATCH_SIZE,
            "cpu",
            generator,
            sampling_weights=sampling_weights,
            prepared_examples=prepared,
        )
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % EVAL_INTERVAL == 0 or step == steps:
            validation_loss = evaluate_instruction_loss(
                model,
                validation,
                tokenizer,
                model.block_size,
                BATCH_SIZE,
                "cpu",
            )
            print(
                f"incremental step {step}/{steps} | train_loss={float(loss):.4f} | "
                f"email_validation_loss={validation_loss}",
                flush=True,
            )
            if validation_loss is not None and validation_loss < best_loss:
                best_loss = float(validation_loss)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("no validated incremental checkpoint was produced")

    model.load_state_dict(best_state)
    return {
        "best_email_validation_loss": best_loss,
        "elapsed_seconds": time.monotonic() - started,
        "completed_steps": step,
    }


def _classify_test(checkpoint: Path, tokenizer: Path, cases: list[TrainingExample]) -> dict[str, float | int]:
    runtime = LocalModelRuntime(checkpoint, tokenizer)
    correct = 0
    for case in cases:
        response = runtime.generate(
            case.instruction,
            max_new_tokens=96,
            temperature=0.0,
        )
        expected = case.response.casefold()
        found = next(
            (label for label in ("legitimate", "spam", "phishing") if label in expected),
            None,
        )
        predicted = next(
            (label for label in ("legitimate", "spam", "phishing") if label in response.casefold()),
            None,
        )
        correct += int(found is not None and predicted == found)
    accuracy = correct / len(cases) if cases else 0.0
    return {"correct": correct, "total": len(cases), "accuracy": accuracy}


def _behavior_gate(checkpoint: Path, tokenizer: Path) -> dict[str, object]:
    runtime = LocalModelRuntime(checkpoint, tokenizer)
    results = []
    for case in load_cases(Path("data/eval/behavior.jsonl")):
        response = runtime.generate(str(case["prompt"]), temperature=0.0)
        results.append(score_case(case, response))
    return summarize_gate(results)


def _write_initial_registry(path: Path, base_eval: dict[str, float | int | str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    revision = _sha256_prefix(DEFAULT_MODEL_DIR / "indoone-small.pt")
    baseline = ModelRecord(
        version="v1",
        model_dir=str(DEFAULT_MODEL_DIR),
        checkpoint=str(DEFAULT_MODEL_DIR / "indoone-small.pt"),
        tokenizer=str(DEFAULT_MODEL_DIR / "tokenizer.json"),
        loss=float(base_eval["loss"]),
        perplexity=float(base_eval["perplexity"]),
        status="active",
        behavioral_gate_passed=True,
        benchmark_version="v1",
        revision=revision,
        artifact_sha256=revision,
    )
    path.write_text(
        json.dumps(
            {
                "active_version": baseline.version,
                "active_revision": baseline.revision,
                "models": [baseline.__dict__],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-benchmark and incrementally train Indoone Email capability on the available device within a 55-minute training budget."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--safety-train", type=Path, default=DEFAULT_SAFETY_TRAIN)
    parser.add_argument("--action-train", type=Path, default=DEFAULT_ACTION_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_SAFETY_VALIDATION)
    parser.add_argument("--test", type=Path, default=DEFAULT_SAFETY_TEST)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--safety-extra", type=Path, default=DEFAULT_SAFETY_EXTRA)
    parser.add_argument("--action-extra", type=Path, default=DEFAULT_ACTION_EXTRA)
    parser.add_argument("--no-generate-extra", action="store_true")
    parser.add_argument("--general-eval", type=Path, default=DEFAULT_GENERAL_EVAL)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training_device: {device}", flush=True)
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    if not args.no_generate_extra and (not args.safety_extra.is_file() or not args.action_extra.is_file()):
        if not DEFAULT_EXPANSION_GENERATOR.is_file():
            raise SystemExit(f"email training expansion generator is missing: {DEFAULT_EXPANSION_GENERATOR}")
        subprocess.run([sys.executable, str(DEFAULT_EXPANSION_GENERATOR)], check=True)

    for required in (
        args.model_dir / "indoone-small.pt",
        args.model_dir / "tokenizer.json",
        args.safety_train,
        args.action_train,
        args.safety_extra,
        args.action_extra,
        args.validation,
        args.test,
        args.anchor,
        args.general_eval,
    ):
        if not required.is_file():
            raise SystemExit(f"required training file is missing: {required}")

    safety = load_examples(args.safety_train) + load_examples(args.safety_extra)
    actions = load_examples(args.action_train) + load_examples(args.action_extra)
    validation = load_examples(args.validation)
    test_cases = load_examples(args.test)
    anchors = load_examples(args.anchor)

    pool, sampling_weights = _merge_pool(safety, actions, anchors)
    print(
        json.dumps(
            {
                "safety_examples": len(safety),
                "action_examples": len(actions),
                "anchor_examples": len(anchors),
                "training_pool": len(pool),
                "sampling_policy": {
                    "email_safety": SAFETY_WEIGHT,
                    "email_actions": ACTION_WEIGHT,
                    "core_anchors": ANCHOR_WEIGHT,
                },
            },
            indent=2,
        ),
        flush=True,
    )

    speed = _benchmark(args.model_dir, pool, sampling_weights, device)
    planned_steps = min(
        MAX_PLANNED_STEPS,
        max(1, int(speed * TRAIN_BUDGET_SECONDS * 0.95)),
    )

    base_checkpoint = args.model_dir / "indoone-small.pt"
    base_tokenizer = args.model_dir / "tokenizer.json"
    base_eval = evaluate_checkpoint(
        base_checkpoint,
        base_tokenizer,
        args.general_eval,
        batch_size=BATCH_SIZE,
    )
    _write_initial_registry(args.registry, base_eval)

    model, tokenizer = _load_model(args.model_dir)
    training = _train(model, tokenizer, pool, sampling_weights, validation, planned_steps, device)

    args.candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_checkpoint = args.candidate_dir / "indoone-small.pt"
    candidate_tokenizer = args.candidate_dir / "tokenizer.json"
    torch.save(
        {
            "config": {**model.config(), "model_version": "indoone-gpt-v2"},
            "model_state": model.cpu().state_dict(),
        },
        candidate_checkpoint,
    )
    shutil.copy2(base_tokenizer, candidate_tokenizer)

    candidate_eval = evaluate_checkpoint(
        candidate_checkpoint,
        candidate_tokenizer,
        args.general_eval,
        batch_size=BATCH_SIZE,
    )
    base_email = _classify_test(base_checkpoint, base_tokenizer, test_cases)
    candidate_email = _classify_test(candidate_checkpoint, candidate_tokenizer, test_cases)
    gate = _behavior_gate(candidate_checkpoint, candidate_tokenizer)

    if candidate_email["accuracy"] < MIN_EMAIL_ACCURACY:
        raise SystemExit(
            f"candidate rejected: email accuracy {candidate_email['accuracy']:.3f} "
            f"is below {MIN_EMAIL_ACCURACY:.2f}"
        )
    if candidate_email["accuracy"] < base_email["accuracy"] + MIN_EMAIL_ACCURACY_GAIN:
        raise SystemExit(
            f"candidate rejected: email accuracy did not improve by at least "
            f"{MIN_EMAIL_ACCURACY_GAIN:.2f}"
        )
    if not bool(gate.get("overall_pass")):
        raise SystemExit("candidate rejected: core behavioral regression gate failed")
    if (
        candidate_eval["loss"] >= base_eval["loss"]
        or candidate_eval["perplexity"] >= base_eval["perplexity"]
    ):
        raise SystemExit(
            "candidate rejected: general loss/perplexity did not strictly improve"
        )

    current = active_record(args.registry)
    version = current.version if current is not None else "v1"
    baseline_revision = current.revision if current is not None else _sha256_prefix(base_checkpoint)
    candidate_revision = _sha256_prefix(candidate_checkpoint)

    candidate = ModelRecord(
        version=version,
        model_dir=str(args.model_dir),
        checkpoint=str(args.model_dir / "indoone-small.pt"),
        tokenizer=str(args.model_dir / "tokenizer.json"),
        loss=float(candidate_eval["loss"]),
        perplexity=float(candidate_eval["perplexity"]),
        status="candidate",
        behavioral_gate_passed=True,
        parent_version=version,
        benchmark_version=current.benchmark_version if current else "v1",
        revision=candidate_revision,
        artifact_sha256=candidate_revision,
    )

    backup_dir = args.candidate_dir / "base_backup"
    backup_dir.mkdir(exist_ok=True)
    for name in ("indoone-small.pt", "tokenizer.json", "metadata.json", "training_history.json"):
        source = args.model_dir / name
        if source.is_file():
            shutil.copy2(source, backup_dir / name)

    try:
        shutil.copy2(candidate_checkpoint, base_checkpoint)
        promoted = promote_candidate(args.registry, candidate)
    except Exception:
        for name in ("indoone-small.pt", "tokenizer.json", "metadata.json", "training_history.json"):
            backup = backup_dir / name
            target = args.model_dir / name
            if backup.is_file():
                shutil.copy2(backup, target)
        raise

    print(
        json.dumps(
            {
                "status": "promoted",
                "version": promoted.version,
                "revision": promoted.revision,
                "baseline_revision": baseline_revision,
                "base_email_accuracy": base_email,
                "candidate_email_accuracy": candidate_email,
                "behavioral_gate": gate,
                "base_evaluation": base_eval,
                "candidate_evaluation": candidate_eval,
                "training": training,
                "hard_max_runtime_seconds": HARD_MAX_RUNTIME_SECONDS,
                "production_model_slot": "models/indoone-small",
                "remote_release_upload": "required: upload the promoted indoone-small.pt to the Indoone-Model indoone-model-v1 release using the same asset name",
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
