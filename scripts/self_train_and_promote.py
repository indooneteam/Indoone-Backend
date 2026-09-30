from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.model_registry import ModelRecord, active_record, promote_candidate
from app.storage.b2 import B2Storage
from app.storage.github_release import GitHubReleaseStorage, get_github_release_storage


DEFAULT_STATE = Path("data/self_update/state.json")
DEFAULT_TRAINING_STATE = Path("data/self_update/training_state.json")
DEFAULT_MODEL_DIR = Path("models/indoone-small")
DEFAULT_REGISTRY = Path("models/registry.json")
DEFAULT_BEHAVIOR_CASES = Path("data/eval/behavior.jsonl")
DEFAULT_MIN_NEW_ITEMS = 20
DEFAULT_STEPS = 500
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_EVAL_INTERVAL = 100
DEFAULT_SEED = 4242


def _load_json(path: Path, default: dict[str, object]) -> dict[str, object]:
    if not path.is_file():
        return dict(default)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_collected_items(path: Path) -> list[dict[str, str]]:
    payload = _load_json(path, {"items": []})
    raw_items = payload.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("self-update state items must be a list")

    items: list[dict[str, str]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        fingerprint = str(raw.get("fingerprint", "")).strip()
        if not fingerprint:
            continue
        items.append(
            {
                "fingerprint": fingerprint,
                "source": str(raw.get("source", "")).strip(),
                "title": str(raw.get("title", "")).strip(),
                "url": str(raw.get("url", "")).strip(),
                "summary": str(raw.get("summary", "")).strip(),
                "published": str(raw.get("published", "")).strip(),
            }
        )
    return items


def _load_training_state(path: Path) -> dict[str, object]:
    payload = _load_json(
        path,
        {
            "schema_version": 1,
            "trained_fingerprints": [],
            "last_promoted_version": None,
        },
    )
    fingerprints = payload.get("trained_fingerprints", [])
    if not isinstance(fingerprints, list) or not all(
        isinstance(item, str) for item in fingerprints
    ):
        raise ValueError("trained_fingerprints must be a string list")
    return payload


def _new_items(
    items: list[dict[str, str]],
    training_state: dict[str, object],
) -> list[dict[str, str]]:
    trained = {
        str(value)
        for value in training_state.get("trained_fingerprints", [])
    }
    return [item for item in items if item["fingerprint"] not in trained]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=cwd)


def _evaluate(
    checkpoint: Path,
    tokenizer: Path,
    *,
    output: Path,
) -> dict[str, object]:
    _run(
        [
            sys.executable,
            "-m",
            "app.ai.evaluate",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer),
            "--corpus",
            "data/processed/test.txt",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation report must be a JSON object")
    return payload


def _behavior_gate(
    checkpoint: Path,
    tokenizer: Path,
    behavior_cases: Path,
    output: Path,
) -> dict[str, object]:
    _run(
        [
            sys.executable,
            "-m",
            "app.ai.behavior_eval",
            "--cases",
            str(behavior_cases),
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer),
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("behavior evaluation report must be a JSON object")
    return payload


def _ensure_training_data() -> None:
    _run([sys.executable, "scripts/build_multilingual_training_pack.py"])
    _run(
        [
            sys.executable,
            "scripts/self_update_knowledge.py",
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/merge_auto_web_training.py",
            "--state",
            str(DEFAULT_STATE),
            "--output",
            "data/processed/generated_multilingual_examples.jsonl",
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "scripts.prepare_dataset",
            "--source",
            "data/raw/indoone_corpus.txt",
            "--output-dir",
            "data/processed",
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/assemble_training_dataset.py",
            "--source",
            "data/raw/indoone_instructions.jsonl",
            "--source",
            "data/raw/core_instruction_seed.jsonl",
            "--source",
            "data/raw/indoone_multilingual_examples.jsonl",
            "--output",
            "data/processed/curated_instructions.jsonl",
            "--validation-output",
            "data/processed/instructions_validation.jsonl",
            "--validation-ratio",
            "0.2",
            "--seed",
            str(DEFAULT_SEED),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/validate_dataset_quality.py",
            "--source",
            "data/processed/curated_instructions.jsonl",
            "--source",
            "data/processed/instructions_validation.jsonl",
            "--source",
            "data/processed/generated_multilingual_examples.jsonl",
            "--source",
            "data/raw/indoone_phone_contacts_examples.jsonl",
        ]
    )
    _run([sys.executable, "scripts/validate_training_manifest.py"])
    _run([sys.executable, "scripts/validate_final_training_recipe.py"])


def _model_storage() -> tuple[str, object] | None:
    github_release = get_github_release_storage()
    if github_release is not None:
        return "github_release", github_release
    if B2Storage.configured():
        return "b2", B2Storage()
    return None


def _download_active_model(storage: object, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("indoone-small.pt", "tokenizer.json"):
        if isinstance(storage, B2Storage):
            storage.download_file(
                f"models/indoone-small/{filename}",
                target / filename,
            )
        elif isinstance(storage, GitHubReleaseStorage):
            storage.download_file(filename, target / filename)
        else:
            raise TypeError("unsupported model storage")

    metadata_path = target / "metadata.json"
    try:
        if isinstance(storage, B2Storage):
            storage.download_file(
                "models/indoone-small/metadata.json",
                metadata_path,
            )
        else:
            storage.download_file("metadata.json", metadata_path)
    except Exception:
        metadata_path.write_text("{}\n", encoding="utf-8")


def _upload_candidate(
    storage_name: str,
    storage: object,
    candidate: Path,
    backup: Path,
) -> None:
    filenames = ["indoone-small.pt", "tokenizer.json"]
    if (candidate / "metadata.json").is_file():
        filenames.append("metadata.json")
    uploaded: list[str] = []

    def upload(filename: str, source: Path) -> None:
        if isinstance(storage, B2Storage):
            storage.upload_file(
                source,
                f"models/indoone-small/{filename}",
            )
            return
        if isinstance(storage, GitHubReleaseStorage):
            _run(
                [
                    "gh",
                    "release",
                    "upload",
                    storage.release_tag,
                    str(source),
                    "--repo",
                    storage.repository,
                    "--clobber",
                ]
            )
            return
        raise TypeError("unsupported model storage")

    def rollback(filename: str) -> None:
        source = backup / filename
        if not source.is_file():
            return
        if isinstance(storage, B2Storage):
            storage.upload_file(
                source,
                f"models/indoone-small/{filename}",
            )
        elif isinstance(storage, GitHubReleaseStorage):
            _run(
                [
                    "gh",
                    "release",
                    "upload",
                    storage.release_tag,
                    str(source),
                    "--repo",
                    storage.repository,
                    "--clobber",
                ]
            )

    try:
        for filename in filenames:
            upload(filename, candidate / filename)
            uploaded.append(filename)
    except Exception:
        for filename in uploaded:
            try:
                rollback(filename)
            except Exception as rollback_error:
                print(
                    f"ROLLBACK FAILED for {filename}: {rollback_error}",
                    file=sys.stderr,
                )
        raise


def _initialize_registry(
    registry_path: Path,
    baseline_version: str,
    baseline: dict[str, object],
) -> ModelRecord:
    current = active_record(registry_path)
    if current is not None:
        return current

    model = ModelRecord(
        version=baseline_version,
        model_dir=str(DEFAULT_MODEL_DIR),
        checkpoint=str(DEFAULT_MODEL_DIR / "indoone-small.pt"),
        tokenizer=str(DEFAULT_MODEL_DIR / "tokenizer.json"),
        loss=float(baseline["loss"]),
        perplexity=float(baseline["perplexity"]),
        status="active",
        behavioral_gate_passed=True,
        benchmark_version="v1",
    )
    _write_json(
        registry_path,
        {
            "active_version": model.version,
            "models": [model.__dict__],
        },
    )
    return model


def run_self_training(
    *,
    state_path: Path = DEFAULT_STATE,
    training_state_path: Path = DEFAULT_TRAINING_STATE,
    registry_path: Path = DEFAULT_REGISTRY,
    behavior_cases: Path = DEFAULT_BEHAVIOR_CASES,
    min_new_items: int = DEFAULT_MIN_NEW_ITEMS,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    eval_interval: int = DEFAULT_EVAL_INTERVAL,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    if min_new_items < 1:
        raise ValueError("min_new_items must be positive")
    if steps < 1 or batch_size < 1 or learning_rate <= 0 or eval_interval < 1:
        raise ValueError("training parameters must be positive")

    items = _load_collected_items(state_path)
    training_state = _load_training_state(training_state_path)
    pending = _new_items(items, training_state)

    if len(pending) < min_new_items:
        return {
            "status": "deferred",
            "reason": "not_enough_new_items",
            "new_items": len(pending),
            "required_items": min_new_items,
        }

    storage_selection = _model_storage()
    if storage_selection is None:
        return {
            "status": "deferred",
            "reason": "model_storage_not_configured",
            "new_items": len(pending),
        }

    if shutil.which("nvidia-smi") is None and os.getenv(
        "INDOONE_ALLOW_CPU_TRAINING", "false"
    ).strip().lower() not in {"1", "true", "yes", "on"}:
        return {
            "status": "deferred",
            "reason": "gpu_required",
            "new_items": len(pending),
        }

    _ensure_training_data()

    version_seed = (
        datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        + "-"
        + hashlib.sha256(
            "".join(item["fingerprint"] for item in pending).encode("utf-8")
        ).hexdigest()[:12]
    )
    version = f"web-{version_seed}"

    storage_name, storage = storage_selection
    with tempfile.TemporaryDirectory(prefix="indoone-self-train-") as tmp:
        root = Path(tmp)
        baseline_dir = root / "baseline"
        candidate_dir = root / "candidate"
        _download_active_model(storage, baseline_dir)

        baseline_eval_path = root / "baseline_evaluation.json"
        baseline_behavior_path = root / "baseline_behavior.json"
        baseline_eval = _evaluate(
            baseline_dir / "indoone-small.pt",
            baseline_dir / "tokenizer.json",
            output=baseline_eval_path,
        )
        baseline_behavior = _behavior_gate(
            baseline_dir / "indoone-small.pt",
            baseline_dir / "tokenizer.json",
            behavior_cases,
            baseline_behavior_path,
        )
        baseline_gate = baseline_behavior.get("gate", {})
        if not isinstance(baseline_gate, dict) or not bool(
            baseline_gate.get("overall_pass")
        ):
            raise SystemExit("active model failed baseline behavioral gate")

        checkpoint_digest = hashlib.sha256()
        with (baseline_dir / "indoone-small.pt").open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                checkpoint_digest.update(chunk)
        baseline_version = f"baseline-{checkpoint_digest.hexdigest()[:12]}"
        current = _initialize_registry(
            registry_path,
            baseline_version=baseline_version,
            baseline=baseline_eval,
        )

        shutil.copytree(baseline_dir, candidate_dir)
        _run(
            [
                sys.executable,
                "scripts/final_sft.py",
                "--model-dir",
                str(candidate_dir),
                "--train-instructions",
                "data/processed/curated_instructions.jsonl",
                "--generated-instructions",
                "data/processed/generated_multilingual_examples.jsonl",
                "--validation-instructions",
                "data/processed/instructions_validation.jsonl",
                "--capability-instructions",
                "data/raw/indoone_phone_contacts_examples.jsonl",
                "--steps",
                str(steps),
                "--batch-size",
                str(batch_size),
                "--learning-rate",
                str(learning_rate),
                "--eval-interval",
                str(eval_interval),
                "--seed",
                str(seed),
            ]
        )

        candidate_eval_path = root / "candidate_evaluation.json"
        candidate_behavior_path = root / "candidate_behavior.json"
        candidate_eval = _evaluate(
            candidate_dir / "indoone-small.pt",
            candidate_dir / "tokenizer.json",
            output=candidate_eval_path,
        )
        candidate_behavior = _behavior_gate(
            candidate_dir / "indoone-small.pt",
            candidate_dir / "tokenizer.json",
            behavior_cases,
            candidate_behavior_path,
        )
        candidate_gate = candidate_behavior.get("gate", {})
        gate_passed = (
            isinstance(candidate_gate, dict)
            and bool(candidate_gate.get("overall_pass"))
        )

        candidate_record = ModelRecord(
            version=version,
            model_dir=str(DEFAULT_MODEL_DIR),
            checkpoint=str(DEFAULT_MODEL_DIR / "indoone-small.pt"),
            tokenizer=str(DEFAULT_MODEL_DIR / "tokenizer.json"),
            loss=float(candidate_eval["loss"]),
            perplexity=float(candidate_eval["perplexity"]),
            status="candidate",
            behavioral_gate_passed=gate_passed,
            parent_version=current.version,
            benchmark_version=current.benchmark_version,
        )

        if not gate_passed:
            return {
                "status": "rejected",
                "reason": "behavioral_gate_failed",
                "version": version,
                "new_items": len(pending),
            }

        if not (
            candidate_record.loss < current.loss
            and candidate_record.perplexity < current.perplexity
        ):
            return {
                "status": "rejected",
                "reason": "candidate_did_not_improve_metrics",
                "version": version,
                "candidate_loss": candidate_record.loss,
                "current_loss": current.loss,
                "candidate_perplexity": candidate_record.perplexity,
                "current_perplexity": current.perplexity,
                "new_items": len(pending),
            }

        promote_candidate(registry_path, candidate_record)
        _upload_candidate(storage_name, storage, candidate_dir, baseline_dir)

        training_state["schema_version"] = 1
        training_state["trained_fingerprints"] = sorted(
            {
                *{
                    str(value)
                    for value in training_state.get("trained_fingerprints", [])
                },
                *(item["fingerprint"] for item in pending),
            }
        )
        training_state["last_promoted_version"] = version
        training_state["last_promoted_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        _write_json(training_state_path, training_state)

        report = {
            "status": "promoted",
            "version": version,
            "new_items": len(pending),
            "baseline": baseline_eval,
            "candidate": candidate_eval,
            "behavioral_gate": candidate_gate,
        }
        _write_json(
            Path("data/self_update/last_promotion.json"),
            report,
        )
        return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely fine-tune and promote the Indoone model from new trusted web knowledge."
    )
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--training-state",
        type=Path,
        default=DEFAULT_TRAINING_STATE,
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--behavior-cases", type=Path, default=DEFAULT_BEHAVIOR_CASES)
    parser.add_argument("--min-new-items", type=int, default=DEFAULT_MIN_NEW_ITEMS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    result = run_self_training(
        state_path=args.state,
        training_state_path=args.training_state,
        registry_path=args.registry,
        behavior_cases=args.behavior_cases,
        min_new_items=args.min_new_items,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
