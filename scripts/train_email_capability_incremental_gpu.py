from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch import nn

from app.ai.inference import LocalModelRuntime
from app.ai.training.evaluate import evaluate_checkpoint
from app.ai.training.model_registry import (
    ModelRecord,
    active_record,
    promote_candidate,
)
from app.ai.training.train import (
    _instruction_batchify,
    _prepare_instruction_examples,
)
from app.ai.training.training_data import load_examples
from app.ai.model import IndooneTransformer
from scripts.generate_email_training_expansion import build


TRAIN_BUDGET_SECONDS = 55 * 60
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_STEPS = 6000
EVAL_INTERVAL = 50
EARLY_STOP_PATIENCE = 10

EMAIL_ADAPTER_DIM = 192
EMAIL_ADAPTER_LAYERS = 6

MIN_EMAIL_ACCURACY = 0.75
MIN_EMAIL_ACCURACY_GAIN = 0.10

MODEL_DIR = Path("models/indoone-small")
CANDIDATE_DIR = Path("models/indoone-email-capability-candidate")
SAFETY_TRAIN = Path("data/raw/indoone_email_safety_examples.jsonl")
ACTION_TRAIN = Path("data/raw/indoone_email_actions_examples.jsonl")
SAFETY_EXTRA = Path("data/raw/indoone_email_safety_additional_generated.jsonl")
ACTION_EXTRA = Path("data/raw/indoone_email_actions_additional_generated.jsonl")
VALIDATION = Path("data/eval/email_safety_validation.jsonl")
TEST = Path("data/eval/email_safety_test.jsonl")
GENERAL_EVAL = Path("data/raw/indoone_corpus.txt")
REGISTRY = Path("models/registry.json")

PERSISTENT_RESUME = Path(
    "/content/drive/MyDrive/IndooneTraining/checkpoints/email_adapter_resume.pt"
)


def _sha256_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _load_checkpoint(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    config = dict(checkpoint["config"])
    state = checkpoint["model_state"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint model_state must be a dictionary")
    return config, state


def _load_base_model(model_dir: Path) -> tuple[IndooneTransformer, object, dict[str, torch.Tensor]]:
    checkpoint = torch.load(
        model_dir / "indoone-small.pt",
        map_location="cpu",
        weights_only=False,
    )
    tokenizer_path = model_dir / "tokenizer.json"

    from app.ai.tokenizer import BPETokenizer

    tokenizer = BPETokenizer.load(tokenizer_path)
    config = dict(checkpoint["config"])
    config.pop("model_version", None)
    config["email_adapter_dim"] = EMAIL_ADAPTER_DIM
    config["email_adapter_layers"] = EMAIL_ADAPTER_LAYERS

    model = IndooneTransformer(
        vocab_size=tokenizer.vocab_size,
        **config,
    )

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state"],
        strict=False,
    )

    unexpected_adapter_keys = [
        key for key in unexpected
        if not key.startswith("blocks.") or "email_adapter_" not in key
    ]
    if unexpected_adapter_keys:
        raise RuntimeError(
            f"unexpected base checkpoint keys: {unexpected_adapter_keys}"
        )

    expected_missing = {
        key
        for key in model.state_dict()
        if "email_adapter_" in key
    }
    if set(missing) != expected_missing:
        raise RuntimeError(
            "base checkpoint compatibility check failed: "
            f"missing={missing}, expected_adapter_keys={sorted(expected_missing)}"
        )

    base_state = {
        key: value.detach().cpu().clone()
        for key, value in checkpoint["model_state"].items()
    }
    return model, tokenizer, base_state


def _freeze_backbone(model: IndooneTransformer) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if "email_adapter_" in name:
            parameter.requires_grad = True
            trainable.append(parameter)

    if not trainable:
        raise RuntimeError("Email adapter parameters were not found.")

    return trainable


def _adapter_parameters(model: IndooneTransformer) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "email_adapter_" in name
    ]


@torch.inference_mode()
def _email_validation_loss(
    model: IndooneTransformer,
    validation,
    tokenizer,
    device: str,
) -> float:
    if not validation:
        raise ValueError("Email validation data is empty")

    model.eval()
    prepared = _prepare_instruction_examples(
        validation,
        tokenizer,
        model.block_size,
    )

    losses: list[float] = []
    generator = torch.Generator().manual_seed(20260930)

    # Evaluate deterministic batches covering the entire validation set.
    for start in range(0, len(validation), BATCH_SIZE):
        indexes = [
            (start + offset) % len(validation)
            for offset in range(BATCH_SIZE)
        ]
        x, y = _instruction_batchify(
            validation,
            tokenizer,
            model.block_size,
            BATCH_SIZE,
            device,
            generator,
            example_indices=indexes,
            prepared_examples=prepared,
        )
        _, loss = model(
            x,
            y,
            use_email_adapter=True,
        )
        if loss is None:
            raise RuntimeError("Email validation loss is None")
        losses.append(float(loss.detach().cpu()))

    return sum(losses) / len(losses)


def _adapter_norm(model: IndooneTransformer) -> float:
    total = 0.0
    for _, parameter in _adapter_parameters(model):
        total += float(torch.sum(parameter.detach().float() ** 2))
    return math.sqrt(total)


def _resume_path() -> Path | None:
    if PERSISTENT_RESUME.parent.is_dir():
        return PERSISTENT_RESUME
    return None


def _save_resume(
    path: Path,
    model: IndooneTransformer,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    baseline_revision: str,
    step: int,
    best_loss: float,
    best_state: dict[str, torch.Tensor] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format": 1,
            "baseline_revision": baseline_revision,
            "step": step,
            "best_loss": best_loss,
            "model_state": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "best_state": best_state,
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.get_state(),
        },
        temporary,
    )
    temporary.replace(path)


def _restore_resume(
    path: Path,
    model: IndooneTransformer,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    baseline_revision: str,
) -> tuple[int, float, dict[str, torch.Tensor] | None]:
    resume = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if resume.get("format") != 1:
        raise ValueError("unsupported resume checkpoint format")
    if resume.get("baseline_revision") != baseline_revision:
        raise ValueError("resume checkpoint belongs to a different base model")

    model.load_state_dict(resume["model_state"], strict=True)
    optimizer.load_state_dict(resume["optimizer_state"])

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to("cuda")

    generator.set_state(resume["generator_state"])

    return (
        int(resume["step"]),
        float(resume["best_loss"]),
        resume.get("best_state"),
    )


def _write_initial_registry(
    path: Path,
    base_eval: dict[str, float | int | str],
    base_checkpoint: Path,
) -> None:
    if path.exists():
        return

    revision = _sha256_prefix(base_checkpoint)
    baseline = ModelRecord(
        version="v1",
        model_dir=str(MODEL_DIR),
        checkpoint=str(base_checkpoint),
        tokenizer=str(MODEL_DIR / "tokenizer.json"),
        loss=float(base_eval["loss"]),
        perplexity=float(base_eval["perplexity"]),
        status="active",
        behavioral_gate_passed=True,
        benchmark_version="v1",
        revision=revision,
        artifact_sha256=revision,
        max_metric_regression=0.0,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "active_version": "v1",
                "active_revision": revision,
                "models": [baseline.__dict__],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def _assert_backbone_unchanged(
    base_checkpoint: Path,
    candidate_checkpoint: Path,
) -> None:
    _, base_state = _load_checkpoint(base_checkpoint)
    _, candidate_state = _load_checkpoint(candidate_checkpoint)

    missing = [
        key for key in base_state
        if key not in candidate_state
    ]
    if missing:
        raise RuntimeError(
            f"candidate is missing backbone parameters: {missing[:10]}"
        )

    changed = []
    for key, base_value in base_state.items():
        candidate_value = candidate_state[key]
        if not torch.equal(base_value, candidate_value):
            changed.append(key)

    if changed:
        raise RuntimeError(
            "backbone changed during Email adapter training: "
            f"{changed[:20]}"
        )


def _save_candidate(
    model: IndooneTransformer,
    tokenizer_path: Path,
) -> Path:
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CANDIDATE_DIR / "indoone-small.pt"
    torch.save(
        {
            "config": {
                **model.config(),
                "model_version": "indoone-gpt-v2",
            },
            "model_state": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
        },
        checkpoint,
    )
    shutil.copy2(tokenizer_path, CANDIDATE_DIR / "tokenizer.json")
    return checkpoint


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU is required. Select a Colab GPU runtime."
        )

    torch.set_float32_matmul_precision("high")

    print(
        f"training_device: cuda",
        flush=True,
    )
    print(
        f"gpu: {torch.cuda.get_device_name(0)}",
        flush=True,
    )

    # Generate the repo's deterministic expanded Email dataset.
    safety_extra, actions_extra = build()

    SAFETY_EXTRA.parent.mkdir(parents=True, exist_ok=True)
    SAFETY_EXTRA.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in safety_extra
        ),
        encoding="utf-8",
    )
    ACTION_EXTRA.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in actions_extra
        ),
        encoding="utf-8",
    )

    required = [
        MODEL_DIR / "indoone-small.pt",
        MODEL_DIR / "tokenizer.json",
        SAFETY_TRAIN,
        ACTION_TRAIN,
        SAFETY_EXTRA,
        ACTION_EXTRA,
        VALIDATION,
        TEST,
        GENERAL_EVAL,
    ]

    for path in required:
        if not path.is_file():
            raise SystemExit(f"required training file is missing: {path}")

    safety = (
        load_examples(SAFETY_TRAIN)
        + load_examples(SAFETY_EXTRA)
    )
    actions = (
        load_examples(ACTION_TRAIN)
        + load_examples(ACTION_EXTRA)
    )
    validation = load_examples(VALIDATION)
    test_cases = load_examples(TEST)

    pool = safety + actions
    if not pool:
        raise RuntimeError("Email-only training pool is empty.")

    sampling_weights = (
        [0.90 / len(safety)] * len(safety)
        + [0.10 / len(actions)] * len(actions)
    )
    total = sum(sampling_weights)
    sampling_weights = [
        value / total for value in sampling_weights
    ]

    base_checkpoint = MODEL_DIR / "indoone-small.pt"
    base_tokenizer = MODEL_DIR / "tokenizer.json"
    baseline_revision = _sha256_prefix(base_checkpoint)

    print(
        json.dumps(
            {
                "email_safety_examples": len(safety),
                "email_action_examples": len(actions),
                "total_email_examples": len(pool),
                "non_email_training_examples": 0,
                "adapter_dimension": EMAIL_ADAPTER_DIM,
                "adapter_layers": EMAIL_ADAPTER_LAYERS,
                "learning_rate": LEARNING_RATE,
                "max_steps": MAX_STEPS,
                "baseline_revision": baseline_revision,
            },
            indent=2,
        ),
        flush=True,
    )

    base_eval = evaluate_checkpoint(
        base_checkpoint,
        base_tokenizer,
        GENERAL_EVAL,
        batch_size=BATCH_SIZE,
    )
    base_email = LocalModelRuntime(
        base_checkpoint,
        base_tokenizer,
    )

    base_email_correct = 0
    for case in test_cases:
        response = base_email.generate(
            case.instruction,
            max_new_tokens=96,
            temperature=0.0,
            use_email_adapter=False,
        )
        expected = case.response.casefold()
        predicted = next(
            (
                label
                for label in ("legitimate", "spam", "phishing")
                if label in response.casefold()
            ),
            None,
        )
        expected_label = next(
            (
                label
                for label in ("legitimate", "spam", "phishing")
                if label in expected
            ),
            None,
        )
        base_email_correct += int(
            expected_label is not None
            and predicted == expected_label
        )

    base_email_result = {
        "correct": base_email_correct,
        "total": len(test_cases),
        "accuracy": (
            base_email_correct / len(test_cases)
            if test_cases
            else 0.0
        ),
    }

    model, tokenizer, _ = _load_base_model(MODEL_DIR)
    model = model.to("cuda")
    trainable = _freeze_backbone(model)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator().manual_seed(4242)

    prepared = _prepare_instruction_examples(
        pool,
        tokenizer,
        model.block_size,
    )

    start_step = 0
    best_loss = float("inf")
    best_state = None

    resume_path = _resume_path()
    if resume_path is not None and resume_path.is_file():
        try:
            start_step, best_loss, best_state = _restore_resume(
                resume_path,
                model,
                optimizer,
                generator,
                baseline_revision,
            )
            print(
                f"resuming Email adapter training from step {start_step}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"resume ignored: {exc}",
                flush=True,
            )

    started = time.monotonic()
    no_improvement = 0
    completed_steps = start_step

    model.train()

    for step in range(start_step + 1, MAX_STEPS + 1):
        if time.monotonic() - started >= TRAIN_BUDGET_SECONDS:
            print(
                "training budget reached; latest Drive checkpoint retained",
                flush=True,
            )
            break

        x, y = _instruction_batchify(
            pool,
            tokenizer,
            model.block_size,
            BATCH_SIZE,
            "cuda",
            generator,
            sampling_weights=sampling_weights,
            prepared_examples=prepared,
        )

        _, task_loss = model(
            x,
            y,
            use_email_adapter=True,
        )

        if task_loss is None:
            raise RuntimeError("training loss is None")

        optimizer.zero_grad(set_to_none=True)
        task_loss.backward()

        nn.utils.clip_grad_norm_(
            trainable,
            max_norm=1.0,
        )

        optimizer.step()

        completed_steps = step

        if (
            step == 1
            or step % EVAL_INTERVAL == 0
            or step == MAX_STEPS
        ):
            validation_loss = _email_validation_loss(
                model,
                validation,
                tokenizer,
                "cuda",
            )

            print(
                f"step {step}/{MAX_STEPS} | "
                f"email_train_loss={float(task_loss):.4f} | "
                f"email_validation_loss={validation_loss:.6f} | "
                f"adapter_norm={_adapter_norm(model):.6f}",
                flush=True,
            )

            if validation_loss < best_loss:
                best_loss = validation_loss
                no_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                no_improvement += 1

            if resume_path is not None:
                _save_resume(
                    resume_path,
                    model,
                    optimizer,
                    generator,
                    baseline_revision,
                    step,
                    best_loss,
                    best_state,
                )
                print(
                    f"Drive checkpoint saved at step {step}",
                    flush=True,
                )

            if no_improvement >= EARLY_STOP_PATIENCE:
                print(
                    "Email validation stopped improving; early stopping.",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError(
            "No validated Email adapter checkpoint was produced."
        )

    model.load_state_dict(best_state)
    adapter_norm = _adapter_norm(model)

    if adapter_norm <= 1e-8:
        raise RuntimeError(
            "Email adapter remained effectively untrained."
        )

    candidate_checkpoint = _save_candidate(
        model.cpu(),
        base_tokenizer,
    )
    candidate_tokenizer = CANDIDATE_DIR / "tokenizer.json"

    # HARD SAFETY ASSERTION: every original backbone tensor must be byte-for-byte
    # identical. Only new email_adapter_* tensors may differ.
    _assert_backbone_unchanged(
        base_checkpoint,
        candidate_checkpoint,
    )
    print(
        "backbone_unchanged: True",
        flush=True,
    )

    candidate_eval = evaluate_checkpoint(
        candidate_checkpoint,
        candidate_tokenizer,
        GENERAL_EVAL,
        batch_size=BATCH_SIZE,
    )

    candidate_runtime = LocalModelRuntime(
        candidate_checkpoint,
        candidate_tokenizer,
    )

    candidate_correct = 0
    for case in test_cases:
        response = candidate_runtime.generate(
            case.instruction,
            max_new_tokens=96,
            temperature=0.0,
            use_email_adapter=True,
        )

        expected = case.response.casefold()
        predicted = next(
            (
                label
                for label in ("legitimate", "spam", "phishing")
                if label in response.casefold()
            ),
            None,
        )
        expected_label = next(
            (
                label
                for label in ("legitimate", "spam", "phishing")
                if label in expected
            ),
            None,
        )

        candidate_correct += int(
            expected_label is not None
            and predicted == expected_label
        )

    candidate_email = {
        "correct": candidate_correct,
        "total": len(test_cases),
        "accuracy": (
            candidate_correct / len(test_cases)
            if test_cases
            else 0.0
        ),
    }

    behavior = []
    for case_path in [Path("data/eval/behavior.jsonl")]:
        if case_path.is_file():
            from app.ai.behavior_eval import load_cases, score_case

            for case in load_cases(case_path):
                response = candidate_runtime.generate(
                    str(case["prompt"]),
                    temperature=0.0,
                )
                behavior.append(
                    score_case(case, response)
                )

    behavior_pass = bool(behavior) and all(
        bool(result.get("passed"))
        for result in behavior
    )

    # General evaluation must remain exactly the same when the Email adapter is
    # disabled. The adapter is activated only by the email router.
    general_same = (
        candidate_eval["loss"] == base_eval["loss"]
        and candidate_eval["perplexity"] == base_eval["perplexity"]
    )

    email_accuracy_pass = (
        candidate_email["accuracy"] >= MIN_EMAIL_ACCURACY
    )
    email_gain_pass = (
        candidate_email["accuracy"]
        >= base_email_result["accuracy"]
        + MIN_EMAIL_ACCURACY_GAIN
    )

    all_gates = (
        email_accuracy_pass
        and email_gain_pass
        and behavior_pass
        and general_same
        and adapter_norm > 1e-8
    )

    print()
    print("==============================================")
    print("EMAIL ADAPTER FINAL GATES")
    print("==============================================")
    print("BASE EMAIL:", json.dumps(base_email_result))
    print("CANDIDATE EMAIL:", json.dumps(candidate_email))
    print("Email accuracy gate:", email_accuracy_pass)
    print("Email gain gate:", email_gain_pass)
    print("Existing behavior gate:", behavior_pass)
    print("General metrics unchanged:", general_same)
    print("Backbone unchanged: True")
    print("Adapter trained:", adapter_norm > 1e-8)
    print("ALL GATES:", all_gates)
    print("==============================================")

    if not all_gates:
        print()
        print("CANDIDATE REJECTED")
        print("OLD MODEL WAS NOT REPLACED.")
        print("Candidate:", candidate_checkpoint)
        print("Baseline:", base_checkpoint)
        print("Persistent resume:", resume_path)
        raise SystemExit(
            "Email adapter candidate failed one or more promotion gates."
        )

    _write_initial_registry(
        REGISTRY,
        base_eval,
        base_checkpoint,
    )

    current = active_record(REGISTRY)
    version = current.version if current is not None else "v1"
    candidate_revision = _sha256_prefix(candidate_checkpoint)

    candidate_record = ModelRecord(
        version=version,
        model_dir=str(MODEL_DIR),
        checkpoint=str(MODEL_DIR / "indoone-small.pt"),
        tokenizer=str(MODEL_DIR / "tokenizer.json"),
        loss=float(candidate_eval["loss"]),
        perplexity=float(candidate_eval["perplexity"]),
        status="candidate",
        behavioral_gate_passed=True,
        parent_version=version if current is not None else None,
        benchmark_version=current.benchmark_version if current else "v1",
        revision=candidate_revision,
        artifact_sha256=candidate_revision,
        max_metric_regression=0.0,
    )

    production_backup = CANDIDATE_DIR / "production_backup"
    production_backup.mkdir(parents=True, exist_ok=True)

    for name in ("indoone-small.pt", "tokenizer.json"):
        source = MODEL_DIR / name
        if source.is_file():
            shutil.copy2(
                source,
                production_backup / name,
            )

    try:
        shutil.copy2(
            candidate_checkpoint,
            base_checkpoint,
        )
        shutil.copy2(
            candidate_tokenizer,
            base_tokenizer,
        )
        promoted = promote_candidate(
            REGISTRY,
            candidate_record,
        )
    except Exception:
        for name in ("indoone-small.pt", "tokenizer.json"):
            backup = production_backup / name
            if backup.is_file():
                shutil.copy2(
                    backup,
                    MODEL_DIR / name,
                )
        raise

    if resume_path is not None and resume_path.is_file():
        resume_path.unlink()

    promoted_copy = Path(
        "/content/drive/MyDrive/IndooneTraining/outputs"
    ) / "email_adapter_promoted"
    promoted_copy.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        base_checkpoint,
        promoted_copy / "indoone-small.pt",
    )
    shutil.copy2(
        base_tokenizer,
        promoted_copy / "tokenizer.json",
    )

    print()
    print("================================================")
    print("✅ EMAIL CAPABILITY UPDATE PROMOTED")
    print("================================================")
    print("Promotion status:", promoted.status)
    print("Version:", promoted.version)
    print("Revision:", promoted.revision)
    print("Email accuracy:", candidate_email)
    print("Backbone unchanged: True")
    print("Old model backup:", production_backup)
    print("Promoted Drive copy:", promoted_copy)
    print("Completed steps:", completed_steps)
    print("================================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
