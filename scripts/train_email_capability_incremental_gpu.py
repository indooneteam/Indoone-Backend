from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from torch import nn

from app.ai.inference import LocalModelRuntime
from app.ai.training.evaluate import evaluate_checkpoint
from app.ai.training.model_registry import ModelRecord, active_record, promote_candidate
from app.ai.training.train import _instruction_batchify, _prepare_instruction_examples, evaluate_instruction_loss
from app.ai.training.training_data import load_examples
from scripts.generate_email_training_expansion import build
from scripts.train_email_capability_incremental import (
    HARD_MAX_RUNTIME_SECONDS,
    MAX_PLANNED_STEPS,
    MIN_EMAIL_ACCURACY,
    MIN_EMAIL_ACCURACY_GAIN,
    SAFETY_WEIGHT,
    ACTION_WEIGHT,
    ANCHOR_WEIGHT,
    _behavior_gate,
    _classify_test,
    _load_model,
    _merge_pool,
    _sha256_prefix,
    _write_initial_registry,
)

TRAIN_BUDGET_SECONDS = 55 * 60
BENCHMARK_STEPS = 20
BATCH_SIZE = 8
LEARNING_RATE = 8e-6
WEIGHT_DECAY = 0.01
EVAL_INTERVAL = 25
MAX_PLANNED_STEPS = 1200

# Email capability is trained with a frozen backbone to reduce catastrophic
# forgetting. Only the final transformer blocks and final layer norm adapt.
TRAINABLE_LAST_BLOCKS = 2
ANCHOR_LAMBDA = 0.25
MAX_WEIGHT_DELTA = 0.003
EARLY_STOP_PATIENCE = 6
MAX_GENERAL_REGRESSION = 0.02

MODEL_DIR = Path("models/indoone-small")
CANDIDATE_DIR = Path("models/indoone-email-capability-candidate")
SAFETY_TRAIN = Path("data/raw/indoone_email_safety_examples.jsonl")
ACTION_TRAIN = Path("data/raw/indoone_email_actions_examples.jsonl")
SAFETY_EXTRA = Path("data/raw/indoone_email_safety_additional_generated.jsonl")
ACTION_EXTRA = Path("data/raw/indoone_email_actions_additional_generated.jsonl")
VALIDATION = Path("data/eval/email_safety_validation.jsonl")
TEST = Path("data/eval/email_safety_test.jsonl")
ANCHOR = Path("data/raw/core_instruction_seed.jsonl")
GENERAL_EVAL = Path("data/raw/indoone_corpus.txt")
REGISTRY = Path("models/registry.json")


def ensure_extra_data() -> None:
    if SAFETY_EXTRA.is_file() and ACTION_EXTRA.is_file():
        return
    safety, actions = build()
    SAFETY_EXTRA.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in safety),
        encoding="utf-8",
    )
    ACTION_EXTRA.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in actions),
        encoding="utf-8",
    )
    print(f"generated_expansion safety={len(safety)} actions={len(actions)}", flush=True)



def _configure_email_tuning(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Freeze the base model and train only a small Email-adaptation slice."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable: list[tuple[str, nn.Parameter]] = []
    start_index = max(0, len(model.blocks) - TRAINABLE_LAST_BLOCKS)

    for index in range(start_index, len(model.blocks)):
        for name, parameter in model.blocks[index].named_parameters():
            parameter.requires_grad = True
            trainable.append((f"blocks.{index}.{name}", parameter))

    for name, parameter in model.ln_f.named_parameters():
        parameter.requires_grad = True
        trainable.append((f"ln_f.{name}", parameter))

    if not trainable:
        raise RuntimeError("No Email-tuning parameters were enabled.")

    return trainable


def _anchor_penalty(
    model: nn.Module,
    baseline_parameters: dict[str, torch.Tensor],
    device: str,
) -> torch.Tensor:
    """Keep the trainable slice close to the original model weights."""
    penalty = torch.zeros((), device=device)
    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        baseline = baseline_parameters[name]
        penalty = penalty + torch.mean((parameter - baseline) ** 2)
        count += 1

    return penalty / max(1, count)


def _clamp_weight_drift(
    model: nn.Module,
    baseline_parameters: dict[str, torch.Tensor],
) -> None:
    """Hard-limit each trainable parameter's absolute drift from the baseline."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            baseline = baseline_parameters[name]
            parameter.copy_(
                baseline
                + torch.clamp(
                    parameter - baseline,
                    min=-MAX_WEIGHT_DELTA,
                    max=MAX_WEIGHT_DELTA,
                )
            )

def benchmark_gpu(model_dir: Path, pool, sampling_weights, device: str) -> float:
    model, tokenizer = _load_model(model_dir)
    model = model.to(device)
    trainable = _configure_email_tuning(model)
    prepared = _prepare_instruction_examples(pool, tokenizer, model.block_size)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator().manual_seed(4242)

    if device == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    model.train()

    for _ in range(BENCHMARK_STEPS):
        x, y = _instruction_batchify(
            pool,
            tokenizer,
            model.block_size,
            BATCH_SIZE,
            device,
            generator,
            sampling_weights=sampling_weights,
            prepared_examples=prepared,
        )
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable],
            0.5,
        )
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.monotonic() - started
    speed = BENCHMARK_STEPS / elapsed if elapsed > 0 else 0.0
    planned = min(
        MAX_PLANNED_STEPS,
        max(1, int(speed * TRAIN_BUDGET_SECONDS * 0.90)),
    )

    print(
        json.dumps(
            {
                "benchmark_steps": BENCHMARK_STEPS,
                "elapsed_seconds": round(elapsed, 2),
                "steps_per_second": round(speed, 4),
                "training_budget_seconds": TRAIN_BUDGET_SECONDS,
                "planned_steps": planned,
                "trainable_last_blocks": TRAINABLE_LAST_BLOCKS,
                "learning_rate": LEARNING_RATE,
            },
            indent=2,
        ),
        flush=True,
    )
    return speed


def train_gpu(
    model,
    tokenizer,
    pool,
    sampling_weights,
    validation,
    steps,
    device,
):
    model = model.to(device)
    trainable = _configure_email_tuning(model)
    prepared = _prepare_instruction_examples(
        pool,
        tokenizer,
        model.block_size,
    )

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    generator = torch.Generator().manual_seed(4242)

    baseline_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    baseline_parameters = {
        name: value.to(device)
        for name, value in baseline_parameters.items()
    }

    best_loss = float("inf")
    best_state = None
    no_improvement = 0
    started = time.monotonic()
    completed_steps = 0

    model.train()

    for step in range(1, steps + 1):

        if time.monotonic() - started >= TRAIN_BUDGET_SECONDS:
            print(
                "training budget reached; stopping at last completed step",
                flush=True,
            )
            break

        x, y = _instruction_batchify(
            pool,
            tokenizer,
            model.block_size,
            BATCH_SIZE,
            device,
            generator,
            sampling_weights=sampling_weights,
            prepared_examples=prepared,
        )

        _, task_loss = model(x, y)

        if task_loss is None:
            raise RuntimeError("training loss is None")

        regularization = _anchor_penalty(
            model,
            baseline_parameters,
            device,
        )

        loss = task_loss + ANCHOR_LAMBDA * regularization

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable],
            0.5,
        )

        optimizer.step()

        _clamp_weight_drift(
            model,
            baseline_parameters,
        )

        completed_steps = step

        if (
            step == 1
            or step % EVAL_INTERVAL == 0
            or step == steps
        ):
            validation_loss = evaluate_instruction_loss(
                model,
                validation,
                tokenizer,
                model.block_size,
                BATCH_SIZE,
                device,
            )

            print(
                f"incremental step {step}/{steps} | "
                f"task_loss={float(task_loss):.4f} | "
                f"total_loss={float(loss):.4f} | "
                f"email_validation_loss={validation_loss}",
                flush=True,
            )

            if (
                validation_loss is not None
                and float(validation_loss) < best_loss
            ):
                best_loss = float(validation_loss)
                no_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                no_improvement += 1

            if no_improvement >= EARLY_STOP_PATIENCE:
                print(
                    "email validation has stopped improving; "
                    "early-stopping the incremental run",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError(
            "no validated incremental checkpoint was produced"
        )

    model.load_state_dict(best_state)

    return {
        "best_email_validation_loss": best_loss,
        "elapsed_seconds": time.monotonic() - started,
        "completed_steps": completed_steps,
        "trainable_last_blocks": TRAINABLE_LAST_BLOCKS,
        "learning_rate": LEARNING_RATE,
        "anchor_lambda": ANCHOR_LAMBDA,
        "max_weight_delta": MAX_WEIGHT_DELTA,
        "early_stop_patience": EARLY_STOP_PATIENCE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Indoone Email capability incremental GPU training.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("CUDA GPU is required for this Colab runner. Select Runtime -> Change runtime type -> GPU.")
    torch.set_float32_matmul_precision("high")
    print(f"training_device: {device}", flush=True)
    print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    ensure_extra_data()

    required = [
        args.model_dir / "indoone-small.pt", args.model_dir / "tokenizer.json",
        SAFETY_TRAIN, ACTION_TRAIN, SAFETY_EXTRA, ACTION_EXTRA,
        VALIDATION, TEST, ANCHOR, GENERAL_EVAL,
    ]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"required training file is missing: {path}")

    safety = load_examples(SAFETY_TRAIN) + load_examples(SAFETY_EXTRA)
    actions = load_examples(ACTION_TRAIN) + load_examples(ACTION_EXTRA)
    validation = load_examples(VALIDATION)
    test_cases = load_examples(TEST)
    anchors = load_examples(ANCHOR)

    pool, sampling_weights = _merge_pool(safety, actions, anchors)
    print(json.dumps({
        "safety_examples": len(safety),
        "action_examples": len(actions),
        "anchor_examples": len(anchors),
        "training_pool": len(pool),
        "sampling_policy": {
            "email_safety": SAFETY_WEIGHT,
            "email_actions": ACTION_WEIGHT,
            "core_anchors": ANCHOR_WEIGHT,
        },
    }, indent=2), flush=True)

    speed = benchmark_gpu(args.model_dir, pool, sampling_weights, device)
    planned_steps = min(MAX_PLANNED_STEPS, max(1, int(speed * TRAIN_BUDGET_SECONDS * 0.95)))

    base_checkpoint = args.model_dir / "indoone-small.pt"
    base_tokenizer = args.model_dir / "tokenizer.json"
    base_eval = evaluate_checkpoint(base_checkpoint, base_tokenizer, GENERAL_EVAL, batch_size=BATCH_SIZE)
    _write_initial_registry(args.registry, base_eval)

    model, tokenizer = _load_model(args.model_dir)
    training = train_gpu(model, tokenizer, pool, sampling_weights, validation, planned_steps, device)

    args.candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_checkpoint = args.candidate_dir / "indoone-small.pt"
    candidate_tokenizer = args.candidate_dir / "tokenizer.json"
    torch.save(
        {"config": {**model.config(), "model_version": "indoone-gpt-v2"}, "model_state": model.cpu().state_dict()},
        candidate_checkpoint,
    )
    shutil.copy2(base_tokenizer, candidate_tokenizer)

    candidate_eval = evaluate_checkpoint(candidate_checkpoint, candidate_tokenizer, GENERAL_EVAL, batch_size=BATCH_SIZE)
    base_email = _classify_test(base_checkpoint, base_tokenizer, test_cases)
    candidate_email = _classify_test(candidate_checkpoint, candidate_tokenizer, test_cases)
    gate = _behavior_gate(candidate_checkpoint, candidate_tokenizer)

    if candidate_email["accuracy"] < MIN_EMAIL_ACCURACY:
        raise SystemExit(f"candidate rejected: email accuracy {candidate_email['accuracy']:.3f} is below {MIN_EMAIL_ACCURACY:.2f}")
    if candidate_email["accuracy"] < base_email["accuracy"] + MIN_EMAIL_ACCURACY_GAIN:
        raise SystemExit(f"candidate rejected: email accuracy did not improve by at least {MIN_EMAIL_ACCURACY_GAIN:.2f}")
    if not bool(gate.get("overall_pass")):
        raise SystemExit("candidate rejected: core behavioral regression gate failed")
    max_loss = base_eval["loss"] * (1.0 + MAX_GENERAL_REGRESSION)
    max_perplexity = base_eval["perplexity"] * (1.0 + MAX_GENERAL_REGRESSION)
    if candidate_eval["loss"] > max_loss or candidate_eval["perplexity"] > max_perplexity:
        raise SystemExit(
            "candidate rejected: general loss/perplexity regression exceeded "
            f"{MAX_GENERAL_REGRESSION:.1%} tolerance"
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
        max_metric_regression=MAX_GENERAL_REGRESSION,
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

    print(json.dumps({
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
        "production_model_slot": str(args.model_dir),
        "remote_release_upload": "required: upload the promoted indoone-small.pt to the Indoone-Model indoone-model-v1 release using the same asset name",
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
