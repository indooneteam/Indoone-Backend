from __future__ import annotations

import json
from pathlib import Path

from app.ai.train import (
    CAPABILITY_INSTRUCTION_WEIGHT,
    CURATED_INSTRUCTION_WEIGHT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_INSTRUCTION_MIX_RATIO,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_TRAINING_STEPS,
    DEFAULT_WEIGHT_DECAY,
    GENERATED_INSTRUCTION_WEIGHT,
)


PATH = Path("training_manifest.json")


def main() -> int:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("training manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise SystemExit("unsupported training manifest schema_version")

    policy = payload.get("training_policy")
    contract = payload.get("feature_contract")
    defaults = payload.get("training_defaults")
    if not isinstance(policy, dict) or not isinstance(contract, dict) or not isinstance(defaults, dict):
        raise SystemExit("training manifest requires training_policy, feature_contract, and training_defaults")

    required_policy = {
        "trigger_on_backend_ai_change",
        "trigger_on_api_change",
        "trigger_on_dataset_change",
        "require_dataset_audit",
        "require_behavioral_gate",
        "require_candidate_evaluation",
        "auto_publish_candidate",
        "auto_promote_only_on_gate",
    }
    missing_policy = required_policy - policy.keys()
    if missing_policy:
        raise SystemExit(f"training manifest is missing policy keys: {sorted(missing_policy)}")
    if not all(isinstance(policy[key], bool) for key in required_policy):
        raise SystemExit("all training policy values must be booleans")
    if not policy["auto_promote_only_on_gate"]:
        raise SystemExit("automatic promotion must remain gate-protected")

    required_fields = contract.get("required_fields")
    categories = contract.get("categories")
    if not isinstance(required_fields, list) or not all(isinstance(item, str) for item in required_fields):
        raise SystemExit("feature_contract.required_fields must be a string list")
    if not isinstance(categories, list) or not categories or not all(isinstance(item, str) for item in categories):
        raise SystemExit("feature_contract.categories must be a non-empty string list")

    for key in ("steps", "batch_size", "checkpoint_interval", "seed"):
        if not isinstance(defaults.get(key), int) or defaults[key] <= 0:
            raise SystemExit(f"training default {key} must be a positive integer")
    if not isinstance(defaults.get("learning_rate"), (int, float)) or defaults["learning_rate"] <= 0:
        raise SystemExit("training default learning_rate must be positive")
    if not isinstance(defaults.get("weight_decay"), (int, float)) or defaults["weight_decay"] < 0:
        raise SystemExit("training default weight_decay must be non-negative")
    expected_defaults = {
        "steps": DEFAULT_TRAINING_STEPS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "checkpoint_interval": DEFAULT_CHECKPOINT_INTERVAL,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "seed": DEFAULT_SEED,
        "instruction_mix_ratio": DEFAULT_INSTRUCTION_MIX_RATIO,
    }
    for key, expected in expected_defaults.items():
        if defaults.get(key) != expected:
            raise SystemExit(f"training default {key} does not match executable recipe: expected {expected}")
    instruction_mix_ratio = defaults.get("instruction_mix_ratio")
    if not isinstance(instruction_mix_ratio, (int, float)) or not 0.0 < float(instruction_mix_ratio) <= 1.0:
        raise SystemExit("training default instruction_mix_ratio must be in (0, 1]")
    sampling = defaults.get("instruction_sampling")
    if not isinstance(sampling, dict) or set(sampling) != {"curated", "generated_multilingual", "capability"}:
        raise SystemExit("training default instruction_sampling must define curated, generated_multilingual, and capability")
    if any(not isinstance(value, (int, float)) or value <= 0 for value in sampling.values()):
        raise SystemExit("instruction sampling probabilities must be positive numbers")
    if abs(sum(float(value) for value in sampling.values()) - 1.0) > 1e-9:
        raise SystemExit("instruction sampling probabilities must sum to 1")
    expected_sampling = {
        "curated": CURATED_INSTRUCTION_WEIGHT,
        "generated_multilingual": GENERATED_INSTRUCTION_WEIGHT,
        "capability": CAPABILITY_INSTRUCTION_WEIGHT,
    }
    if sampling != expected_sampling:
        raise SystemExit("instruction sampling probabilities do not match executable recipe")

    print("training_manifest=valid")
    print("automatic_training=enabled")
    print("automatic_promotion=gate_protected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
