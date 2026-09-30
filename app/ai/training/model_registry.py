from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelRecord:
    version: str
    model_dir: str
    checkpoint: str
    tokenizer: str
    loss: float
    perplexity: float
    status: str
    behavioral_gate_passed: bool = False
    parent_version: str | None = None
    benchmark_version: str = "v1"


def _load_registry(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"active_version": None, "models": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model registry must contain a JSON object")
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("model registry models must be a list")
    return payload


def _write_registry(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_records(path: Path) -> list[ModelRecord]:
    payload = _load_registry(path)
    records: list[ModelRecord] = []
    for raw in payload["models"]:
        if not isinstance(raw, dict):
            raise ValueError("model registry contains an invalid model record")
        records.append(ModelRecord(**raw))
    return records


def active_record(path: Path) -> ModelRecord | None:
    payload = _load_registry(path)
    active = payload.get("active_version")
    if active is None:
        return None
    for record in load_records(path):
        if record.version == active:
            return record
    raise ValueError("active model version is missing from registry")


def should_promote(candidate: ModelRecord, current: ModelRecord | None) -> bool:
    if candidate.status != "candidate":
        raise ValueError("only candidate models can be promoted")
    if not candidate.version.strip():
        raise ValueError("candidate version cannot be empty")
    if candidate.benchmark_version != "v1":
        raise ValueError("unsupported benchmark version")
    if not math.isfinite(candidate.loss) or not math.isfinite(candidate.perplexity):
        raise ValueError("evaluation metrics must be finite")
    if candidate.loss < 0 or candidate.perplexity < 0:
        raise ValueError("evaluation metrics cannot be negative")
    if not candidate.behavioral_gate_passed:
        return False
    if current is None:
        return True
    if candidate.version == current.version:
        raise ValueError("candidate version must differ from the active version")
    if candidate.benchmark_version != current.benchmark_version:
        raise ValueError("benchmark versions must match")
    return candidate.loss < current.loss and candidate.perplexity < current.perplexity


def promote_candidate(path: Path, candidate: ModelRecord) -> ModelRecord:
    current = active_record(path)
    if not should_promote(candidate, current):
        raise ValueError("candidate must pass the behavioral gate and improve both loss and perplexity")

    payload = _load_registry(path)
    records = [record for record in load_records(path) if record.version != candidate.version]
    if current is not None:
        records = [
            ModelRecord(
                **{**asdict(record), "status": "retired" if record.version == current.version else record.status}
            )
            for record in records
        ]

    promoted = ModelRecord(
        **{
            **asdict(candidate),
            "status": "active",
            "parent_version": current.version if current is not None else candidate.parent_version,
        }
    )
    records.append(promoted)
    payload["active_version"] = promoted.version
    payload["models"] = [asdict(record) for record in records]
    _write_registry(path, payload)
    return promoted
