from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelRecord:
    """A deployable model revision within a stable public model version."""

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
    revision: str | None = None
    artifact_sha256: str | None = None


def _load_registry(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"active_version": None, "active_revision": None, "models": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model registry must contain a JSON object")
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("model registry models must be a list")
    payload.setdefault("active_revision", None)
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
    active_version = payload.get("active_version")
    active_revision = payload.get("active_revision")
    if active_version is None:
        return None

    records = load_records(path)
    if active_revision:
        for record in records:
            if record.version == active_version and record.revision == active_revision:
                return record

    matches = [record for record in records if record.version == active_version]
    active_matches = [record for record in matches if record.status == "active"]
    if len(active_matches) == 1:
        return active_matches[0]
    if len(matches) == 1:
        return matches[0]
    raise ValueError("active model revision is missing or ambiguous in registry")


def should_promote(candidate: ModelRecord, current: ModelRecord | None) -> bool:
    if candidate.status != "candidate":
        raise ValueError("only candidate models can be promoted")
    if not candidate.version.strip():
        raise ValueError("candidate version cannot be empty")
    if candidate.benchmark_version != "v1":
        raise ValueError("benchmark versions must use v1")
    if not math.isfinite(candidate.loss) or not math.isfinite(candidate.perplexity):
        raise ValueError("evaluation metrics must be finite")
    if candidate.loss < 0 or candidate.perplexity < 0:
        raise ValueError("evaluation metrics cannot be negative")
    if not candidate.behavioral_gate_passed:
        return False
    if current is None:
        return True

    if candidate.benchmark_version != current.benchmark_version:
        raise ValueError("benchmark versions must match")
    if candidate.version == current.version:
        if not candidate.revision or not candidate.revision.strip():
            raise ValueError("candidate revision cannot be empty for a same-version update")
        if candidate.revision == current.revision:
            raise ValueError("candidate revision must differ from the active revision")

    return candidate.loss < current.loss and candidate.perplexity < current.perplexity


def promote_candidate(path: Path, candidate: ModelRecord) -> ModelRecord:
    current = active_record(path)
    if not should_promote(candidate, current):
        raise ValueError(
            "candidate must pass the behavioral gate and improve both loss and perplexity"
        )

    payload = _load_registry(path)
    records: list[ModelRecord] = []
    for record in load_records(path):
        if record.version == candidate.version and record.revision == candidate.revision:
            continue
        if (
            current is not None
            and record.version == current.version
            and record.revision == current.revision
        ):
            records.append(ModelRecord(**{**asdict(record), "status": "retired"}))
        else:
            records.append(record)

    promoted = ModelRecord(
        **{
            **asdict(candidate),
            "status": "active",
            "parent_version": current.version if current is not None else candidate.parent_version,
        }
    )
    records.append(promoted)
    payload["active_version"] = promoted.version
    payload["active_revision"] = promoted.revision
    payload["models"] = [asdict(record) for record in records]
    _write_registry(path, payload)
    return promoted
