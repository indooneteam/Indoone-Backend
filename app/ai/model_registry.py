from __future__ import annotations

import json
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
    parent_version: str | None = None


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
    if candidate.loss < 0 or candidate.perplexity < 0:
        raise ValueError("evaluation metrics cannot be negative")
    if current is None:
        return True
    return candidate.loss < current.loss and candidate.perplexity < current.perplexity


def promote_candidate(path: Path, candidate: ModelRecord) -> ModelRecord:
    current = active_record(path)
    if not should_promote(candidate, current):
        raise ValueError("candidate does not improve both loss and perplexity")

    payload = _load_registry(path)
    records = [record for record in load_records(path) if record.version != candidate.version]
    if current is not None:
        records = [
            ModelRecord(
                **{**asdict(record), "status": "retired" if record.version == current.version else record.status}
            )
            for record in records
        ]
    records.append(ModelRecord(**{**asdict(candidate), "status": "active"}))
    payload["active_version"] = candidate.version
    payload["models"] = [asdict(record) for record in records]
    _write_registry(path, payload)
    return records[-1]
