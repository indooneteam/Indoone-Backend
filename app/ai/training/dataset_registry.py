from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetRecord:
    version: str
    path: str
    sha256: str
    source_type: str
    license: str
    approved: bool


def fingerprint_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"dataset file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_dataset(
    registry_path: Path,
    version: str,
    dataset_path: Path,
    source_type: str,
    license: str,
    approved: bool,
) -> DatasetRecord:
    version = version.strip()
    source_type = source_type.strip()
    license = license.strip()
    if not version:
        raise ValueError("dataset version cannot be empty")
    if not source_type:
        raise ValueError("dataset source_type cannot be empty")
    if not license:
        raise ValueError("dataset license cannot be empty")
    if not approved:
        raise ValueError("dataset must be explicitly approved before registration")

    record = DatasetRecord(
        version=version,
        path=str(dataset_path),
        sha256=fingerprint_file(dataset_path),
        source_type=source_type,
        license=license,
        approved=True,
    )

    payload = {"datasets": []}
    if registry_path.exists():
        loaded = json.loads(registry_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not isinstance(loaded.get("datasets", []), list):
            raise ValueError("dataset registry must contain a datasets list")
        payload = loaded

    datasets = [item for item in payload["datasets"] if isinstance(item, dict) and item.get("version") != version]
    datasets.append(asdict(record))
    payload["datasets"] = datasets
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry_path.with_suffix(registry_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(registry_path)
    return record


def load_datasets(registry_path: Path) -> list[DatasetRecord]:
    if not registry_path.exists():
        return []
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets", []), list):
        raise ValueError("dataset registry must contain a datasets list")
    return [DatasetRecord(**item) for item in payload["datasets"]]
