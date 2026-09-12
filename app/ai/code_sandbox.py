from __future__ import annotations

from dataclasses import asdict, dataclass

from app.ai.coding import MAX_CODE_LENGTH, SUPPORTED_LANGUAGES

MAX_EXECUTION_SECONDS = 5
MAX_OUTPUT_BYTES = 16_384
MAX_MEMORY_MB = 128


@dataclass(frozen=True)
class SandboxRequest:
    language: str
    code: str
    timeout_seconds: int = MAX_EXECUTION_SECONDS
    max_output_bytes: int = MAX_OUTPUT_BYTES
    max_memory_mb: int = MAX_MEMORY_MB


def build_sandbox_request(language: str, code: str) -> dict[str, object]:
    language = language.strip().lower()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported language: {language}")
    if len(code) > MAX_CODE_LENGTH:
        raise ValueError(f"code exceeds {MAX_CODE_LENGTH} characters")
    if not code.strip():
        raise ValueError("code must not be empty")
    request = SandboxRequest(language=language, code=code)
    return {**asdict(request), "execution": "disabled", "reason": "No runtime is enabled in this contract."}


def sandbox_execution_tool(payload: str) -> str:
    import json

    data = json.loads(payload)
    language = str(data.get("language", ""))
    code = data.get("code", "")
    if not isinstance(code, str):
        raise ValueError("code must be a string")
    return json.dumps(build_sandbox_request(language, code), ensure_ascii=False, sort_keys=True)
