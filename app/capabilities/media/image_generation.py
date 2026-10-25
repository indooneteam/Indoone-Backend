from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

import httpx


MAX_PROMPT_LENGTH = 4_000
MAX_IMAGE_BYTES = 8_000_000


@dataclass(frozen=True)
class ImageGenerationResult:
    """Normalized image-generation result independent of model runtime."""

    provider: str
    image_base64: str | None = None
    mime_type: str = "image/png"
    model: str = ""
    metadata: dict[str, Any] | None = None


async def generate_image(prompt: str, width: int, height: int) -> ImageGenerationResult:
    """Call a configured local/self-hosted image model endpoint.

    The endpoint is expected to return JSON containing either ``image_base64``
    or ``images`` (a list of base64 strings), plus optional model metadata.
    No hosted AI provider is assumed.
    """
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError("prompt must contain 1-4000 characters")
    if not 256 <= width <= 2048 or not 256 <= height <= 2048:
        raise ValueError("width and height must be between 256 and 2048")

    endpoint = os.getenv("INDOONE_IMAGE_GENERATOR_URL", "").strip()
    if not endpoint:
        raise RuntimeError("image generation model endpoint is not configured")

    payload = {"prompt": prompt, "width": width, "height": height}
    timeout = float(os.getenv("INDOONE_IMAGE_GENERATOR_TIMEOUT", "120"))
    headers = {"Accept": "application/json"}
    token = os.getenv("INDOONE_IMAGE_GENERATOR_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        response = await client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        if len(response.content) > MAX_IMAGE_BYTES:
            raise RuntimeError("image generator response is too large")
        data: Any = response.json()

    if not isinstance(data, dict):
        raise RuntimeError("image generator response must be a JSON object")

    image_base64 = data.get("image_base64")
    if not isinstance(image_base64, str) or not image_base64:
        images = data.get("images")
        if isinstance(images, list) and images and isinstance(images[0], str):
            image_base64 = images[0]

    if image_base64:
        try:
            decoded = base64.b64decode(image_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("image generator returned invalid base64 image data") from exc
        if not decoded or len(decoded) > MAX_IMAGE_BYTES:
            raise RuntimeError("generated image is empty or too large")

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else None
    return ImageGenerationResult(
        provider=str(data.get("provider", "local")),
        image_base64=image_base64,
        mime_type=str(data.get("mime_type", "image/png")),
        model=str(data.get("model", "")),
        metadata=metadata,
    )
