from __future__ import annotations

from utils.helper import IMAGE_MODELS


UPSTREAM_MODEL_PREFIX = "upstream/"


def local_image_fallback_model(requested_model: str, local_model_ids: set[str]) -> str | None:
    """Pick a locally supported image model after an external image request fails."""
    model = requested_model.removeprefix(UPSTREAM_MODEL_PREFIX)
    available = local_model_ids & IMAGE_MODELS
    if model in available:
        return model
    if "gpt-image-2" in available:
        return "gpt-image-2"
    return sorted(available)[0] if available else None
