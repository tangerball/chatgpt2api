from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.types import ASGIApp, Receive, Scope, Send
from PIL import Image

from api.support import require_identity
from services.config import config
from services.image_storage_service import image_storage_service


logger = logging.getLogger(__name__)
IMAGE_ENDPOINTS = {"/v1/images/generations", "/v1/images/edits"}
MAX_MIRROR_RESPONSE_BYTES = 128 * 1024 * 1024
MAX_MIRROR_IMAGE_BYTES = 64 * 1024 * 1024


MODEL_ENDPOINTS = {
    "/v1/models": "GET",
    "/v1/images/generations": "POST",
    "/v1/images/edits": "POST",
    "/v1/chat/completions": "POST",
    "/v1/responses": "POST",
    "/v1/messages": "POST",
    "/v1/search": "POST",
}


def _decode_image(value: object) -> bytes | None:
    if not isinstance(value, str) or len(value) > (MAX_MIRROR_IMAGE_BYTES + 2) // 3 * 4 + 4:
        return None
    try:
        image = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    return image if len(image) <= MAX_MIRROR_IMAGE_BYTES else None


def _same_upstream_image_url(url: str, base_url: str) -> bool:
    try:
        parsed = urlsplit(url)
        base = urlsplit(base_url)
        same_origin = (
            parsed.scheme.lower(), parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        ) == (
            base.scheme.lower(), base.hostname, base.port or (443 if base.scheme == "https" else 80)
        )
    except ValueError:
        return False
    return (
        same_origin
        and not parsed.username
        and not parsed.password
        and not parsed.fragment
        and parsed.path.startswith("/images/")
        and all(part not in {".", ".."} and "%" not in part for part in parsed.path.split("/"))
    )


async def _download_image(url: str, base_url: str) -> bytes | None:
    if not _same_upstream_image_url(url, base_url):
        return None
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=15, read=60, write=15, pool=15),
        follow_redirects=False,
        trust_env=False,
    ) as client:
        async with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
            if response.status_code != 200:
                return None
            image = bytearray()
            async for chunk in response.aiter_bytes():
                if len(image) + len(chunk) > MAX_MIRROR_IMAGE_BYTES:
                    return None
                image.extend(chunk)
            return bytes(image)


def _save_image(image_data: bytes) -> None:
    with Image.open(io.BytesIO(image_data)) as image:
        image.load()
        if image.format == "PNG":
            png_data = image_data
        else:
            converted = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            output = io.BytesIO()
            converted.save(output, format="PNG")
            png_data = output.getvalue()
    image_storage_service.save(png_data)


async def _mirror_images(body: bytes, base_url: str) -> None:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return
    for item in payload["data"]:
        if not isinstance(item, dict):
            continue
        image = _decode_image(item.get("b64_json"))
        if image is None and isinstance(item.get("url"), str):
            try:
                image = await _download_image(item["url"], base_url)
            except httpx.RequestError:
                logger.warning("Could not download an external upstream image")
        if image:
            try:
                await asyncio.to_thread(_save_image, image)
            except Exception:
                logger.exception("Could not save an external upstream image")


async def forward(request: Request, upstream: dict[str, object]):
    try:
        require_identity(request.headers.get("authorization"))
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    base_url = str(upstream["base_url"])
    api_key = str(upstream["api_key"])
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept-Encoding": "identity",
    }
    for name in ("content-type", "accept", "anthropic-version"):
        if value := request.headers.get(name):
            headers[name] = value

    url = base_url + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=15, read=None, write=600, pool=15),
        follow_redirects=False,
        trust_env=False,
    )
    try:
        outbound = client.build_request(
            request.method,
            url,
            headers=headers,
            content=request.stream() if request.method != "GET" else None,
        )
        response = await client.send(outbound, stream=True)
    except httpx.RequestError:
        await client.aclose()
        return JSONResponse(status_code=502, content={"error": "External model upstream unavailable"})

    content_type = response.headers.get("content-type", "").lower()
    captured = bytearray() if (
        request.url.path in IMAGE_ENDPOINTS
        and 200 <= response.status_code < 300
        and (not content_type or "json" in content_type)
    ) else None
    complete = False

    async def chunks():
        nonlocal captured, complete
        try:
            async for chunk in response.aiter_bytes():
                if captured is not None:
                    if len(captured) + len(chunk) <= MAX_MIRROR_RESPONSE_BYTES:
                        captured.extend(chunk)
                    else:
                        captured = None
                        logger.warning("External upstream image response exceeds the mirror size limit")
                yield chunk
            complete = True
        finally:
            if not complete:
                await response.aclose()
                await client.aclose()

    async def finish():
        try:
            if complete and captured is not None:
                await _mirror_images(bytes(captured), base_url)
        except Exception:
            logger.exception("Could not mirror external upstream images")
        finally:
            await response.aclose()
            await client.aclose()

    response_headers = {}
    for name in ("content-type", "cache-control", "retry-after"):
        if value := response.headers.get(name):
            response_headers[name] = value
    return StreamingResponse(
        chunks(),
        status_code=response.status_code,
        headers=response_headers,
        background=BackgroundTask(finish),
    )


class ExternalUpstreamMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = scope["method"]
        path = scope["path"]
        upstream = config.get_external_upstream_settings()
        if not upstream["enabled"] or MODEL_ENDPOINTS.get(path) != method:
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive)
        response = await forward(request, upstream)
        await response(scope, receive, send)
