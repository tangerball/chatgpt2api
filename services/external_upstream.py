from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import re
import time
from threading import RLock
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.types import ASGIApp, Receive, Scope, Send
from PIL import Image

from services.config import config
from services.image_model_routing import local_image_fallback_model
from services.image_storage_service import image_storage_service
from services.protocol.openai_v1_models import list_models as list_local_models


logger = logging.getLogger(__name__)
IMAGE_ENDPOINTS = {"/v1/images/generations", "/v1/images/edits"}
MAX_MIRROR_RESPONSE_BYTES = 128 * 1024 * 1024
MAX_MIRROR_IMAGE_BYTES = 64 * 1024 * 1024
UPSTREAM_MODEL_PREFIX = "upstream/"
MODEL_CATALOG_TTL_SECONDS = 60
_model_catalog_lock = RLock()
_model_catalog_cache: dict[str, object] = {"base_url": "", "expires_at": 0.0, "models": {}}


def _replace_model_in_body(body: bytes, content_type: str, model: str) -> bytes | None:
    if "application/json" in content_type.lower():
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        payload["model"] = model
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()

    if "multipart/form-data" not in content_type.lower():
        return None
    boundary_match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;\s]+))", content_type, re.IGNORECASE)
    if not boundary_match:
        return None
    boundary = (boundary_match.group(1) or boundary_match.group(2)).encode()
    delimiter = b"--" + boundary
    parts = body.split(delimiter)
    for index, part in enumerate(parts):
        header_end = part.find(b"\r\n\r\n")
        if header_end < 0 or not re.search(rb'content-disposition:[^\r\n]*\bname="model"', part[:header_end], re.IGNORECASE):
            continue
        value_start = header_end + 4
        value_end = part.find(b"\r\n", value_start)
        if value_end < 0:
            return None
        parts[index] = part[:value_start] + model.encode() + part[value_end:]
        return delimiter.join(parts)
    return None


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
    from api.support import require_identity

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


async def _fetch_upstream_models(upstream: dict[str, object]) -> dict[str, dict[str, object]] | None:
    """读取上游模型目录；上游不可用时返回 None，让本地账号池目录继续工作。"""
    base_url = str(upstream["base_url"])
    now = time.monotonic()
    with _model_catalog_lock:
        cached = _model_catalog_cache["models"]
        if _model_catalog_cache["base_url"] == base_url and now < float(_model_catalog_cache["expires_at"]):
            return dict(cached) if isinstance(cached, dict) else None
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=20, write=10, pool=10),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = await client.get(
                base_url + "/v1/models",
                headers={"Authorization": f"Bearer {upstream['api_key']}", "Accept-Encoding": "identity"},
            )
        payload = response.json()
        if response.is_error or not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("upstream model catalog response is invalid")
        models = {
            str(item["id"]).strip(): dict(item)
            for item in payload["data"]
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
    except (httpx.RequestError, ValueError, json.JSONDecodeError) as exc:
        logger.warning({"event": "external_model_catalog_failed", "error_type": type(exc).__name__})
        with _model_catalog_lock:
            cached = _model_catalog_cache["models"]
            return dict(cached) if _model_catalog_cache["base_url"] == base_url and isinstance(cached, dict) else None
    with _model_catalog_lock:
        _model_catalog_cache.update({"base_url": base_url, "expires_at": time.monotonic() + MODEL_CATALOG_TTL_SECONDS, "models": models})
    return models


async def _merged_model_response(upstream: dict[str, object]) -> JSONResponse:
    """合并本地号池与外部上游模型；重名模型使用前缀显式选择外部额度。"""
    local_failed = False
    try:
        local_payload = await asyncio.to_thread(list_local_models)
    except Exception as exc:  # noqa: BLE001 - the regular local endpoint owns its normal error response
        logger.warning({"event": "local_model_catalog_failed", "error_type": type(exc).__name__})
        local_failed = True
        local_payload = {}
    local_items = local_payload.get("data", []) if isinstance(local_payload, dict) else []
    if not isinstance(local_items, list):
        local_items = []
    merged = {str(item.get("id")): dict(item) for item in local_items if isinstance(item, dict) and item.get("id")}
    external_items = await _fetch_upstream_models(upstream)
    if local_failed and external_items is None:
        return JSONResponse(status_code=502, content={"error": {"message": "Model catalogs unavailable"}})
    external_items = external_items or {}
    for model_id, item in external_items.items():
        external_id = f"{UPSTREAM_MODEL_PREFIX}{model_id}" if model_id in merged else model_id
        merged.setdefault(external_id, {**item, "id": external_id, "root": external_id})
    return JSONResponse(content={"object": "list", "data": [merged[key] for key in sorted(merged)]})


def _model_from_request(path: str, body: bytes, content_type: str) -> tuple[str, bytes, bool]:
    """提取 JSON 或 multipart 中的模型名，并仅在选择上游别名时剥离路由前缀。"""
    model = ""
    explicit_upstream = False
    if "application/json" in content_type.lower():
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict):
            model = str(payload.get("model") or "")
            if model.startswith(UPSTREAM_MODEL_PREFIX):
                explicit_upstream = True
                model = model[len(UPSTREAM_MODEL_PREFIX):]
                payload["model"] = model
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    elif "multipart/form-data" in content_type.lower():
        boundary_match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;\s]+))", content_type, re.IGNORECASE)
        if boundary_match:
            boundary = (boundary_match.group(1) or boundary_match.group(2)).encode()
            delimiter = b"--" + boundary
            parts = body.split(delimiter)
            for index, part in enumerate(parts):
                header_end = part.find(b"\r\n\r\n")
                if header_end < 0 or not re.search(rb'content-disposition:[^\r\n]*\bname="model"', part[:header_end], re.IGNORECASE):
                    continue
                value_start = header_end + 4
                value_end = part.find(b"\r\n", value_start)
                if value_end < 0:
                    break
                try:
                    model = part[value_start:value_end].decode().strip()
                except UnicodeDecodeError:
                    break
                if model.startswith(UPSTREAM_MODEL_PREFIX):
                    explicit_upstream = True
                    model = model[len(UPSTREAM_MODEL_PREFIX):]
                    parts[index] = part[:value_start] + model.encode() + part[value_end:]
                    body = delimiter.join(parts)
                break
    if not model and path == "/v1/images/generations":
        model = "gpt-image-2"
    return model, body, explicit_upstream


async def _local_model_ids() -> set[str]:
    """读取本地账号池实际支持的模型名；目录故障不阻断对已配置上游的显式请求。"""
    try:
        payload = await asyncio.to_thread(list_local_models)
    except Exception as exc:  # noqa: BLE001 - preserve direct upstream access when the local catalog is down
        logger.warning({"event": "local_model_catalog_failed", "error_type": type(exc).__name__})
        return set()
    items = payload.get("data", []) if isinstance(payload, dict) else []
    return {
        str(item.get("id")).strip()
        for item in items
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    } if isinstance(items, list) else set()


def _replay_body(body: bytes, receive: Receive) -> Receive:
    """重放已检查的请求体，避免模型路由检查吞掉本地或远端处理所需的数据。"""
    sent = False

    async def replay() -> dict[str, object]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


def handle_image_task(payload: dict[str, object], mode: str) -> dict[str, object] | None:
    """Route the web image-task workflow to the external provider when its model is selected."""
    upstream = config.get_external_upstream_settings()
    if not upstream["enabled"]:
        return None

    requested_model = str(payload.get("model") or "gpt-image-2")
    model = requested_model.removeprefix(UPSTREAM_MODEL_PREFIX)

    base_url = str(upstream["base_url"])
    api_key = str(upstream["api_key"])
    endpoint = "/v1/images/edits" if mode == "edit" else "/v1/images/generations"
    fields: dict[str, object] = {
        "prompt": str(payload.get("prompt") or ""),
        "model": model,
        "n": int(payload.get("n") or 1),
        "quality": str(payload.get("quality") or "auto"),
        "response_format": "b64_json",
    }
    if payload.get("size"):
        fields["size"] = str(payload["size"])

    headers = {"Authorization": f"Bearer {api_key}", "Accept-Encoding": "identity"}
    timeout = httpx.Timeout(connect=15, read=600, write=600, pool=15)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client:
            if mode == "edit":
                from services.protocol.openai_v1_image_edit import _composite_mask

                images = payload.get("images") if isinstance(payload.get("images"), list) else []
                masks = payload.get("mask") if isinstance(payload.get("mask"), list) else []
                images = _composite_mask(images, masks)
                files = [
                    ("image", (str(filename), image_data, str(mime_type)))
                    for image_data, filename, mime_type in images
                ]
                if not files:
                    raise RuntimeError("image is required")
                response = client.post(
                    base_url + endpoint,
                    headers=headers,
                    data=fields,
                    files=files,
                )
            else:
                response = client.post(
                    base_url + endpoint,
                    headers={**headers, "Content-Type": "application/json"},
                    json=fields,
                )
    except httpx.RequestError as exc:
        logger.warning({"event": "external_image_task_failed", "error_type": type(exc).__name__})
        raise RuntimeError("External model upstream unavailable") from exc

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(f"External model upstream returned HTTP {response.status_code} with invalid JSON") from exc
    if response.is_error:
        error = result.get("error") if isinstance(result, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        raise RuntimeError(str(message or f"External model upstream returned HTTP {response.status_code}"))
    if not isinstance(result, dict) or not isinstance(result.get("data"), list):
        raise RuntimeError("External model upstream returned an invalid image response")

    # Persist external outputs in the local image library and keep task history compact.
    for item in result["data"]:
        if not isinstance(item, dict):
            continue
        image_data = _decode_image(item.get("b64_json"))
        if image_data is None:
            continue
        try:
            stored = image_storage_service.save(image_data, str(payload.get("base_url") or "") or None)
            item["url"] = stored.url
            item.pop("b64_json", None)
        except Exception:
            logger.exception("Could not save an external upstream image task result")

    return {"data": result["data"], "usage": result.get("usage")}


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
        if path == "/v1/models":
            from api.support import require_identity

            request = Request(scope, receive)
            try:
                require_identity(request.headers.get("authorization"))
            except HTTPException as exc:
                response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
            else:
                response = await _merged_model_response(upstream)
            await response(scope, receive, send)
            return

        request = Request(scope, receive)
        body = await request.body() if method not in {"GET", "HEAD"} else b""
        model, upstream_body, explicit_upstream = _model_from_request(
            path,
            body,
            request.headers.get("content-type", ""),
        )
        image_request = path in IMAGE_ENDPOINTS
        if path == "/v1/search" or explicit_upstream or image_request:
            use_external = True
        else:
            local_models = await _local_model_ids()
            use_external = bool(model) and model not in local_models

        if not use_external:
            await self.app(scope, _replay_body(body, receive), send)
            return

        forwarded_request = Request(scope, _replay_body(upstream_body, receive))
        response = await forward(forwarded_request, upstream)
        if image_request and response.status_code >= 400:
            local_model_ids = await _local_model_ids()
            fallback_model = local_image_fallback_model(model, local_model_ids)
            if fallback_model is None:
                await response(scope, receive, send)
                return
            local_body = _replace_model_in_body(
                upstream_body,
                request.headers.get("content-type", ""),
                fallback_model,
            )
            if local_body is None:
                await response(scope, receive, send)
                return
            body_iterator = getattr(response, "body_iterator", None)
            close_iterator = getattr(body_iterator, "aclose", None)
            if close_iterator is not None:
                await close_iterator()
            logger.warning({
                "event": "external_image_request_fallback",
                "path": path,
                "upstream_status": response.status_code,
                "model": model,
                "fallback_model": fallback_model,
            })
            await self.app(scope, _replay_body(local_body, receive), send)
            return
        await response(scope, receive, send)
