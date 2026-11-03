from __future__ import annotations

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.types import ASGIApp, Receive, Scope, Send

from api.support import require_identity
from services.config import config


MODEL_ENDPOINTS = {
    "/v1/models": "GET",
    "/v1/images/generations": "POST",
    "/v1/images/edits": "POST",
    "/v1/chat/completions": "POST",
    "/v1/responses": "POST",
    "/v1/messages": "POST",
    "/v1/search": "POST",
}


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

    async def chunks():
        async for chunk in response.aiter_bytes():
            yield chunk

    async def close():
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
        background=BackgroundTask(close),
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
