from __future__ import annotations

import copy
import os
import unittest
from unittest import mock

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "chatgpt2api")

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from api import ai, system
from services import external_upstream
from services.config import config


LOCAL_KEY = os.environ["CHATGPT2API_AUTH_KEY"]
LOCAL_AUTH = {"Authorization": f"Bearer {LOCAL_KEY}"}
REMOTE_KEY = "test-remote-key"


class ExternalUpstreamTests(unittest.TestCase):
    def setUp(self):
        original_data = config.data
        config.data = copy.deepcopy(original_data)
        self.addCleanup(setattr, config, "data", original_data)
        save_patcher = mock.patch.object(config, "_save")
        save_patcher.start()
        self.addCleanup(save_patcher.stop)

        self.requests: list[tuple[httpx.Request, bytes]] = []
        self.upstream_status = 200
        self.upstream_headers = {"content-type": "application/json"}
        self.upstream_content = b'{"object":"list","data":[{"id":"remote-model"}]}'
        self.upstream_error = False

        async def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append((request, await request.aread()))
            if self.upstream_error:
                raise httpx.ConnectError("unavailable", request=request)
            return httpx.Response(
                self.upstream_status,
                headers=self.upstream_headers,
                content=self.upstream_content,
            )

        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(handle)
        client_patcher = mock.patch.object(
            external_upstream.httpx,
            "AsyncClient",
            side_effect=lambda **kwargs: real_client(transport=transport, **kwargs),
        )
        client_patcher.start()
        self.addCleanup(client_patcher.stop)

        app = FastAPI()
        app.add_middleware(external_upstream.ExternalUpstreamMiddleware)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.include_router(ai.create_router())
        app.include_router(system.create_router("test"))
        self.client = TestClient(app)

    def configure(self):
        response = self.client.post(
            "/api/external-upstream",
            headers=LOCAL_AUTH,
            json={"enabled": True, "base_url": "http://upstream.example:6679", "api_key": REMOTE_KEY},
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_settings_require_admin_and_never_return_key(self):
        self.assertEqual(self.client.get("/api/external-upstream").status_code, 401)
        self.configure()
        response = self.client.get("/api/external-upstream", headers=LOCAL_AUTH)
        self.assertEqual(response.json()["upstream"], {
            "enabled": True,
            "base_url": "http://upstream.example:6679",
            "has_api_key": True,
        })
        self.assertNotIn(REMOTE_KEY, response.text)
        general = self.client.get("/api/settings", headers=LOCAL_AUTH)
        self.assertNotIn(REMOTE_KEY, general.text)
        self.assertNotIn("external_upstream", general.json()["config"])
        self.client.post("/api/external-upstream", headers=LOCAL_AUTH, json={
            "enabled": True, "base_url": "http://upstream.example:6679", "api_key": ""
        })
        self.assertEqual(config.get_external_upstream_settings()["api_key"], REMOTE_KEY)

    def test_invalid_settings_and_clearing_key(self):
        response = self.client.post("/api/external-upstream", headers=LOCAL_AUTH, json={
            "enabled": True, "base_url": "http://upstream.example:6679", "api_key": ""
        })
        self.assertEqual(response.status_code, 400)
        self.configure()
        response = self.client.post("/api/external-upstream", headers=LOCAL_AUTH, json={
            "enabled": False, "base_url": "http://upstream.example:6679", "clear_api_key": True
        })
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["upstream"]["has_api_key"])

    def test_local_auth_precedes_upstream_and_replaces_credentials(self):
        self.configure()
        self.assertEqual(self.client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code, 401)
        self.assertEqual(self.requests, [])
        response = self.client.get("/v1/models", headers={**LOCAL_AUTH, "Cookie": "session=secret"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "remote-model")
        forwarded, _ = self.requests[0]
        self.assertEqual(str(forwarded.url), "http://upstream.example:6679/v1/models")
        self.assertEqual(forwarded.headers["authorization"], f"Bearer {REMOTE_KEY}")
        self.assertNotIn("cookie", forwarded.headers)
        self.assertNotIn(LOCAL_KEY, str(forwarded.headers))

    def test_multipart_upload_is_forwarded_without_reencoding(self):
        self.configure()
        response = self.client.post(
            "/v1/images/edits?trace=1",
            headers=LOCAL_AUTH,
            data={"prompt": "edit", "model": "gpt-image-2"},
            files={"image": ("source.png", b"image-bytes", "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        forwarded, body = self.requests[0]
        self.assertEqual(forwarded.url.query, b"trace=1")
        self.assertIn("multipart/form-data", forwarded.headers["content-type"])
        self.assertIn(b"image-bytes", body)
        self.assertIn(b"source.png", body)

    def test_stream_and_upstream_error_status_are_preserved(self):
        self.configure()
        self.upstream_headers = {"content-type": "text/event-stream"}
        self.upstream_content = b'data: {"delta":"hello"}\n\ndata: [DONE]\n\n'
        response = self.client.post(
            "/v1/chat/completions", headers=LOCAL_AUTH,
            json={"model": "remote-model", "stream": True, "messages": []},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(response.content, self.upstream_content)

        self.upstream_status = 429
        self.upstream_headers = {"content-type": "application/json", "retry-after": "10"}
        self.upstream_content = b'{"error":{"message":"rate limited"}}'
        response = self.client.get("/v1/models", headers=LOCAL_AUTH)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "10")
        self.assertEqual(response.content, self.upstream_content)

    def test_unavailable_upstream_and_cors_preflight(self):
        self.configure()
        self.upstream_error = True
        response = self.client.get("/v1/models", headers=LOCAL_AUTH)
        self.assertEqual(response.status_code, 502)
        self.assertNotIn(REMOTE_KEY, response.text)
        preflight = self.client.options("/v1/models", headers={
            "Origin": "https://tangerball.cn:19529",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        })
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(preflight.headers["access-control-allow-origin"], "*")

    def test_disabled_upstream_uses_local_models(self):
        with mock.patch.object(ai.openai_v1_models, "list_models", return_value={
            "object": "list", "data": [{"id": "local-model"}]
        }):
            response = self.client.get("/v1/models", headers=LOCAL_AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "local-model")
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
