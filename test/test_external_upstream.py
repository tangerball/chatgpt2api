from __future__ import annotations

import base64
import copy
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "chatgpt2api")

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from PIL import Image

from api import ai, system
from services import external_upstream
from services.config import config
from services.image_storage_service import image_storage_service


LOCAL_KEY = os.environ["CHATGPT2API_AUTH_KEY"]
LOCAL_AUTH = {"Authorization": f"Bearer {LOCAL_KEY}"}
REMOTE_KEY = "test-remote-key"


class ExternalUpstreamTests(unittest.TestCase):
    def setUp(self):
        original_data = config.data
        config.data = copy.deepcopy(original_data)
        self.addCleanup(setattr, config, "data", original_data)
        external_upstream._model_catalog_cache.update({"base_url": "", "expires_at": 0.0, "models": {}})
        local_models_patcher = mock.patch.object(external_upstream, "list_local_models", return_value={
            "object": "list", "data": [{"id": "local-model"}],
        })
        local_models_patcher.start()
        self.addCleanup(local_models_patcher.stop)
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        file_patcher = mock.patch("services.config.EXTERNAL_UPSTREAM_FILE", Path(temporary.name) / "external_upstream.json")
        file_patcher.start()
        self.addCleanup(file_patcher.stop)
        data_patcher = mock.patch("services.config.DATA_DIR", Path(temporary.name))
        data_patcher.start()
        self.addCleanup(data_patcher.stop)
        index_patcher = mock.patch.object(image_storage_service, "index_file", Path(temporary.name) / "image_index.json")
        index_patcher.start()
        self.addCleanup(index_patcher.stop)
        config.data["image_storage"] = {"mode": "local"}

        self.requests: list[tuple[httpx.Request, bytes]] = []
        self.upstream_status = 200
        self.upstream_headers = {"content-type": "application/json"}
        self.upstream_content = b'{"object":"list","data":[{"id":"remote-model"}]}'
        self.upstream_image_content: bytes | None = None
        self.upstream_error = False

        async def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append((request, await request.aread()))
            if request.url.path.startswith("/images/"):
                return httpx.Response(
                    200 if self.upstream_image_content is not None else 404,
                    content=self.upstream_image_content or b"",
                )
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

    @staticmethod
    def image_bytes(format: str = "PNG") -> bytes:
        output = io.BytesIO()
        Image.new("RGB", (2, 2), "red").save(output, format=format)
        return output.getvalue()

    def image_items(self) -> list[dict[str, object]]:
        response = self.client.get("/api/images", headers=LOCAL_AUTH)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["items"]

    def configure(self):
        response = self.client.post(
            "/api/external-upstream",
            headers=LOCAL_AUTH,
            json={"enabled": True, "base_url": "http://upstream.example:6679", "api_key": REMOTE_KEY},
        )
        self.assertEqual(response.status_code, 200, response.text)

    @staticmethod
    def expire_external_catalog():
        external_upstream._model_catalog_cache["expires_at"] = 0.0

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
        self.assertEqual({item["id"] for item in response.json()["data"]}, {"local-model", "remote-model"})
        forwarded, _ = self.requests[0]
        self.assertEqual(str(forwarded.url), "http://upstream.example:6679/v1/models")
        self.assertEqual(forwarded.headers["authorization"], f"Bearer {REMOTE_KEY}")
        self.assertNotIn("cookie", forwarded.headers)
        self.assertNotIn(LOCAL_KEY, str(forwarded.headers))

    def test_daixu_root_and_v1_base_urls_route_to_openai_v1_once(self):
        for base_url in ("https://daixu.qzz.io/", "https://daixu.qzz.io/v1"):
            with self.subTest(base_url=base_url):
                response = self.client.post(
                    "/api/external-upstream",
                    headers=LOCAL_AUTH,
                    json={"enabled": True, "base_url": base_url, "api_key": REMOTE_KEY},
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["upstream"]["base_url"], "https://daixu.qzz.io")

                self.requests.clear()
                self.expire_external_catalog()
                response = self.client.get("/v1/models", headers=LOCAL_AUTH)
                self.assertEqual(response.status_code, 200, response.text)
                forwarded, _ = self.requests[0]
                self.assertEqual(str(forwarded.url), "https://daixu.qzz.io/v1/models")

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

    def test_image_generation_prefers_external_upstream_for_local_model_collision(self):
        self.configure()
        self.upstream_content = b'{"data":[{"url":"https://upstream.example/image.png"}]}'

        response = self.client.post(
            "/v1/images/generations",
            headers=LOCAL_AUTH,
            json={"model": "local-model", "prompt": "red square", "n": 1},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.requests), 1)
        forwarded, body = self.requests[0]
        self.assertEqual(str(forwarded.url), "http://upstream.example:6679/v1/images/generations")
        self.assertEqual(json.loads(body)["model"], "local-model")

    def test_image_generation_falls_back_to_local_when_external_upstream_fails(self):
        self.configure()
        self.upstream_status = 503
        self.upstream_content = b'{"error":{"message":"upstream unavailable"}}'
        local_result = {"data": [{"url": "https://local.example/image.png"}]}
        with mock.patch.object(ai.openai_v1_image_generations, "handle", return_value=local_result) as local_handler:
            response = self.client.post(
                "/v1/images/generations",
                headers=LOCAL_AUTH,
                json={"model": "local-model", "prompt": "red square", "n": 1},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), local_result)
        self.assertEqual(len(self.requests), 1)
        local_handler.assert_called_once()
        self.assertEqual(local_handler.call_args.args[0]["model"], "local-model")

    def test_base64_image_is_saved_without_changing_upstream_response(self):
        self.configure()
        image = self.image_bytes()
        self.upstream_content = json.dumps({"data": [{
            "b64_json": base64.b64encode(image).decode(),
            "url": "http://upstream.example:6679/images/remote.png",
        }]}).encode()

        response = self.client.post(
            "/v1/images/generations",
            headers=LOCAL_AUTH,
            json={"model": "remote-model", "prompt": "red square"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, self.upstream_content)
        self.assertEqual(len(self.requests), 1)
        items = self.image_items()
        self.assertEqual(len(items), 1)
        self.assertEqual(image_storage_service.get_bytes(items[0]["path"]), image)

    def test_same_origin_image_url_is_saved_as_png(self):
        self.configure()
        self.upstream_image_content = self.image_bytes("JPEG")
        self.upstream_content = json.dumps({"data": [{
            "url": "http://upstream.example:6679/images/2026/11/03/remote.jpg",
        }]}).encode()

        response = self.client.post(
            "/v1/images/edits", headers=LOCAL_AUTH,
            data={"model": "remote-model", "prompt": "red square"},
            files={"image": ("source.png", self.image_bytes(), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, self.upstream_content)
        self.assertEqual([request.method for request, _ in self.requests], ["POST", "GET"])
        items = self.image_items()
        self.assertEqual(len(items), 1)
        saved = image_storage_service.get_bytes(items[0]["path"])
        with Image.open(io.BytesIO(saved)) as image:
            self.assertEqual(image.format, "PNG")

    def test_untrusted_image_url_and_upstream_errors_are_not_saved(self):
        self.configure()
        self.upstream_content = json.dumps({"data": [{
            "url": "http://other.example/images/remote.png",
        }]}).encode()
        response = self.client.post(
            "/v1/images/generations", headers=LOCAL_AUTH,
            json={"model": "remote-model", "prompt": "red square"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.image_items(), [])

        self.upstream_status = 429
        self.upstream_content = json.dumps({"data": [{
            "b64_json": base64.b64encode(self.image_bytes()).decode(),
        }]}).encode()
        response = self.client.post(
            "/v1/images/generations", headers=LOCAL_AUTH,
            json={"model": "remote-model", "prompt": "red square"},
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.image_items(), [])

    def test_storage_failure_does_not_change_upstream_response(self):
        self.configure()
        self.upstream_content = json.dumps({"data": [{
            "b64_json": base64.b64encode(self.image_bytes()).decode(),
        }]}).encode()
        with mock.patch.object(image_storage_service, "save", side_effect=OSError("full")):
            with self.assertLogs(external_upstream.logger, level="ERROR"):
                response = self.client.post(
                    "/v1/images/generations", headers=LOCAL_AUTH,
                    json={"model": "remote-model", "prompt": "red square"},
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, self.upstream_content)
        self.assertEqual(self.image_items(), [])

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
        response = self.client.post("/v1/chat/completions", headers=LOCAL_AUTH, json={
            "model": "remote-model", "messages": [{"role": "user", "content": "hello"}],
        })
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "10")
        self.assertEqual(response.content, self.upstream_content)
        self.expire_external_catalog()
        response = self.client.get("/v1/models", headers=LOCAL_AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertIn("local-model", {item["id"] for item in response.json()["data"]})

    def test_unavailable_upstream_and_cors_preflight(self):
        self.configure()
        self.upstream_error = True
        response = self.client.get("/v1/models", headers=LOCAL_AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()["data"]], ["local-model"])
        response = self.client.post("/v1/chat/completions", headers=LOCAL_AUTH, json={
            "model": "remote-model", "messages": [{"role": "user", "content": "hello"}],
        })
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

    def test_local_imported_model_stays_routed_locally_while_upstream_is_enabled(self):
        self.configure()
        with mock.patch.object(ai.openai_v1_chat_complete, "handle", return_value={"model": "local-model", "choices": []}):
            response = self.client.post("/v1/chat/completions", headers=LOCAL_AUTH, json={
                "model": "local-model", "messages": [{"role": "user", "content": "hello"}],
            })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model"], "local-model")
        self.assertEqual(self.requests, [])

    def test_upstream_model_alias_routes_remotely_and_strips_alias(self):
        self.configure()
        self.upstream_content = b'{"id":"ok","model":"local-model"}'
        response = self.client.post("/v1/chat/completions", headers=LOCAL_AUTH, json={
            "model": "upstream/local-model", "messages": [{"role": "user", "content": "hello"}],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.requests), 1)
        forwarded, body = self.requests[0]
        self.assertEqual(json.loads(body)["model"], "local-model")
        self.assertEqual(str(forwarded.url), "http://upstream.example:6679/v1/chat/completions")

    def test_colliding_model_id_selects_local_or_upstream_quota_by_prefix(self):
        self.configure()
        self.upstream_content = b'{"object":"list","data":[{"id":"local-model"}]}'
        catalog = self.client.get("/v1/models", headers=LOCAL_AUTH)
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual({item["id"] for item in catalog.json()["data"]}, {
            "local-model", "upstream/local-model",
        })
        self.requests.clear()

        with mock.patch.object(ai.openai_v1_chat_complete, "handle", return_value={"model": "local-model"}):
            local_response = self.client.post("/v1/chat/completions", headers=LOCAL_AUTH, json={
                "model": "local-model", "messages": [{"role": "user", "content": "local"}],
            })
        self.assertEqual(local_response.status_code, 200, local_response.text)
        self.assertEqual(self.requests, [])

        upstream_response = self.client.post("/v1/chat/completions", headers=LOCAL_AUTH, json={
            "model": "upstream/local-model", "messages": [{"role": "user", "content": "remote"}],
        })
        self.assertEqual(upstream_response.status_code, 200, upstream_response.text)
        self.assertEqual(len(self.requests), 1)
        _, body = self.requests[0]
        self.assertEqual(json.loads(body)["model"], "local-model")

    def test_multipart_upstream_alias_is_stripped_without_changing_file_parts(self):
        self.configure()
        response = self.client.post(
            "/v1/images/edits",
            headers=LOCAL_AUTH,
            data={"prompt": "edit", "model": "upstream/remote-model"},
            files={"image": ("source.png", b"image-bytes", "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        forwarded, body = self.requests[0]
        self.assertIn("multipart/form-data", forwarded.headers["content-type"])
        self.assertIn(b'name="model"\r\n\r\nremote-model', body)
        self.assertNotIn(b"upstream/remote-model", body)
        self.assertIn(b"image-bytes", body)
        self.assertIn(b"source.png", body)

    def test_image_task_external_model_uses_provider_and_saves_local_image(self):
        self.configure()
        image = self.image_bytes()
        sync_requests: list[tuple[httpx.Request, bytes]] = []

        def handle_sync(request: httpx.Request) -> httpx.Response:
            sync_requests.append((request, request.read()))
            return httpx.Response(200, json={"data": [{
                "b64_json": base64.b64encode(image).decode(),
                "revised_prompt": "a red square",
            }]})

        real_client = httpx.Client
        transport = httpx.MockTransport(handle_sync)
        with mock.patch.object(
            external_upstream.httpx,
            "Client",
            side_effect=lambda **kwargs: real_client(transport=transport, **kwargs),
        ):
            result = external_upstream.handle_image_task({
                "model": "gpt-image-2.5-sunburst",
                "prompt": "red square",
                "n": "2",
                "size": "1024x1024",
                "base_url": "https://canvas.example",
            }, "generate")

        self.assertEqual(len(sync_requests), 1)
        request, body = sync_requests[0]
        self.assertEqual(str(request.url), "http://upstream.example:6679/v1/images/generations")
        self.assertEqual(request.headers["authorization"], f"Bearer {REMOTE_KEY}")
        request_payload = json.loads(body)
        self.assertEqual(request_payload["model"], "gpt-image-2.5-sunburst")
        self.assertIsInstance(request_payload["n"], int)
        self.assertEqual(request_payload["n"], 2)
        self.assertEqual(request_payload["response_format"], "b64_json")
        self.assertIsNotNone(result)
        self.assertNotIn("b64_json", result["data"][0])
        self.assertEqual(result["data"][0]["url"].split("/images/")[0], "https://canvas.example")
        items = self.image_items()
        self.assertEqual(len(items), 1)
        self.assertEqual(image_storage_service.get_bytes(items[0]["path"]), image)

    def test_image_task_uses_upstream_first_even_when_model_is_available_locally(self):
        self.configure()
        real_client = httpx.Client
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [{"url": "https://upstream.example/image.png"}]}))
        with mock.patch.object(
            external_upstream.httpx,
            "Client",
            side_effect=lambda **kwargs: real_client(transport=transport, **kwargs),
        ):
            result = external_upstream.handle_image_task({"model": "local-model", "prompt": "local"}, "generate")
        self.assertIsNotNone(result)
        self.assertEqual(result["data"][0]["url"], "https://upstream.example/image.png")

    def test_image_task_edit_forwards_multipart_to_external_provider(self):
        self.configure()
        sync_requests: list[tuple[httpx.Request, bytes]] = []

        def handle_sync(request: httpx.Request) -> httpx.Response:
            sync_requests.append((request, request.read()))
            return httpx.Response(200, json={"data": [{"url": "https://upstream.example/image.png"}]})

        real_client = httpx.Client
        transport = httpx.MockTransport(handle_sync)
        with mock.patch.object(
            external_upstream.httpx,
            "Client",
            side_effect=lambda **kwargs: real_client(transport=transport, **kwargs),
        ):
            result = external_upstream.handle_image_task({
                "model": "upstream/remote-model",
                "prompt": "edit",
                "images": [(b"input-image", "source.png", "image/png")],
            }, "edit")

        request, body = sync_requests[0]
        self.assertEqual(str(request.url), "http://upstream.example:6679/v1/images/edits")
        self.assertIn(b'name="model"\r\n\r\nremote-model', body)
        self.assertIn(b"source.png", body)
        self.assertIn(b"input-image", body)
        self.assertEqual(result["data"][0]["url"], "https://upstream.example/image.png")

    def test_upstream_catalog_failure_still_lists_local_imported_models(self):
        self.configure()
        self.upstream_error = True
        response = self.client.get("/v1/models", headers=LOCAL_AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()["data"]], ["local-model"])
        with mock.patch.object(ai.openai_v1_chat_complete, "handle", return_value={"model": "local-model", "choices": []}):
            response = self.client.post("/v1/chat/completions", headers=LOCAL_AUTH, json={
                "model": "local-model", "messages": [{"role": "user", "content": "still local"}],
            })
        self.assertEqual(response.status_code, 200, response.text)

    def test_local_catalog_failure_still_lists_external_models(self):
        self.configure()
        self.upstream_content = b'{"object":"list","data":[{"id":"remote-model"}]}'
        with mock.patch.object(external_upstream, "list_local_models", side_effect=RuntimeError("local catalog unavailable")):
            response = self.client.get("/v1/models", headers=LOCAL_AUTH)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([item["id"] for item in response.json()["data"]], ["remote-model"])

    def test_all_model_catalogs_unavailable_returns_gateway_error(self):
        self.configure()
        self.upstream_error = True
        with mock.patch.object(external_upstream, "list_local_models", side_effect=RuntimeError("local catalog unavailable")):
            response = self.client.get("/v1/models", headers=LOCAL_AUTH)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["message"], "Model catalogs unavailable")

    def test_model_catalog_keeps_both_sources_when_model_ids_collide(self):
        self.configure()
        self.upstream_content = b'{"object":"list","data":[{"id":"local-model"},{"id":"remote-model"}]}'
        response = self.client.get("/v1/models", headers=LOCAL_AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual({item["id"] for item in response.json()["data"]}, {
            "local-model", "upstream/local-model", "remote-model",
        })


if __name__ == "__main__":
    unittest.main()
