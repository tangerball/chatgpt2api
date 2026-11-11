from __future__ import annotations

import json
import unittest
from unittest import mock

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from services import external_upstream as upstream


REJECTION = {"error": {
    "code": "invalid_value", "param": "complexity",
    "type": "image_generation_user_error",
    "message": "Complexity 'medium' is not supported for this image model.",
}}
SUCCESS = {"data": [{"url": "https://images.example/poster.png"}]}
INPUT = {"model": "gpt-image-1", "prompt": "海报", "n": 1,
         "size": "1024x1024", "output_format": "png"}


class ImageCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.catalog = ["gpt-image-1", "gpt-image-2.5-sunburst", "gpt-image-2.5-flare"]
        self.catalog_status = 200
        self.first_status = 400
        self.first_body = REJECTION
        self.retry_status = 200
        self.retry_body = SUCCESS
        self.local_calls = 0
        self.transient_error_once = False
        upstream._model_catalog_cache.update({"base_url": "", "expires_at": 0.0, "models": {}})

        def handle(request):
            body = request.read()
            self.requests.append((request, body))
            if self.transient_error_once and request.url.path in upstream.IMAGE_ENDPOINTS:
                self.transient_error_once = False
                raise httpx.ConnectError("connection reset", request=request)
            if request.url.path == "/v1/models":
                return httpx.Response(self.catalog_status, json={"data": [{"id": x} for x in self.catalog]})
            image_calls = [r for r, _ in self.requests if r.url.path in upstream.IMAGE_ENDPOINTS]
            status, payload = (self.first_status, self.first_body) if len(image_calls) == 1 else (self.retry_status, self.retry_body)
            return httpx.Response(status, json=payload)

        transport = httpx.MockTransport(handle)
        real_async, real_sync = httpx.AsyncClient, httpx.Client
        patches = [
            mock.patch.object(upstream.httpx, "AsyncClient", side_effect=lambda **kw: real_async(transport=transport, **kw)),
            mock.patch.object(upstream.httpx, "Client", side_effect=lambda **kw: real_sync(transport=transport, **kw)),
            mock.patch.object(upstream.config, "get_external_upstream_settings", return_value={
                "enabled": True, "base_url": "https://gateway.example", "api_key": "fixture-key",
            }),
            mock.patch("api.support.require_identity"),
            mock.patch.object(upstream, "list_local_models", return_value={"data": [{"id": "gpt-image-2"}]}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        app = FastAPI()
        app.add_middleware(upstream.ExternalUpstreamMiddleware)

        @app.post("/v1/images/generations")
        @app.post("/v1/images/edits")
        async def local():
            self.local_calls += 1
            return JSONResponse(status_code=429, content={"error": "no available image quota"})

        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def generate(self, **overrides):
        return self.client.post("/v1/images/generations", json={**INPUT, **overrides})

    def image_requests(self):
        return [(r, b) for r, b in self.requests if r.url.path in upstream.IMAGE_ENDPOINTS]

    def test_original_request_recovers_without_local_quota(self):
        response = self.generate()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), SUCCESS)
        self.assertEqual(self.local_calls, 0)
        first, retry = self.image_requests()
        self.assertEqual(json.loads(first[1]), INPUT)
        self.assertEqual(json.loads(retry[1]), {**INPUT, "model": "gpt-image-2.5-sunburst"})
        self.assertEqual(response.headers["x-image-requested-model"], "gpt-image-1")
        self.assertEqual(response.headers["x-image-upstream-model"], "gpt-image-2.5-sunburst")
        self.assertEqual(int(retry[0].headers["content-length"]), len(retry[1]))
        self.assertNotIn("transfer-encoding", retry[0].headers)

    def test_json_edit_preserves_images_mask_and_quality(self):
        payload = {**INPUT, "images": ["data:image/png;base64,AAAA"], "mask": "mask", "quality": "low"}
        response = self.client.post("/v1/images/edits", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(self.image_requests()[1][1]), {**payload, "model": "gpt-image-2.5-sunburst"})

    def test_multipart_edit_preserves_binary_files_and_boundary(self):
        response = self.client.post("/v1/images/edits?trace=1", data={
            "model": "gpt-image-1", "prompt": "edit", "quality": "low",
        }, files=[
            ("image[]", ("first.png", b"first\x00\xff\r\nimage", "image/png")),
            ("image[]", ("second.png", b"second-image", "image/png")),
            ("mask", ("mask.png", b"mask-bytes", "image/png")),
        ])
        self.assertEqual(response.status_code, 200)
        (first, body), (retry, retry_body) = self.image_requests()
        self.assertEqual(retry_body, body.replace(b'name="model"\r\n\r\ngpt-image-1\r\n',
            b'name="model"\r\n\r\ngpt-image-2.5-sunburst\r\n'))
        self.assertEqual(first.headers["content-type"], retry.headers["content-type"])
        self.assertEqual(retry.url.query, b"trace=1")
        self.assertEqual(int(retry.headers["content-length"]), len(retry_body))
        self.assertNotIn("transfer-encoding", retry.headers)

    def test_success_is_never_retried(self):
        self.first_status, self.first_body = 200, SUCCESS
        response = self.generate(quality="medium", complexity="low")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(json.loads(self.requests[0][1])["complexity"], "low")

    def test_unrelated_error_does_not_change_model(self):
        self.first_body = {"error": {"code": "invalid_value", "param": "size", "message": "invalid size"}}
        self.assertEqual(self.generate().status_code, 429)
        self.assertEqual(len(self.requests), 1)

    def test_502_is_not_retried(self):
        self.first_status = 502
        self.assertEqual(self.generate().status_code, 429)
        self.assertEqual(len(self.requests), 1)

    def test_unknown_model_is_not_substituted(self):
        self.assertEqual(self.generate(model="some-other-image-model").status_code, 429)
        self.assertEqual(len(self.requests), 1)

    def test_new_image_model_is_not_retried(self):
        self.assertEqual(self.generate(model="gpt-image-2.5-flare").status_code, 429)
        self.assertEqual(len(self.requests), 1)

    def test_unlisted_model_is_never_used(self):
        self.catalog = ["gpt-image-1"]
        self.assertEqual(self.generate().status_code, 429)
        self.assertEqual(len(self.image_requests()), 1)

    def test_flare_used_only_when_listed_and_sunburst_absent(self):
        self.catalog = ["gpt-image-2.5-flare"]
        self.assertEqual(self.generate().status_code, 200)
        self.assertEqual(json.loads(self.image_requests()[1][1])["model"], "gpt-image-2.5-flare")

    def test_catalog_failure_preserves_local_fallback(self):
        self.catalog_status = 503
        self.assertEqual(self.generate().status_code, 429)
        self.assertEqual(len(self.image_requests()), 1)
        self.assertEqual(self.local_calls, 1)

    def test_retry_is_bounded_then_local_fallback(self):
        self.retry_status, self.retry_body = 400, REJECTION
        self.assertEqual(self.generate().status_code, 429)
        self.assertEqual(len(self.image_requests()), 2)
        self.assertEqual(self.local_calls, 1)

    def test_invalid_success_response_still_falls_back(self):
        self.first_status, self.first_body = 200, {"data": []}
        self.assertEqual(self.generate().status_code, 429)
        self.assertEqual(len(self.image_requests()), 1)

    def test_image_task_generation_uses_same_compatibility_retry(self):
        result = upstream.handle_image_task(INPUT, "generate")
        self.assertEqual(result["data"], SUCCESS["data"])
        first, retry = self.image_requests()
        before, after = json.loads(first[1]), json.loads(retry[1])
        self.assertEqual(after, {**before, "model": "gpt-image-2.5-sunburst"})

    def test_image_task_edit_replays_all_reference_images(self):
        result = upstream.handle_image_task({**INPUT, "images": [
            (b"first-image", "first.png", "image/png"), (b"second-image", "second.png", "image/png"),
        ]}, "edit")
        self.assertEqual(result["data"], SUCCESS["data"])
        self.assertEqual(len(self.image_requests()), 2)
        body = self.image_requests()[1][1]
        for value in (b"first-image", b"second-image", b"gpt-image-2.5-sunburst"):
            self.assertIn(value, body)

    def test_image_task_retry_is_bounded(self):
        self.retry_status, self.retry_body = 400, REJECTION
        with self.assertRaisesRegex(RuntimeError, "Complexity"):
            upstream.handle_image_task(INPUT, "generate")
        self.assertEqual(len(self.image_requests()), 2)

    def test_middleware_retries_transient_upstream_connection_for_image_edit(self):
        self.transient_error_once = True
        self.first_status, self.first_body = 200, SUCCESS
        response = self.client.post(
            "/v1/images/edits",
            data={"model": "gpt-image-2", "prompt": "edit"},
            files={"image": ("source.png", b"image-bytes", "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.image_requests()), 2)
        self.assertEqual(self.local_calls, 0)


if __name__ == "__main__":
    unittest.main()
