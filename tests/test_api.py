import asyncio
import csv
import io
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image
from psd_tools import PSDImage

from scripts import api


PROJECT_DIR = Path(__file__).resolve().parent.parent


def image_bytes(color=(120, 40, 200, 180)):
    stream = io.BytesIO()
    Image.new("RGBA", (8, 6), color).save(stream, format="PNG")
    return stream.getvalue()


def fake_masks(image, processor, prompt, mask_threshold, min_mask_area, max_masks):
    columns = np.indices((image.height, image.width))[1]
    return [
        {"segmentation": columns < 4, "area": 24, "prompt": "hair"},
        {"segmentation": columns >= 4, "area": 24, "prompt": "shirt"},
    ]


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        workspace = tempfile.TemporaryDirectory(prefix=".api-check-", dir=PROJECT_DIR)
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name)
        self.log_path = self.root / "logs" / "api_request.txt"
        self.app = api.create_app(
            self.root / "output", self.root / "models", self.log_path
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://api.test"
        )
        self.addAsyncCleanup(self.client.aclose)
        self.processor = object()
        processor_patch = patch.object(
            api, "get_sam3_processor", return_value=self.processor
        )
        masks_patch = patch.object(api, "get_masks", side_effect=fake_masks)
        self.load_processor = processor_patch.start()
        self.get_masks = masks_patch.start()
        self.addCleanup(processor_patch.stop)
        self.addCleanup(masks_patch.stop)

    async def segment(self, content=None, **settings):
        return await self.client.post(
            "/segment",
            files={"image": ("source.png", content or image_bytes(), "image/png")},
            data={"text_prompt": "hair\nshirt", **settings},
        )

    async def test_segment_and_create_layer_return_downloadable_png_urls(self):
        response = await self.segment(
            confidence_threshold="0.6", mask_threshold="0.4",
            max_masks="12", min_mask_area="2",
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertRegex(result["segmentation_id"], r"^[0-9a-f]{32}$")
        self.assertEqual(len(result["images"]), 1)
        self.load_processor.assert_called_once_with(0.6, str(self.root / "models"))
        self.assertEqual(self.get_masks.call_args.args[1:],
                         (self.processor, "hair\nshirt", 0.4, 2, 12))
        preview = await self.client.get(result["images"][0])
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.headers["content-type"], "image/png")

        response = await self.client.post(
            "/create-layer", json={"segmentation_id": result["segmentation_id"]}
        )
        self.assertEqual(response.status_code, 200, response.text)
        layers = response.json()
        self.assertEqual(layers["segmentation_id"], result["segmentation_id"])
        self.assertEqual(len(layers["images"]), 2)
        for index, url in enumerate(layers["images"]):
            self.assertTrue(url.startswith("http://api.test/images/"))
            image_response = await self.client.get(url)
            self.assertEqual(image_response.status_code, 200)
            with Image.open(io.BytesIO(image_response.content)) as image:
                self.assertEqual(image.size, (8, 6))
                self.assertEqual(image.mode, "RGBA")
                expected = np.zeros((6, 8, 4), dtype=np.uint8)
                expected[:, index * 4:(index + 1) * 4] = (120, 40, 200, 180)
                np.testing.assert_array_equal(np.asarray(image), expected)
        self.assertEqual(self.get_masks.call_count, 1)

    async def test_every_request_is_appended_to_the_request_log(self):
        headers = {
            "referer": "https://example.test/source,page",
            "user-agent": "LayerDivider Test Agent, 1.0",
        }
        first = await self.client.get("/openapi.json?from=test", headers=headers)
        second = await self.client.get("/missing", headers=headers)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 404)

        with self.log_path.open(encoding="utf-8", newline="") as log_file:
            rows = list(csv.reader(log_file))

        self.assertEqual(len(rows), 2)
        for row, route in zip(rows, ("/openapi.json", "/missing")):
            self.assertEqual(len(row), 5)
            self.assertRegex(
                row[0], re.compile(r"^\[\d{2}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}\]$")
            )
            self.assertEqual(
                row[1:],
                ["GET", headers["referer"], route, headers["user-agent"]],
            )

    async def test_segmentations_are_isolated_and_survive_app_restart(self):
        first = (await self.segment(image_bytes((255, 0, 0, 255)))).json()
        second = (await self.segment(image_bytes((0, 255, 0, 255)))).json()
        self.assertNotEqual(first["segmentation_id"], second["segmentation_id"])
        restarted = api.create_app(
            self.root / "output", self.root / "models", self.log_path
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restarted), base_url="http://api.test"
        ) as client:
            for result, color in ((first, (255, 0, 0, 255)),
                                  (second, (0, 255, 0, 255))):
                response = await client.post(
                    "/create-layer", json={"segmentation_id": result["segmentation_id"]}
                )
                self.assertEqual(response.status_code, 200, response.text)
                png = await client.get(response.json()["images"][0])
                with Image.open(io.BytesIO(png.content)) as image:
                    self.assertEqual(image.getpixel((0, 0)), color)
        self.assertEqual(self.get_masks.call_count, 2)

    async def test_invalid_requests_do_not_start_model_inference(self):
        response = await self.segment(b"this is not an image")
        self.assertEqual(response.status_code, 400)
        for settings in ({"text_prompt": "   "}, {"confidence_threshold": "1.1"},
                         {"mask_threshold": "-1"}, {"max_masks": "0"},
                         {"min_mask_area": "-1"}):
            response = await self.segment(**settings)
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual((await self.client.post("/segment")).status_code, 422)
        self.load_processor.assert_not_called()

    async def test_unknown_or_invalid_segmentation_ids(self):
        response = await self.client.post(
            "/create-layer", json={"segmentation_id": "0" * 32}
        )
        self.assertEqual(response.status_code, 404)
        for payload in ({}, {"segmentation_id": "../../private"}):
            response = await self.client.post("/create-layer", json=payload)
            self.assertEqual(response.status_code, 422)

    async def test_create_layer_reuses_the_uuid_folder_and_stable_urls(self):
        job_id = (await self.segment()).json()["segmentation_id"]
        first = (await self.client.post("/create-layer", json={"segmentation_id": job_id})).json()
        second = (await self.client.post("/create-layer", json={"segmentation_id": job_id})).json()
        self.assertEqual(first["images"], second["images"])
        folder = self.root / "output" / "jobs" / "layers" / job_id
        self.assertEqual(sorted(path.name for path in folder.iterdir()), ["001_hair.png", "002_shirt.png"])
        self.assertEqual(len(list((self.root / "output" / "jobs" / "state").iterdir())), 1)

    async def test_model_failures_have_useful_http_errors(self):
        self.load_processor.side_effect = RuntimeError("CUDA is unavailable.")
        response = await self.segment()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "CUDA is unavailable.")
        self.load_processor.side_effect = None
        self.get_masks.side_effect = ValueError("SAM 3 found no usable masks.")
        response = await self.segment()
        self.assertEqual(response.status_code, 422)

    async def test_deleted_layers_stop_being_served_and_are_not_recreated(self):
        job_id = (await self.segment()).json()["segmentation_id"]
        first = (await self.client.post("/create-layer", json={"segmentation_id": job_id})).json()
        filename = Path(urlsplit(first["images"][0]).path).name
        response = await self.client.request(
            "DELETE",
            "/delete-layer",
            json={"segmentation_id": job_id, "filename": filename},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["images"], first["images"][1:])
        self.assertEqual((await self.client.get(first["images"][0])).status_code, 404)
        second = (await self.client.post("/create-layer", json={"segmentation_id": job_id})).json()
        self.assertEqual(second["images"], first["images"][1:])
        self.assertEqual((await self.client.get(second["images"][0])).status_code, 200)

        repeated = await self.client.post(
            "/delete-layer", json={"segmentation_id": job_id, "filename": filename}
        )
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assertEqual(repeated.json()["images"], second["images"])

    async def test_edit_layer_replaces_only_the_selected_png(self):
        job_id = (await self.segment()).json()["segmentation_id"]
        layers = (await self.client.post(
            "/create-layer", json={"segmentation_id": job_id}
        )).json()
        filename = Path(urlsplit(layers["images"][0]).path).name
        second_before = (await self.client.get(layers["images"][1])).content

        def edited_masks(image, processor, prompt, mask_threshold, min_mask_area, max_masks):
            rows = np.indices((image.height, image.width))[0]
            return [{
                "segmentation": rows < 3,
                "area": 24,
                "score": 0.9,
                "prompt": prompt,
            }]

        self.get_masks.side_effect = edited_masks
        response = await self.client.patch(
            "/edit-layer",
            json={
                "segmentation_id": job_id,
                "filename": filename,
                "confidence_threshold": 0.8,
                "mask_threshold": 0.7,
                "max_masks": 9,
                "min_mask_area": 3,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["images"], layers["images"])
        self.load_processor.assert_called_with(0.8, str(self.root / "models"))
        self.assertEqual(
            self.get_masks.call_args.args[1:],
            (self.processor, "hair", 0.7, 3, 9),
        )

        edited = await self.client.get(layers["images"][0])
        with Image.open(io.BytesIO(edited.content)) as image:
            expected = np.zeros((6, 8, 4), dtype=np.uint8)
            expected[:3] = (120, 40, 200, 180)
            np.testing.assert_array_equal(np.asarray(image), expected)
        self.assertEqual((await self.client.get(layers["images"][1])).content, second_before)

        _, saved = self.app.state.jobs.mask(job_id, 0)
        self.assertEqual(saved, {
            "confidence_threshold": 0.8,
            "mask_threshold": 0.7,
            "max_masks": 9,
            "min_mask_area": 3,
        })

    async def test_create_psd_returns_a_downloadable_layered_file(self):
        job_id = (await self.segment()).json()["segmentation_id"]
        await self.client.post("/create-layer", json={"segmentation_id": job_id})
        response = await self.client.post(
            "/create-psd", json={"segmentation_id": job_id}
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["segmentation_id"], job_id)
        self.assertTrue(result["psd"].startswith("http://api.test/psd/_"))

        download = await self.client.get(result["psd"])
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.content[:4], b"8BPS")
        psd = PSDImage.open(io.BytesIO(download.content))
        self.assertEqual(psd.size, (8, 6))
        self.assertEqual([layer.name for layer in psd], ["001_hair", "002_shirt"])

        second = await self.client.post(
            "/create-psd", json={"segmentation_id": job_id}
        )
        self.assertNotEqual(second.json()["psd"], result["psd"])

    async def test_source_images_and_masks_are_not_public(self):
        result = (await self.segment()).json()
        segmentation_id = result["segmentation_id"]
        for path in (
            f"/images/{segmentation_id}/source.png",
            f"/jobs/{segmentation_id}/source.png",
            f"/images/..%2fstate/{segmentation_id}/source.png",
            f"/images/..%2fstate/{segmentation_id}/mask_000.npz",
            f"/previews/..%2fstate/{segmentation_id}/job.json",
        ):
            self.assertEqual((await self.client.get(path)).status_code, 404)

    async def test_concurrent_requests_serialize_the_cached_processor(self):
        active = 0
        maximum = 0
        counter_lock = threading.Lock()

        def inference(*args):
            nonlocal active, maximum
            with counter_lock:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.02)
                return fake_masks(*args)
            finally:
                with counter_lock:
                    active -= 1

        self.get_masks.side_effect = inference
        responses = await asyncio.gather(*(self.segment() for _ in range(4)))
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(maximum, 1)
        self.assertEqual(len({item.json()["segmentation_id"] for item in responses}), 4)

    async def test_new_routes_validate_jobs_layers_and_filenames(self):
        unknown = "0" * 32
        for method, route, payload in (
            ("POST", "/create-psd", {"segmentation_id": unknown}),
            ("PATCH", "/edit-layer", {
                "segmentation_id": unknown, "filename": "001_hair.png"
            }),
            ("DELETE", "/delete-layer", {
                "segmentation_id": unknown, "filename": "001_hair.png"
            }),
        ):
            response = await self.client.request(method, route, json=payload)
            self.assertEqual(response.status_code, 404, response.text)

        job_id = (await self.segment()).json()["segmentation_id"]
        for route, payload in (
            ("/create-psd", {"segmentation_id": job_id}),
            ("/edit-layer", {
                "segmentation_id": job_id, "filename": "001_hair.png"
            }),
        ):
            response = await self.client.post(route, json=payload)
            self.assertEqual(response.status_code, 422, response.text)

        for route in ("/edit-layer", "/delete-layer"):
            response = await self.client.post(route, json={
                "segmentation_id": job_id, "filename": "../source.png"
            })
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.get_masks.call_count, 1)

    async def test_openapi_documents_all_routes(self):
        schema = (await self.client.get("/openapi.json")).json()
        self.assertIn("post", schema["paths"]["/segment"])
        self.assertIn("post", schema["paths"]["/create-layer"])
        self.assertIn("post", schema["paths"]["/create-psd"])
        self.assertIn("post", schema["paths"]["/edit-layer"])
        self.assertIn("patch", schema["paths"]["/edit-layer"])
        self.assertIn("delete", schema["paths"]["/delete-layer"])
        segment_body = schema["paths"]["/segment"]["post"]["requestBody"]["content"]
        self.assertIn("multipart/form-data", segment_body)
        response = schema["components"]["schemas"]["ImageResponse"]["properties"]
        self.assertEqual(response["images"]["type"], "array")
        psd = schema["components"]["schemas"]["PSDResponse"]["properties"]
        self.assertEqual(psd["psd"]["type"], "string")


if __name__ == "__main__":
    unittest.main()
