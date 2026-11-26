import base64
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import httpx
import numpy as np
from gradio.state_holder import SessionState
from PIL import Image
from psd_tools import PSDImage

from ldivider.ld_ui import LayerDividerUI


PROJECT_DIR = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class UiJobTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        workspace = tempfile.TemporaryDirectory(prefix=".ui-job-check-", dir=PROJECT_DIR)
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name)
        self.ui = LayerDividerUI(self.root / "output", self.root / "models")
        self.state = SessionState(self.ui.demo)
        self.source = Image.new("RGBA", (8, 6), (100, 50, 200, 180))
        self.columns = np.indices((6, 8))[1]
        self.masks = [
            {"segmentation": self.columns < 4, "area": 24, "prompt": "hair", "score": .9},
            {"segmentation": self.columns >= 4, "area": 24, "prompt": "shirt", "score": .8},
        ]

    async def call(self, name, inputs, state=None, event=None, ui=None):
        ui = ui or self.ui
        index = next(i for i, fn in enumerate(ui.demo.fns) if fn.fn and fn.fn.__name__ == name)
        result = await ui.demo.process_api(index, inputs, state=state or self.state, event_data=event)
        names = {component._id: name for name, component in ui.components.items()}
        return {names[key]: value for key, value in zip(ui.demo.dependencies[index]["outputs"], result["data"])}

    async def segment(self, state=None, color=None):
        stream = io.BytesIO()
        image = Image.new("RGBA", self.source.size, color) if color else self.source
        image.save(stream, format="PNG")
        image_data = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
        with patch("ldivider.ld_ui.get_sam3_processor", return_value=object()), patch("ldivider.ld_ui.get_masks", return_value=self.masks):
            return await self.call("segment_image", [image_data, "hair\nshirt", .3, .45, 50, 100], state)

    async def ready_job(self):
        job_id = (await self.segment())["job_id"]
        await self.call("create_layers", [job_id])
        return job_id

    async def select(self, job_id, index=0, ui=None, state=None):
        ui = ui or self.ui
        event = gr.SelectData(ui.components["gallery"], {"index": index, "value": "layer", "selected": True})
        return await self.call("select_layer", [job_id], state, event, ui)

    async def test_only_segment_allocates_a_new_job(self):
        job_id = (await self.segment())["job_id"]
        self.assertRegex(job_id, r"^[0-9a-f]{32}$")
        self.assertEqual(list((self.ui.jobs.layers_dir / job_id).iterdir()), [])
        first = await self.call("create_layers", [job_id])
        paths = self.ui.jobs.layer_paths(job_id)
        self.assertTrue(all(Path(path).parent == self.root / "output" / "jobs" / "layers" / job_id for path in paths))
        self.assertEqual(len(first["gallery"]), 2)
        self.assertEqual(first["job_id"], job_id)
        again = await self.call("create_layers", [job_id])
        self.assertEqual(again["job_id"], job_id)
        self.assertEqual(self.ui.jobs.layer_paths(job_id), paths)
        self.assertEqual(len(list(self.ui.jobs.state_dir.iterdir())), 1)
        self.assertEqual(len(list((self.ui.jobs.layers_dir / job_id).glob("*.png"))), 2)
        second = await self.segment(color=(0, 255, 0, 255))
        self.assertNotEqual(second["job_id"], job_id)
        self.assertEqual(second["gallery"], [])
        self.assertEqual(len(list(self.ui.jobs.state_dir.iterdir())), 2)
        self.assertTrue(all(Path(path).is_file() for path in paths))

    async def test_selection_opens_editor_with_saved_part_and_settings(self):
        job_id = await self.ready_job()
        selected = await self.select(job_id)
        self.assertEqual(self.state[self.ui.components["selected"]._id], 0)
        opened = await self.call("open_layer_from_preview", [job_id, json.dumps({"index": 0})])
        self.assertTrue(opened["editor"]["visible"])
        self.assertEqual(opened["edit_prompt"], "hair")
        self.assertEqual(opened["edit_confidence_threshold"], .3)
        self.assertEqual(opened["edit_mask_threshold"], .45)
        self.assertEqual(opened["edit_max_masks"], 50)
        self.assertEqual(opened["edit_min_mask_area"], 100)
        with self.assertRaisesRegex(gr.Error, "Select a layer"):
            await self.call("open_layer_from_preview", [job_id, json.dumps({"index": 999})], state=SessionState(self.ui.demo))

    async def test_each_pencil_opens_its_own_layer_without_a_gallery_selection(self):
        job_id = await self.ready_job()
        for index, name in ((1, "shirt"), (0, "hair")):
            opened = await self.call(
                "open_layer_from_preview", [job_id, json.dumps({"index": index, "request": index})]
            )
            self.assertEqual(opened["edit_prompt"], name)
            self.assertTrue(opened["editor"]["visible"])
            self.assertEqual(self.state[self.ui.components["selected"]._id], index)
        self.assertNotIn("edit_button", self.ui.components)

    async def test_preview_is_unsaved_and_replace_updates_only_selected_file(self):
        job_id = await self.ready_job()
        await self.select(job_id)
        paths = self.ui.jobs.layer_paths(job_id)
        before = [digest(path) for path in paths]
        source_hash = digest(self.ui.jobs.state_dir / job_id / "source.png")
        other_mask_hash = digest(self.ui.jobs.state_dir / job_id / "mask_001.npz")
        candidates = [
            {"segmentation": self.columns >= 4, "area": 24, "prompt": "hair", "score": .99},
            {"segmentation": self.columns < 2, "area": 12, "prompt": "hair", "score": .6},
        ]
        with patch("ldivider.ld_ui.get_sam3_processor", return_value=object()), patch("ldivider.ld_ui.get_masks", return_value=candidates) as inference:
            await self.call("preview_layer", [job_id, None, .7, .6, 35, 10, None])
        self.assertEqual(inference.call_args.args[2], "hair")
        np.testing.assert_array_equal(np.asarray(inference.call_args.args[0]), np.asarray(self.source))
        self.assertEqual([digest(path) for path in paths], before)
        with patch("ldivider.ld_ui.get_masks", side_effect=AssertionError("The preview should be reused")):
            edited = await self.call("apply_layer_edit", [job_id, None, .7, .6, 35, 10, None])
        self.assertEqual(edited["job_id"], job_id)
        self.assertFalse(edited["editor"]["visible"])
        self.assertIsNone(edited["psd_file"])
        self.assertEqual(self.ui.jobs.layer_paths(job_id), paths)
        self.assertNotEqual(digest(paths[0]), before[0])
        self.assertEqual(digest(paths[1]), before[1])
        self.assertEqual(digest(self.ui.jobs.state_dir / job_id / "source.png"), source_hash)
        self.assertEqual(digest(self.ui.jobs.state_dir / job_id / "mask_001.npz"), other_mask_hash)
        mask, settings = self.ui.jobs.mask(job_id, 0)
        np.testing.assert_array_equal(mask["segmentation"], self.columns < 2)
        self.assertEqual(settings["confidence_threshold"], .7)
        await self.call("create_layers", [job_id])
        with Image.open(paths[0]) as image:
            np.testing.assert_array_equal(np.asarray(image)[:, :, 3] > 0, self.columns < 2)
        self.assertEqual(len(list(self.ui.jobs.state_dir.iterdir())), 1)

    async def test_cancel_discards_preview_and_changed_sliders_rerun_inference(self):
        job_id = await self.ready_job()
        await self.select(job_id)
        paths = self.ui.jobs.layer_paths(job_id)
        before = [digest(path) for path in paths]
        candidates = [{"segmentation": self.columns < 2, "area": 12, "prompt": "hair"}]
        with patch("ldivider.ld_ui.get_sam3_processor", return_value=object()), patch("ldivider.ld_ui.get_masks", return_value=candidates) as inference:
            await self.call("preview_layer", [job_id, None, .7, .6, 35, 10, None])
            closed = await self.call("close_editor", [])
            self.assertFalse(closed["editor"]["visible"])
            self.assertEqual([digest(path) for path in paths], before)
            await self.call("preview_layer", [job_id, None, .7, .6, 35, 10, None])
            await self.call("apply_layer_edit", [job_id, None, .8, .6, 35, 10, None])
        self.assertEqual(inference.call_count, 3)
        self.assertEqual(self.ui.jobs.mask(job_id, 0)[1]["confidence_threshold"], .8)

    async def test_restoring_a_job_needs_no_inference_and_keeps_edited_settings(self):
        job_id = await self.ready_job()
        settings = dict(confidence_threshold=.7, mask_threshold=.6, max_masks=35, min_mask_area=10)
        self.ui.jobs.replace_layer(job_id, 0, self.masks[0], settings)
        restarted = LayerDividerUI(self.root / "output", self.root / "models")
        state = SessionState(restarted.demo)
        with patch("ldivider.ld_ui.get_masks", side_effect=AssertionError("Restore must not infer")):
            restored = await self.call("restore_job", [job_id], state=state, ui=restarted)
        self.assertEqual(restored["job_id"], job_id)
        self.assertEqual(restored["text_prompt"], "hair\nshirt")
        self.assertEqual(len(restored["gallery"]), 2)
        await self.select(job_id, ui=restarted, state=state)
        opened = await self.call("open_layer_from_preview", [job_id, json.dumps({"index": 0})], state=state, ui=restarted)
        self.assertEqual(opened["edit_prompt"], "hair")
        self.assertEqual(opened["edit_confidence_threshold"], .7)
        unknown = await self.call("restore_job", ["0" * 32], state=state, ui=restarted)
        self.assertEqual(unknown["job_id"], "")

    async def test_failed_regeneration_keeps_existing_files(self):
        job_id = await self.ready_job()
        await self.select(job_id)
        paths = self.ui.jobs.layer_paths(job_id)
        before = [digest(path) for path in paths]
        with patch("ldivider.ld_ui.get_sam3_processor", return_value=object()), patch("ldivider.ld_ui.get_masks", side_effect=ValueError("No masks found")):
            with self.assertRaisesRegex(gr.Error, "No masks found"):
                await self.call("apply_layer_edit", [job_id, None, .9, .8, 50, 100, None])
        self.assertEqual([digest(path) for path in paths], before)

    async def test_psd_uses_the_current_jobs_layers_and_is_downloadable(self):
        with self.assertRaisesRegex(gr.Error, "Segment"):
            await self.call("generate_psd", [""])
        job_id = (await self.segment())["job_id"]
        with self.assertRaisesRegex(gr.Error, "Create Layers"):
            await self.call("generate_psd", [job_id])
        await self.call("create_layers", [job_id])
        result = await self.call("generate_psd", [job_id])
        saved = list((self.root / "output" / "jobs" / "psd").glob("_*.psd"))
        self.assertEqual(len(saved), 1)
        psd = PSDImage.open(saved[0])
        self.assertEqual([layer.name for layer in psd], ["001_hair", "002_shirt"])
        self.assertEqual(psd[0].topil().getpixel((0, 0)), (100, 50, 200, 180))
        path = quote(result["psd_file"]["name"].replace("\\", "/"), safe="/:")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.ui.demo.app), base_url="http://gradio.test") as client:
            response = await client.get("/file=" + path)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, saved[0].read_bytes())

    async def test_delete_removes_saved_files_and_stays_deleted_after_restore_and_create(self):
        job_id = await self.ready_job()
        paths = [Path(path) for path in self.ui.jobs.layer_paths(job_id)]
        before = digest(paths[1])
        private = self.ui.jobs.state_dir / job_id
        source_hash = digest(private / "source.png")
        await self.call("generate_psd", [job_id])
        request = json.dumps({"action": "delete", "index": 0, "filename": paths[0].name})
        deleted = await self.call("open_layer_from_preview", [job_id, request])
        self.assertFalse(paths[0].exists())
        self.assertFalse((private / "mask_000.npz").exists())
        self.assertEqual(digest(paths[1]), before)
        self.assertEqual(digest(private / "source.png"), source_hash)
        self.assertEqual(deleted["job_id"], job_id)
        self.assertEqual(len(deleted["gallery"]), 1)
        self.assertEqual(len(deleted["files"]), 1)
        self.assertIsNone(deleted["psd_file"])
        self.assertFalse(deleted["editor"]["visible"])
        self.assertIsNone(self.state[self.ui.components["draft"]._id])
        # A duplicate/stale click must never delete the layer that moved to index 0.
        await self.call("open_layer_from_preview", [job_id, request])
        recreated = await self.call("create_layers", [job_id])
        self.assertEqual(len(recreated["gallery"]), 1)
        self.assertEqual(self.ui.jobs.layer_paths(job_id), [str(paths[1])])
        self.assertFalse(paths[0].exists())
        restarted = LayerDividerUI(self.root / "output", self.root / "models")
        restored = await self.call("restore_job", [job_id], state=SessionState(restarted.demo), ui=restarted)
        self.assertEqual(len(restored["gallery"]), 1)
        result = await self.call("generate_psd", [job_id])
        psd = PSDImage.open(result["psd_file"]["name"])
        self.assertEqual([layer.name for layer in psd], ["002_shirt"])

    async def test_deleting_all_layers_keeps_the_job_without_recreating_layers(self):
        job_id = await self.ready_job()
        for index, path in enumerate(self.ui.jobs.layer_paths(job_id)):
            result = await self.call("open_layer_from_preview", [job_id, json.dumps({
                "action": "delete", "index": index, "filename": Path(path).name,
            })])
        self.assertEqual(result["job_id"], job_id)
        self.assertEqual(result["gallery"], [])
        self.assertEqual(result["files"], [])
        self.assertEqual(list((self.ui.jobs.layers_dir / job_id).iterdir()), [])
        self.assertEqual(list((self.ui.jobs.state_dir / job_id).glob("mask_*.npz")), [])
        self.assertEqual((await self.call("create_layers", [job_id]))["gallery"], [])
        self.assertEqual((await self.call("restore_job", [job_id]))["job_id"], job_id)
        with Image.open(self.ui.jobs.preview_path(job_id)) as image:
            np.testing.assert_array_equal(np.asarray(image), np.asarray(self.source))
        with self.assertRaisesRegex(gr.Error, "No layers remain"):
            await self.call("generate_psd", [job_id])

    async def test_old_job_mask_filenames_survive_deleting_a_middle_layer(self):
        self.masks.append({"segmentation": self.columns == 7, "area": 6, "prompt": "shoe"})
        job_id = await self.ready_job()
        paths = [Path(path) for path in self.ui.jobs.layer_paths(job_id)]
        # Simulate jobs saved before masks had persistent filenames in job.json.
        manifest = self.ui.jobs.state_dir / job_id / "job.json"
        job = json.loads(manifest.read_text(encoding="utf-8"))
        for layer in job["layers"]:
            layer.pop("mask_file")
        manifest.write_text(json.dumps(job), encoding="utf-8")
        self.ui.jobs.delete_layer(job_id, paths[1].name)
        np.testing.assert_array_equal(self.ui.jobs.mask(job_id, 1)[0]["segmentation"], self.columns == 7)
        opened = await self.call("open_layer_from_preview", [job_id, json.dumps({
            "action": "edit", "index": 2, "filename": paths[2].name,
        })])
        self.assertEqual(opened["edit_prompt"], "shoe")
        updated = {"segmentation": self.columns >= 6, "area": 12, "prompt": "shoe"}
        with patch("ldivider.ld_ui.get_sam3_processor", return_value=object()), patch("ldivider.ld_ui.get_masks", return_value=[updated]):
            await self.call("apply_layer_edit", [job_id, None, .7, .6, 35, 10, None])
        self.assertEqual(self.ui.jobs.layer_paths(job_id), [str(paths[0]), str(paths[2])])
        self.assertFalse((manifest.parent / "mask_001.npz").exists())
        with np.load(manifest.parent / "mask_002.npz") as mask:
            np.testing.assert_array_equal(mask["segmentation"], self.columns >= 6)

    async def test_editor_keeps_its_target_if_another_session_deletes_an_earlier_layer(self):
        job_id = await self.ready_job()
        paths = [Path(path) for path in self.ui.jobs.layer_paths(job_id)]
        await self.call("open_layer_from_preview", [job_id, json.dumps({"index": 1, "filename": paths[1].name})])
        self.ui.jobs.delete_layer(job_id, paths[0].name)
        updated = {"segmentation": self.columns >= 6, "area": 12, "prompt": "shirt"}
        with patch("ldivider.ld_ui.get_sam3_processor", return_value=object()), patch("ldivider.ld_ui.get_masks", return_value=[updated]) as inference:
            await self.call("preview_layer", [job_id, None, .7, .6, 35, 10, None])
            await self.call("apply_layer_edit", [job_id, None, .7, .6, 35, 10, None])
        self.assertEqual(inference.call_count, 1)
        self.assertEqual(inference.call_args.args[2], "shirt")
        np.testing.assert_array_equal(self.ui.jobs.mask(job_id, 0)[0]["segmentation"], self.columns >= 6)

    async def test_editor_rejects_its_target_deleted_in_another_session(self):
        job_id = await self.ready_job()
        paths = [Path(path) for path in self.ui.jobs.layer_paths(job_id)]
        await self.call("open_layer_from_preview", [job_id, json.dumps({"index": 0, "filename": paths[0].name})])
        self.ui.jobs.delete_layer(job_id, paths[0].name)
        before = digest(paths[1])
        with patch("ldivider.ld_ui.get_masks", side_effect=AssertionError("Deleted target must not infer")):
            for callback in ("preview_layer", "apply_layer_edit"):
                with self.assertRaisesRegex(gr.Error, "deleted"):
                    await self.call(callback, [job_id, None, .7, .6, 35, 10, None])
        self.assertEqual(digest(paths[1]), before)

    async def test_delete_rejects_paths_outside_the_layer_folder(self):
        job_id = await self.ready_job()
        paths = self.ui.jobs.layer_paths(job_id)
        before = [digest(path) for path in paths]
        for filename in (None, "../source.png", "..\\source.png", str(Path(paths[0]).resolve())):
            with self.assertRaisesRegex(gr.Error, "valid layer"):
                await self.call("open_layer_from_preview", [job_id, json.dumps({
                    "action": "delete", "index": 0, "filename": filename,
                })])
        self.assertEqual([digest(path) for path in paths], before)


if __name__ == "__main__":
    unittest.main()
