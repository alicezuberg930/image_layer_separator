import tempfile
import unittest
import uuid
from pathlib import Path


import numpy as np
from PIL import Image
from psd_tools import PSDImage

from ldivider.ld_psd import export_layers_to_psd


PROJECT_DIR = Path(__file__).resolve().parent.parent


class PsdExportTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory(prefix=".psd-check-", dir=PROJECT_DIR)
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name)
        self.destination = self.root / "output" / "jobs" / "psd" / "psd"
        self.layers = []
        self.paths = []
        for name, color, box in (
            ("001_hair", (255, 0, 0, 128), (0, 0, 5, 5)),
            ("002_shirt", (0, 0, 255, 200), (3, 1, 7, 5)),
        ):
            layer = Image.new("RGBA", (8, 6), (0, 0, 0, 0))
            layer.paste(color, box)
            path = self.root / f"{name}.png"
            layer.save(path)
            self.layers.append(layer)
            self.paths.append(path)

    def test_saved_psd_preserves_all_layers_names_pixels_and_transparency(self):
        path = Path(export_layers_to_psd(self.paths, self.destination))
        self.assertEqual(path.parent, self.destination)
        self.assertEqual(path.read_bytes()[:4], b"8BPS")
        psd = PSDImage.open(path)
        self.assertEqual(psd.size, (8, 6))
        self.assertEqual(psd.depth, 8)
        self.assertEqual([layer.name for layer in psd], [path.stem for path in self.paths])
        for layer, expected in zip(psd, self.layers):
            self.assertTrue(layer.visible)
            self.assertEqual(layer.bbox, (0, 0, 8, 6))
            np.testing.assert_array_equal(np.asarray(layer.topil()), np.asarray(expected))

        merged = np.asarray(psd.composite(force=True, apply_icc=False))
        expected = np.asarray(Image.alpha_composite(*self.layers))
        np.testing.assert_allclose(merged[:, :, 3], expected[:, :, 3], atol=1)
        visible = expected[:, :, 3] > 0
        np.testing.assert_allclose(merged[visible, :3], expected[visible, :3], atol=1)

    def test_every_export_has_a_new_uuid_without_overwriting_files(self):
        first = Path(export_layers_to_psd(self.paths, self.destination))
        original = first.read_bytes()
        second = Path(export_layers_to_psd(self.paths, self.destination))
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), original)
        for path in (first, second):
            self.assertTrue(path.stem.startswith("_"))
            self.assertEqual(uuid.UUID(path.stem[1:]).version, 4)
            self.assertEqual(path.suffix, ".psd")

    def test_no_layers_reports_the_required_workflow(self):
        with self.assertRaisesRegex(ValueError, "Click Create Layers"):
            export_layers_to_psd([], self.destination)
        self.assertFalse(self.destination.exists())

    def test_mismatched_canvas_sizes_do_not_create_a_psd(self):
        Image.new("RGBA", (4, 4)).save(self.paths[1])
        with self.assertRaisesRegex(ValueError, "same canvas size"):
            export_layers_to_psd(self.paths, self.destination)
        self.assertFalse(self.destination.exists())

    def test_a_fully_transparent_layer_is_retained(self):
        Image.new("RGBA", (8, 6)).save(self.paths[0])
        psd = PSDImage.open(export_layers_to_psd(self.paths, self.destination))
        self.assertEqual(len(psd), 2)
        self.assertEqual(psd[0].topil().getchannel("A").getbbox(), None)

    def test_sparse_canvas_has_a_small_lossless_merged_preview(self):
        source = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        source.paste((120, 40, 200, 160), (200, 200, 260, 260))
        path = self.root / "sparse.png"
        source.save(path)
        saved = Path(export_layers_to_psd([path], self.destination))
        raw_preview_bytes = 512 * 512 * 4
        self.assertLess(saved.stat().st_size, raw_preview_bytes // 10)
        psd = PSDImage.open(saved)
        self.assertEqual(psd.size, source.size)
        np.testing.assert_array_equal(np.asarray(psd[0].topil()), np.asarray(source))



if __name__ == "__main__":
    unittest.main()
