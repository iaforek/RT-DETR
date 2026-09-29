"""Dataset integrity checks; no PyTorch or pytest required."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from experiment import generate


class DatasetIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / "data"
        with contextlib.redirect_stdout(io.StringIO()):
            generate(SimpleNamespace(data=str(cls.root), seed=42, train=24, val=8, test=8))
        cls.records = json.loads((cls.root / "manifest.json").read_text())["records"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_encodings_preserve_rgb_and_raw_palette(self):
        for record in self.records:
            folder = self.root / record["path"]
            with Image.open(folder / "image.png") as image:
                rgb = np.array(image.convert("RGB"))
            raw = np.fromfile(folder / "indices.bin", dtype=np.uint8).reshape(64, 64)
            palette = np.fromfile(folder / "palette.rgb", dtype=np.uint8).reshape(256, 3)
            np.testing.assert_array_equal(rgb, palette[raw])
            self.assertEqual((folder / "indices.bin").stat().st_size, 4096)
            for variant in ("image_level0.png", "image_level9.png", "image_reindexed.png", "image.gif"):
                with Image.open(folder / variant) as image:
                    np.testing.assert_array_equal(rgb, np.array(image.convert("RGB")))

    def test_diffs_masks_and_boxes_agree(self):
        classes = set()
        for record in self.records:
            classes.add(record["class"])
            folder = self.root / record["path"]
            with Image.open(folder / "mask.png") as mask:
                expected_box = mask.getbbox() or (0, 0, 0, 0)
                values = np.array(mask)
            with Image.open(folder / "diff.png") as diff:
                np.testing.assert_array_equal(values, np.array(diff))
            self.assertEqual(record["box"], [v / 64 for v in expected_box])
            self.assertEqual(bool(record["class"]), bool(values.any()))
        self.assertEqual(classes, {0, 1, 2, 3})

    def test_splits_and_serializations(self):
        paths = [r["path"] for r in self.records]
        self.assertEqual(len(paths), len(set(paths)))
        for record in self.records:
            self.assertEqual(Path(record["path"]).parts[0], record["split"])
            folder = self.root / record["path"]
            self.assertNotEqual((folder / "image.png").read_bytes(),
                                (folder / "image_level0.png").read_bytes())


if __name__ == "__main__":
    unittest.main()
