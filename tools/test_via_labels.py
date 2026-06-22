import json
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from via_labels import labels_to_via, via_to_labels


class ViaLabelsTest(unittest.TestCase):
    def make_tempdir(self):
        test_tmp = Path.cwd() / "target" / "test-tmp"
        path = test_tmp / f"via-labels-{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def test_labels_to_via_uses_annotation_dimensions(self):
        root = self.make_tempdir()
        labels_dir = root / "labels"
        labels_dir.mkdir()
        (labels_dir / "frame001.txt").write_text(
            "0 0.500000 0.500000 0.250000 0.200000\n", encoding="utf-8"
        )
        annotations_path = root / "annotations.jsonl"
        annotations_path.write_text(
            json.dumps(
                {
                    "image": "data/generated_samples/images/frame001.jpg",
                    "image_width": 1280,
                    "image_height": 720,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        via = labels_to_via(
            labels_dir=labels_dir,
            images_dir=root / "images",
            annotations_path=annotations_path,
        )

        item = next(iter(via.values()))
        self.assertEqual(item["filename"], "frame001.jpg")
        self.assertEqual(
            item["regions"][0]["shape_attributes"],
            {"name": "rect", "x": 480, "y": 288, "width": 320, "height": 144},
        )
        self.assertEqual(item["regions"][0]["region_attributes"], {"class": "subtitle"})

    def test_via_to_labels_writes_normalized_yolo_labels(self):
        root = self.make_tempdir()
        output_dir = root / "labels"
        via = {
            "frame001.jpg-0": {
                "filename": "frame001.jpg",
                "size": 0,
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "rect",
                            "x": 480,
                            "y": 288,
                            "width": 320,
                            "height": 144,
                        },
                        "region_attributes": {"class": "subtitle"},
                    }
                ],
                "file_attributes": {"width": 1280, "height": 720},
            }
        }

        via_to_labels(via, output_dir)

        self.assertEqual(
            (output_dir / "frame001.txt").read_text(encoding="utf-8"),
            "0 0.500000 0.500000 0.250000 0.200000\n",
        )


if __name__ == "__main__":
    unittest.main()
