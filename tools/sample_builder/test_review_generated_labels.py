import json
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_generated_labels import ReviewApp, merge_candidate_groups, merge_label_boxes, parse_label_file, write_label_box


class ReviewGeneratedLabelsTest(unittest.TestCase):
    def make_tempdir(self):
        test_tmp = Path.cwd() / "target" / "test-tmp"
        path = test_tmp / f"review-labels-{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def test_write_label_box_updates_pixel_rect_as_normalized_yolo(self):
        root = self.make_tempdir()
        label_path = root / "frame001.txt"
        label_path.write_text(
            "0 0.500000 0.500000 0.250000 0.200000\n"
            "0 0.100000 0.100000 0.100000 0.100000\n",
            encoding="utf-8",
        )

        write_label_box(
            label_path,
            0,
            {"x": 480, "y": 288, "width": 320, "height": 144},
            image_width=1280,
            image_height=720,
        )

        self.assertEqual(
            label_path.read_text(encoding="utf-8"),
            "0 0.500000 0.500000 0.250000 0.200000\n"
            "0 0.100000 0.100000 0.100000 0.100000\n",
        )

    def test_write_label_box_clamps_to_image_bounds(self):
        root = self.make_tempdir()
        label_path = root / "frame001.txt"
        label_path.write_text("0 0.500000 0.500000 0.250000 0.200000\n", encoding="utf-8")

        write_label_box(
            label_path,
            0,
            {"x": -10, "y": 90, "width": 120, "height": 40},
            image_width=100,
            image_height=100,
        )

        self.assertEqual(
            label_path.read_text(encoding="utf-8"),
            "0 0.500000 0.950000 1.000000 0.100000\n",
        )

    def test_merge_candidate_groups_detects_near_similar_height_boxes(self):
        boxes = [
            {"index": 0, "x": 10, "y": 100, "width": 80, "height": 28, "masked": False},
            {"index": 1, "x": 98, "y": 102, "width": 90, "height": 30, "masked": False},
            {"index": 2, "x": 260, "y": 100, "width": 90, "height": 52, "masked": False},
            {"index": 3, "x": 110, "y": 170, "width": 70, "height": 28, "masked": True},
        ]

        self.assertEqual(
            merge_candidate_groups(boxes),
            [
                {
                    "indices": [0, 1],
                    "rect": {
                        "x": 10.0,
                        "y": 100.0,
                        "width": 178.0,
                        "height": 32.0,
                        "area": 5696.0,
                        "aspect": 5.5625,
                    },
                }
            ],
        )

    def test_merge_label_boxes_writes_union_to_first_label_only(self):
        root = self.make_tempdir()
        label_path = root / "frame001.txt"
        label_path.write_text(
            "0 0.250000 0.500000 0.200000 0.100000\n"
            "0 0.450000 0.500000 0.200000 0.100000\n"
            "0 0.800000 0.500000 0.100000 0.100000\n",
            encoding="utf-8",
        )

        keep_index = merge_label_boxes(
            label_path,
            [1, 0],
            {"x": 15, "y": 45, "width": 40, "height": 10},
            image_width=100,
            image_height=100,
        )

        self.assertEqual(keep_index, 0)
        self.assertEqual(
            label_path.read_text(encoding="utf-8"),
            "0 0.350000 0.500000 0.400000 0.100000\n"
            "0 0.450000 0.500000 0.200000 0.100000\n"
            "0 0.800000 0.500000 0.100000 0.100000\n",
        )
        self.assertEqual([box.index for box in parse_label_file(label_path)], [0, 1, 2])

    def test_review_app_merge_boxes_masks_merged_source_labels(self):
        root = self.make_tempdir()
        images_dir = root / "images"
        labels_dir = root / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        (images_dir / "frame001.jpg").write_bytes(b"placeholder")
        (root / "annotations.jsonl").write_text(
            json.dumps({"image": "frame001.jpg", "image_width": 100, "image_height": 100}) + "\n",
            encoding="utf-8",
        )
        (labels_dir / "frame001.txt").write_text(
            "0 0.250000 0.500000 0.200000 0.100000\n"
            "0 0.450000 0.500000 0.200000 0.100000\n",
            encoding="utf-8",
        )

        result = ReviewApp(root).merge_boxes("frame001", [0, 1])

        self.assertEqual(result["keep_index"], 0)
        self.assertEqual(result["masked_indices"], [1])
        self.assertEqual(
            (labels_dir / "frame001.txt").read_text(encoding="utf-8"),
            "0 0.350000 0.500000 0.400000 0.100000\n"
            "0 0.450000 0.500000 0.200000 0.100000\n",
        )
        masks = json.loads((root / "label_masks.json").read_text(encoding="utf-8"))
        self.assertEqual(masks["items"]["frame001"]["1"]["reason"], "merge")

    def test_review_app_merge_candidates_flattens_all_images(self):
        root = self.make_tempdir()
        images_dir = root / "images"
        labels_dir = root / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        for stem in ("frame001", "frame002"):
            (images_dir / f"{stem}.jpg").write_bytes(b"placeholder")
        (root / "annotations.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"image": "frame001.jpg", "image_width": 100, "image_height": 100}),
                    json.dumps({"image": "frame002.jpg", "image_width": 100, "image_height": 100}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (labels_dir / "frame001.txt").write_text(
            "0 0.250000 0.500000 0.200000 0.100000\n"
            "0 0.450000 0.500000 0.200000 0.100000\n",
            encoding="utf-8",
        )
        (labels_dir / "frame002.txt").write_text(
            "0 0.250000 0.500000 0.200000 0.100000\n",
            encoding="utf-8",
        )

        payload = ReviewApp(root).merge_candidates({})

        self.assertEqual(payload["stats"], {"merge_candidates": 1})
        self.assertEqual(payload["items"][0]["stem"], "frame001")
        self.assertEqual(payload["items"][0]["indices"], [0, 1])


if __name__ == "__main__":
    unittest.main()
