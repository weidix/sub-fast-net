import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthesize_samples import (
    SampleBox,
    collect_image_paths,
    parse_subtitle_file,
    sample_stem,
    yolo_label_from_box,
)


class SyntheticSampleTests(unittest.TestCase):
    def test_parse_subtitle_file_reads_srt_text_blocks(self):
        path = Path("unused.srt")
        content = """1
00:00:01,000 --> 00:00:03,000
第一行字幕
第二行字幕

2
00:00:04,000 --> 00:00:05,000
下一句
"""

        subtitles = parse_subtitle_file(path, text=content)

        self.assertEqual(subtitles, ["第一行字幕\n第二行字幕", "下一句"])

    def test_parse_subtitle_file_reads_plain_lines(self):
        subtitles = parse_subtitle_file(
            Path("unused.txt"), text="第一句\n\n第二句\n"
        )

        self.assertEqual(subtitles, ["第一句", "第二句"])

    def test_yolo_label_from_box_uses_normalized_center_and_size(self):
        label = yolo_label_from_box(SampleBox(x=10, y=20, width=30, height=40), 100, 200)

        self.assertEqual(label, "0 0.250000 0.200000 0.300000 0.200000")

    def test_collect_image_paths_keeps_supported_images_sorted(self):
        paths = collect_image_paths(
            Path("images"),
            [
                Path("images/b.png"),
                Path("images/a.jpg"),
                Path("images/readme.txt"),
                Path("images/c.JPEG"),
            ],
        )

        self.assertEqual(
            paths,
            [Path("images/a.jpg"), Path("images/b.png"), Path("images/c.JPEG")],
        )

    def test_sample_stem_is_stable_and_zero_padded(self):
        self.assertEqual(sample_stem(12), "synthetic_000012")


if __name__ == "__main__":
    unittest.main()
