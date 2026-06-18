import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_samples import (
    DEFAULT_DET_MODEL_NAME,
    Detection,
    filter_detections_by_region,
    frame_index_selected,
    make_text_detection_options,
    normalize_quad,
    open_video_capture,
    parse_paddle_detections,
    parse_video_args,
    sample_stem,
    seek_video_capture,
    write_boxed_image,
    yolo_bbox_from_quad,
)


class SampleBuilderTests(unittest.TestCase):
    def test_parse_video_args_splits_multiple_inputs(self):
        args = parse_video_args(["a.mp4", "b.mkv"])

        self.assertEqual(args.videos, ["a.mp4", "b.mkv"])
        self.assertEqual(args.det_limit_side_len, 960)
        self.assertEqual(args.det_model_name, DEFAULT_DET_MODEL_NAME)
        self.assertEqual(args.start_frame, 0)
        self.assertEqual(args.video_backend, "opencv")
        self.assertTrue(args.yolo_labels)
        self.assertFalse(args.boxed_images)
        self.assertIsNone(args.filter_region)

    def test_parse_video_args_enables_boxed_images(self):
        args = parse_video_args(["--boxed-images", "a.mp4"])

        self.assertTrue(args.boxed_images)

    def test_parse_video_args_accepts_filter_region(self):
        args = parse_video_args(["--filter-region", "0,600,1920,816", "a.mp4"])

        self.assertEqual(args.filter_region, "0,600,1920,816")

    def test_parse_video_args_accepts_resume_start_frame(self):
        args = parse_video_args(["--start-frame", "3000", "a.mp4"])

        self.assertEqual(args.start_frame, 3000)

    def test_parse_video_args_accepts_detector_model_name(self):
        args = parse_video_args(["--det-model-name", "PP-OCRv4_server_det", "a.mp4"])

        self.assertEqual(args.det_model_name, "PP-OCRv4_server_det")

    def test_parse_video_args_accepts_ffmpeg_video_backend(self):
        args = parse_video_args(["--video-backend", "ffmpeg", "a.mp4"])

        self.assertEqual(args.video_backend, "ffmpeg")

    def test_frame_index_selected_respects_stride(self):
        self.assertTrue(frame_index_selected(0, 12))
        self.assertFalse(frame_index_selected(11, 12))
        self.assertTrue(frame_index_selected(12, 12))

    def test_normalize_quad_clamps_to_image_bounds(self):
        quad = [[-5, 10], [110, 10], [110, 60], [-5, 60]]

        self.assertEqual(
            normalize_quad(quad, width=100, height=50),
            [[0.0, 10.0], [99.0, 10.0], [99.0, 49.0], [0.0, 49.0]],
        )

    def test_yolo_bbox_from_quad_uses_normalized_center_and_size(self):
        detection = Detection(
            polygon=[[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]],
            score=0.92,
        )

        self.assertEqual(
            yolo_bbox_from_quad(detection, width=100, height=200),
            [0, 0.2, 0.15, 0.2, 0.1],
        )

    def test_filter_detections_by_region_keeps_center_inside_region(self):
        inside = Detection(
            polygon=[[10.0, 70.0], [30.0, 70.0], [30.0, 90.0], [10.0, 90.0]],
            score=0.92,
        )
        outside = Detection(
            polygon=[[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]],
            score=0.91,
        )

        self.assertEqual(
            filter_detections_by_region([inside, outside], (0, 50, 100, 100)),
            [inside],
        )

    def test_filter_detections_by_region_uses_source_frame_offset(self):
        detection = Detection(
            polygon=[[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]],
            score=0.92,
        )

        self.assertEqual(
            filter_detections_by_region(
                [detection], (0, 600, 100, 720), offset_x=0, offset_y=600
            ),
            [detection],
        )

    def test_filter_detections_by_region_keeps_bbox_intersecting_region(self):
        detection = Detection(
            polygon=[[10.0, 40.0], [30.0, 40.0], [30.0, 80.0], [10.0, 80.0]],
            score=0.92,
        )

        self.assertEqual(
            filter_detections_by_region([detection], (0, 70, 100, 100)),
            [detection],
        )

    def test_text_detection_options_disable_mkldnn_and_use_detector_only(self):
        self.assertEqual(
            make_text_detection_options(),
            {
                "model_name": "PP-OCRv5_server_det",
                "enable_mkldnn": False,
                "limit_side_len": 960,
            },
        )

    def test_text_detection_options_uses_requested_limit_side_len(self):
        self.assertEqual(
            make_text_detection_options(640),
            {
                "model_name": "PP-OCRv5_server_det",
                "enable_mkldnn": False,
                "limit_side_len": 640,
            },
        )

    def test_text_detection_options_uses_requested_model_name(self):
        self.assertEqual(
            make_text_detection_options(960, "PP-OCRv4_server_det"),
            {
                "model_name": "PP-OCRv4_server_det",
                "enable_mkldnn": False,
                "limit_side_len": 960,
            },
        )

    def test_parse_paddle_detections_reads_text_detection_result_dict(self):
        result = [
            {
                "dt_polys": [
                    [[10, 20], [30, 20], [30, 40], [10, 40]],
                    [[50, 60], [80, 60], [80, 90], [50, 90]],
                ],
                "dt_scores": [0.91, 0.42],
            }
        ]

        detections = parse_paddle_detections(result, width=100, height=100)

        self.assertEqual(
            detections,
            [
                Detection(
                    polygon=[
                        [10.0, 20.0],
                        [30.0, 20.0],
                        [30.0, 40.0],
                        [10.0, 40.0],
                    ],
                    score=0.91,
                ),
                Detection(
                    polygon=[
                        [50.0, 60.0],
                        [80.0, 60.0],
                        [80.0, 90.0],
                        [50.0, 90.0],
                    ],
                    score=0.42,
                ),
            ],
        )

    def test_parse_paddle_detections_accepts_array_like_scores(self):
        class ArrayLike:
            def __init__(self, value):
                self.value = value

            def __bool__(self):
                raise ValueError("ambiguous truth value")

            def tolist(self):
                return self.value

        result = {
            "dt_polys": ArrayLike([[[10, 20], [30, 20], [30, 40], [10, 40]]]),
            "dt_scores": ArrayLike([0.91]),
        }

        self.assertEqual(
            parse_paddle_detections(result, width=100, height=100),
            [
                Detection(
                    polygon=[
                        [10.0, 20.0],
                        [30.0, 20.0],
                        [30.0, 40.0],
                        [10.0, 40.0],
                    ],
                    score=0.91,
                )
            ],
        )

    def test_sample_stem_does_not_use_video_file_name(self):
        self.assertEqual(sample_stem("video0001", 150), "video0001_f00000150")

    def test_open_video_capture_uses_default_opencv_backend(self):
        class FakeCv2:
            def __init__(self):
                self.calls = []

            def VideoCapture(self, *args):
                self.calls.append(args)
                return "capture"

        cv2 = FakeCv2()

        self.assertEqual(open_video_capture(cv2, Path("a.mp4"), "opencv"), "capture")
        self.assertEqual(cv2.calls, [("a.mp4",)])

    def test_open_video_capture_uses_ffmpeg_backend(self):
        class FakeCv2:
            CAP_FFMPEG = 1900

            def __init__(self):
                self.calls = []

            def VideoCapture(self, *args):
                self.calls.append(args)
                return "capture"

        cv2 = FakeCv2()

        self.assertEqual(open_video_capture(cv2, Path("a.mp4"), "ffmpeg"), "capture")
        self.assertEqual(cv2.calls, [("a.mp4", 1900)])

    def test_seek_video_capture_uses_requested_start_frame(self):
        class FakeCv2:
            CAP_PROP_POS_FRAMES = 1

        class FakeCapture:
            def __init__(self):
                self.calls = []

            def set(self, prop, value):
                self.calls.append((prop, value))
                return True

        capture = FakeCapture()

        self.assertEqual(seek_video_capture(FakeCv2(), capture, 248910), 248910)
        self.assertEqual(capture.calls, [(1, 248910)])

    def test_seek_video_capture_falls_back_when_seek_fails(self):
        class FakeCv2:
            CAP_PROP_POS_FRAMES = 1

        class FakeCapture:
            def set(self, prop, value):
                return False

        self.assertEqual(seek_video_capture(FakeCv2(), FakeCapture(), 248910), 0)

    def test_write_boxed_image_draws_detection_polygons(self):
        class FakeImage:
            def __init__(self):
                self.copied = False

            def copy(self):
                copied = FakeImage()
                copied.copied = True
                return copied

        class FakeCv2:
            def __init__(self):
                self.lines = []
                self.writes = []

            def polylines(self, image, points, is_closed, color, thickness):
                self.lines.append((image, points, is_closed, color, thickness))

            def imwrite(self, path, image):
                self.writes.append((path, image))

        cv2 = FakeCv2()
        image = FakeImage()
        detection = Detection(
            polygon=[[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]],
            score=0.92,
        )

        write_boxed_image(cv2, Path("boxed.jpg"), image, [detection])

        self.assertEqual(len(cv2.lines), 1)
        self.assertTrue(cv2.lines[0][0].copied)
        self.assertTrue(cv2.lines[0][2])
        self.assertEqual(cv2.lines[0][3], (0, 255, 0))
        self.assertEqual(cv2.lines[0][4], 2)
        self.assertEqual(cv2.writes, [("boxed.jpg", cv2.lines[0][0])])


if __name__ == "__main__":
    unittest.main()
