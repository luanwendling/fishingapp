import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ai.confidence import ConfidenceScorer
from ai.measurement import MeasurementEngine


def make_frame(fish_box: tuple[int, int, int, int], size: tuple[int, int] = (200, 400)) -> np.ndarray:
    h, w = size
    frame = np.full((h, w, 3), 240, dtype=np.uint8)
    x1, y1, x2, y2 = fish_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 140, 40), -1)
    return frame


class MeasurementConfidenceTests(unittest.TestCase):
    def test_measurement_average_best_frames(self) -> None:
        engine = MeasurementEngine(ruler_length_cm=40.0, top_k_frames=2)

        frames = [
            make_frame((100, 80, 250, 120)),
            make_frame((105, 80, 255, 120)),
            make_frame((200, 80, 240, 120)),  # outlier
        ]

        frame_detections = [
            [
                {"class": "fish", "confidence": 0.95, "bbox": [100, 80, 250, 120]},
                {"class": "ruler", "confidence": 0.96, "bbox": [40, 60, 360, 140]},
            ],
            [
                {"class": "fish", "confidence": 0.93, "bbox": [105, 80, 255, 120]},
                {"class": "ruler", "confidence": 0.92, "bbox": [40, 60, 360, 140]},
            ],
            [
                {"class": "fish", "confidence": 0.91, "bbox": [200, 80, 240, 120]},
                {"class": "ruler", "confidence": 0.95, "bbox": [40, 60, 360, 140]},
            ],
        ]

        per_frame = engine.measure_from_detections(frames=frames, frame_detections=frame_detections)
        avg_length, selected = engine.average_best_frames(per_frame)

        self.assertEqual(len(selected), 2)
        self.assertGreater(avg_length, 17.0)
        self.assertLess(avg_length, 25.0)

    def test_confidence_is_bounded(self) -> None:
        engine = MeasurementEngine(ruler_length_cm=40.0, top_k_frames=2)

        frames = [
            make_frame((100, 80, 250, 120)),
            make_frame((104, 80, 254, 120)),
        ]

        frame_detections = [
            [
                {"class": "fish", "confidence": 0.9, "bbox": [100, 80, 250, 120]},
                {"class": "ruler", "confidence": 0.95, "bbox": [40, 60, 360, 140]},
            ],
            [
                {"class": "fish", "confidence": 0.88, "bbox": [104, 80, 254, 120]},
                {"class": "ruler", "confidence": 0.9, "bbox": [40, 60, 360, 140]},
            ],
        ]

        per_frame = engine.measure_from_detections(frames=frames, frame_detections=frame_detections)
        _, selected = engine.average_best_frames(per_frame)
        score = ConfidenceScorer().score(selected)

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_visible_number_estimation_supports_vertical_ruler(self) -> None:
        engine = MeasurementEngine(ruler_length_cm=40.0, top_k_frames=2)

        frame = np.full((300, 200, 3), 255, dtype=np.uint8)
        ruler_bbox = (50.0, 20.0, 110.0, 240.0)
        fish_bbox = (58.0, 30.0, 100.0, 195.0)

        # Vertical ruler with two red number blobs.
        cv2.rectangle(frame, (62, 100), (78, 145), (0, 0, 255), -1)
        cv2.rectangle(frame, (82, 150), (98, 195), (0, 0, 255), -1)
        # Markers to let mocked OCR identify each blob ROI.
        cv2.rectangle(frame, (65, 112), (70, 117), (255, 0, 0), -1)
        cv2.rectangle(frame, (85, 162), (90, 167), (0, 255, 0), -1)

        def fake_ocr(roi_bgr: np.ndarray) -> int:
            b_mean = float(np.mean(roi_bgr[:, :, 0]))
            g_mean = float(np.mean(roi_bgr[:, :, 1]))
            return 30 if b_mean >= g_mean else 40

        with patch.object(MeasurementEngine, "_ocr_red_number", side_effect=fake_ocr):
            with patch.object(MeasurementEngine, "_calculate_pixels_per_cm_from_ticks", return_value=5.0):
                length_cm = engine._estimate_length_from_visible_number(frame, ruler_bbox, fish_bbox)

        self.assertIsNotNone(length_cm)
        self.assertAlmostEqual(float(length_cm), 44.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
