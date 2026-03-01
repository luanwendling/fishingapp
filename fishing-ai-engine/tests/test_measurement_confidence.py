import unittest

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


if __name__ == "__main__":
    unittest.main()
