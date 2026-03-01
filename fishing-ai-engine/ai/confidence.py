"""Confidence score composition for final output."""

from __future__ import annotations

import numpy as np

from .measurement import FrameMeasurement
from .utils import clamp


class ConfidenceScorer:
    """Combine model confidence and frame consistency into one score."""

    def score(self, selected_frames: list[FrameMeasurement]) -> float:
        if not selected_frames:
            return 0.0

        yolo_conf = float(
            np.mean([(m.fish_confidence + m.ruler_confidence) / 2.0 for m in selected_frames])
        )

        lengths = np.array([m.length_cm for m in selected_frames], dtype=float)
        if len(lengths) == 1 or float(np.mean(lengths)) == 0.0:
            stability = 0.7
        else:
            coef_var = float(np.std(lengths) / np.mean(lengths))
            stability = 1.0 - clamp(coef_var, 0.0, 1.0)

        final_score = (0.7 * yolo_conf) + (0.3 * stability)
        return round(clamp(final_score, 0.0, 1.0), 4)
