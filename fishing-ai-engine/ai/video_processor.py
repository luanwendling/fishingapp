"""Video IO and frame extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameSample:
    """Represents one sampled frame and its timestamp."""

    index: int
    timestamp_ms: int
    frame: np.ndarray


class VideoProcessor:
    """Read a video and extract frames with fixed temporal spacing."""

    def __init__(self, frame_interval_ms: int, max_frames: int) -> None:
        self.frame_interval_ms = frame_interval_ms
        self.max_frames = max_frames

    def extract_frames(self, video_path: str | Path) -> list[FrameSample]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        duration_ms = 0
        if frame_count > 0 and fps > 0:
            duration_ms = int((frame_count / fps) * 1000.0)

        if duration_ms <= 0:
            # Conservative fallback when metadata is missing.
            timestamps = list(range(0, self.frame_interval_ms * self.max_frames, self.frame_interval_ms))
        else:
            timestamps = list(range(0, max(1, duration_ms + 1), self.frame_interval_ms))

        if len(timestamps) > self.max_frames and self.max_frames > 0:
            selected_idx = np.linspace(0, len(timestamps) - 1, self.max_frames, dtype=int)
            timestamps = [timestamps[i] for i in selected_idx]

        frames: list[FrameSample] = []

        try:
            for idx, timestamp_ms in enumerate(timestamps):
                capture.set(cv2.CAP_PROP_POS_MSEC, int(timestamp_ms))
                ok, frame = capture.read()
                if not ok:
                    continue

                frames.append(
                    FrameSample(
                        index=idx,
                        timestamp_ms=int(timestamp_ms),
                        frame=frame,
                    )
                )
        finally:
            capture.release()

        return frames
