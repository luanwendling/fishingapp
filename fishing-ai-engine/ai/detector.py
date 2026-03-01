"""YOLOv8 detector wrapper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .utils import Detection


class YoloDetector:
    """Load and run YOLO model, returning normalized detections."""

    def __init__(self, model_path: str | Path, target_classes: tuple[str, ...]) -> None:
        self.model_path = Path(model_path)
        self.target_classes = set(target_classes)

        project_root = Path(__file__).resolve().parents[1]
        os.environ.setdefault("YOLO_CONFIG_DIR", str(project_root / ".yolo"))
        os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mpl"))

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {self.model_path}. "
                "Add a trained model with fish/ruler classes."
            )

        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Unable to import ultralytics YOLO.") from exc

        self._model = YOLO(str(self.model_path))
        self._names = self._resolve_class_names()

    def _resolve_class_names(self) -> dict[int, str]:
        names = self._model.names
        if isinstance(names, dict):
            normalized = {int(k): str(v) for k, v in names.items()}
        else:
            normalized = {idx: str(name) for idx, name in enumerate(names)}

        missing = [label for label in self.target_classes if label not in normalized.values()]
        if missing:
            raise ValueError(
                "Model classes missing required labels: "
                f"{', '.join(missing)}. Found: {sorted(set(normalized.values()))}"
            )
        return normalized

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self._model.predict(frame, verbose=False)
        if not results:
            return []

        detections: list[dict[str, Any]] = []
        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_idx = int(box.cls.item())
            class_name = self._names.get(cls_idx)
            if class_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det = Detection(
                object_class=class_name,
                confidence=float(box.conf.item()),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
            )
            detections.append(det.to_dict())

        return detections
