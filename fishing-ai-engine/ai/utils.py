"""Small utility helpers used across modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """Standardized detection format for a single object."""

    object_class: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "class": self.object_class,
            "confidence": float(self.confidence),
            "bbox": [float(v) for v in self.bbox],
        }


def bbox_width(bbox: tuple[float, float, float, float]) -> float:
    """Return bbox horizontal size in pixels."""

    return max(0.0, float(bbox[2] - bbox[0]))


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp value between limits."""

    return max(minimum, min(maximum, value))
