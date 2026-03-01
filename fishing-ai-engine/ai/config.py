"""Configuration objects for the Fishing AI Engine."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Runtime configuration for the pipeline."""

    model_path: Path = Field(default=Path("ai/models/best.pt"))
    frame_interval_ms: int = Field(default=200, ge=1)
    max_frames: int = Field(default=120, ge=1)
    ruler_length_cm: float = Field(default=40.0, gt=0)
    top_k_frames: int = Field(default=5, ge=1)
    required_classes: tuple[str, str] = ("fish", "ruler")
