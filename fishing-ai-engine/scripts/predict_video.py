#!/usr/bin/env python3
"""Run prediction on reference video using trained model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fish+ruler detections on video")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/luan/IdeaProjects/FishingAPP/fishing-ai-engine/ai/models/best.pt"),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/luan/IdeaProjects/FishingAPP/obsidian/fishing/video-modelo/peixe.mp4"),
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("/home/luan/IdeaProjects/FishingAPP/fishing-ai-engine/runs/predict"),
    )
    parser.add_argument("--name", type=str, default="fishing_ai_v1_predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", "/home/luan/IdeaProjects/FishingAPP/fishing-ai-engine/.yolo")

    from ultralytics import YOLO

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.source.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")

    args.project.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    model.predict(
        source=str(args.source),
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        save=True,
        verbose=False,
    )

    print(f"Prediction artifacts written to: {args.project / args.name}")


if __name__ == "__main__":
    main()
