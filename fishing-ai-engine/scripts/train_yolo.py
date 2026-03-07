#!/usr/bin/env python3
"""Train YOLOv8 fish+ruler model and copy best.pt to engine models folder."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 fish+ruler model")
    parser.add_argument("--data", type=Path, default=Path("/home/luan/IdeaProjects/FishingAPP/dataset/data.yaml"))
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", type=str, default="fishing_ai_v2")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("/home/luan/IdeaProjects/FishingAPP/fishing-ai-engine/runs/detect"),
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("/home/luan/IdeaProjects/FishingAPP/fishing-ai-engine/ai/models/best.pt"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device (e.g., 'cpu', '0', '0,1').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", "/home/luan/IdeaProjects/FishingAPP/fishing-ai-engine/.yolo")

    from ultralytics import YOLO

    args.project.mkdir(parents=True, exist_ok=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)

    model_id = args.model
    try:
        model = YOLO(model_id)
    except Exception:
        # Fallback if pretrained weights are unavailable in offline environment.
        model_id = "yolov8s.yaml"
        model = YOLO(model_id)

    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        name=args.name,
        project=str(args.project),
        exist_ok=True,
    )
    if args.device:
        train_kwargs["device"] = args.device

    result = model.train(**train_kwargs)

    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but best.pt not found: {best}")

    shutil.copy2(best, args.output_model)
    print(f"Copied best model to: {args.output_model}")


if __name__ == "__main__":
    main()
