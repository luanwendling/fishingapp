"""CLI and orchestration for Fishing AI measurement pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from .confidence import ConfidenceScorer
from .config import Settings
from .detector import YoloDetector
from .measurement import MeasurementEngine
from .video_processor import VideoProcessor


class PipelineOutput(BaseModel):
    """Final MVP payload."""

    length_cm: float
    confidence_score: float
    status: str = "review"


def run_pipeline(video_path: str | Path, settings: Settings) -> PipelineOutput:
    processor = VideoProcessor(
        frame_interval_ms=settings.frame_interval_ms,
        max_frames=settings.max_frames,
    )
    detector = YoloDetector(
        model_path=settings.model_path,
        target_classes=settings.required_classes,
    )
    measurer = MeasurementEngine(
        ruler_length_cm=settings.ruler_length_cm,
        top_k_frames=settings.top_k_frames,
    )
    scorer = ConfidenceScorer()

    frames = processor.extract_frames(video_path)
    detections_per_frame = [detector.detect(sample.frame) for sample in frames]

    per_frame_measurements = measurer.measure_from_detections(
        frames=[sample.frame for sample in frames],
        frame_detections=detections_per_frame,
    )
    avg_length, selected = measurer.average_best_frames(per_frame_measurements)
    confidence = scorer.score(selected)

    return PipelineOutput(
        length_cm=round(float(avg_length), 2),
        confidence_score=confidence,
        status="review",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fishing AI offline measurement pipeline")
    parser.add_argument("--video", required=True, help="Path to input .mp4 video")
    parser.add_argument(
        "--model",
        default="ai/models/best.pt",
        help="Path to YOLOv8 model with fish/ruler classes",
    )
    parser.add_argument(
        "--frame-interval-ms",
        type=int,
        default=200,
        help="Frame sampling interval in milliseconds",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Maximum number of sampled frames",
    )
    parser.add_argument(
        "--ruler-length-cm",
        type=float,
        default=40.0,
        help="Physical ruler length in centimeters",
    )
    parser.add_argument(
        "--top-k-frames",
        type=int,
        default=5,
        help="How many best frames to average",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings(
        model_path=Path(args.model),
        frame_interval_ms=args.frame_interval_ms,
        max_frames=args.max_frames,
        ruler_length_cm=args.ruler_length_cm,
        top_k_frames=args.top_k_frames,
    )
    output = run_pipeline(args.video, settings)
    print(json.dumps(output.model_dump(), indent=2))


if __name__ == "__main__":
    main()
