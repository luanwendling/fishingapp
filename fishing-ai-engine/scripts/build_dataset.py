#!/usr/bin/env python3
"""Build YOLO dataset from full videos using pseudo-label + heuristic fallback."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.w * self.h


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> BBox:
    return BBox(
        x1=max(0, min(int(x1), w - 1)),
        y1=max(0, min(int(y1), h - 1)),
        x2=max(1, min(int(x2), w)),
        y2=max(1, min(int(y2), h)),
    )


def to_yolo_line(class_id: int, bbox: BBox, img_w: int, img_h: int) -> str:
    cx = ((bbox.x1 + bbox.x2) / 2.0) / img_w
    cy = ((bbox.y1 + bbox.y2) / 2.0) / img_h
    bw = bbox.w / img_w
    bh = bbox.h / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def ensure_dirs(dataset_dir: Path, clean: bool) -> None:
    if clean and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for rel in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dataset_dir / rel).mkdir(parents=True, exist_ok=True)


def write_data_yaml(dataset_dir: Path) -> None:
    text = """path: ../dataset
train: images/train
val: images/val

names:
  0: fish
  1: ruler
"""
    (dataset_dir / "data.yaml").write_text(text, encoding="utf-8")


def collect_frames_interval(video_path: Path, interval_ms: int) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    duration_ms = int((total_frames / max(1e-6, fps)) * 1000.0) if total_frames > 0 else 0

    if duration_ms <= 0:
        cap.release()
        return []

    samples: list[tuple[int, np.ndarray]] = []
    for ts in range(0, duration_ms + 1, interval_ms):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts)
        ok, frame = cap.read()
        if ok and frame is not None:
            samples.append((ts, frame))

    cap.release()
    return samples


def load_pseudo_model(model_path: Path):
    if not model_path.exists():
        return None

    os.environ.setdefault("YOLO_CONFIG_DIR", "/home/luan/IdeaProjects/FishingAPP/fishing-ai-engine/.yolo")
    os.environ.setdefault("MPLCONFIGDIR", "/home/luan/IdeaProjects/FishingAPP/fishing-ai-engine/.mpl")

    from ultralytics import YOLO

    return YOLO(str(model_path))


def select_pair_from_model(
    model: Any,
    frame: np.ndarray,
    fish_conf_min: float,
    ruler_conf_min: float,
) -> tuple[BBox | None, BBox | None]:
    h, w = frame.shape[:2]
    results = model.predict(frame, verbose=False, conf=min(fish_conf_min, ruler_conf_min))
    if not results:
        return None, None

    boxes = results[0].boxes
    if boxes is None:
        return None, None

    names = results[0].names
    fish_candidates: list[tuple[float, BBox]] = []
    ruler_candidates: list[tuple[float, BBox]] = []

    for box in boxes:
        cls_idx = int(box.cls.item())
        name = str(names.get(cls_idx, cls_idx))
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bb = clamp_bbox(int(x1), int(y1), int(x2), int(y2), w, h)

        if name == "fish" and conf >= fish_conf_min:
            fish_candidates.append((conf, bb))
        elif name == "ruler" and conf >= ruler_conf_min:
            ruler_candidates.append((conf, bb))

    if not fish_candidates or not ruler_candidates:
        return None, None

    fish = max(fish_candidates, key=lambda x: x[0])[1]
    ruler = max(ruler_candidates, key=lambda x: x[1].w)[1]
    return fish, ruler


def detect_ruler_heuristic(frame: np.ndarray) -> BBox | None:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: tuple[float, BBox] | None = None
    image_area = float(w * h)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.03 * image_area or area > 0.8 * image_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect < 1.4 or aspect > 10.0:
            continue

        bb = clamp_bbox(x, y, x + bw, y + bh, w, h)
        score = (area / image_area) * 0.6 + (bb.w / w) * 0.4
        if best is None or score > best[0]:
            best = (score, bb)

    return best[1] if best else None


def detect_fish_heuristic(frame: np.ndarray, ruler: BBox) -> BBox | None:
    h, w = frame.shape[:2]
    pad = int(0.08 * ruler.w)
    roi = clamp_bbox(ruler.x1 - pad, ruler.y1, ruler.x2 + pad, ruler.y2, w, h)
    crop = frame[roi.y1 : roi.y2, roi.x1 : roi.x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    sat = cv2.inRange(hsv[:, :, 1], 35, 255)
    dark = cv2.inRange(gray, 0, 170)
    edges = cv2.Canny(gray, 50, 120)
    mask = cv2.bitwise_or(sat, dark)
    mask = cv2.bitwise_or(mask, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    roi_area = float(crop.shape[0] * crop.shape[1])
    best: tuple[float, BBox] | None = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.006 * roi_area or area > 0.60 * roi_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect < 1.2:
            continue
        bb = clamp_bbox(x + roi.x1, y + roi.y1, x + bw + roi.x1, y + bh + roi.y1, w, h)
        score = (area / roi_area) * 0.7 + min(1.0, aspect / 4.0) * 0.3
        if best is None or score > best[0]:
            best = (score, bb)

    return best[1] if best else None


def validate_pair(frame: np.ndarray, fish: BBox, ruler: BBox) -> bool:
    h, w = frame.shape[:2]
    if fish.area < int(0.0015 * w * h):
        return False
    if ruler.area < int(0.015 * w * h):
        return False

    overlap_x1 = max(fish.x1, ruler.x1)
    overlap_y1 = max(fish.y1, ruler.y1)
    overlap_x2 = min(fish.x2, ruler.x2)
    overlap_y2 = min(fish.y2, ruler.y2)
    overlap = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
    overlap_ratio = overlap / (fish.area + 1e-6)
    return overlap_ratio >= 0.35


def build_dataset(
    videos_dir: Path,
    dataset_dir: Path,
    frame_interval_ms: int,
    val_ratio: float,
    seed: int,
    clean: bool,
    pseudo_model_path: Path,
    fish_conf_min: float,
    ruler_conf_min: float,
) -> None:
    ensure_dirs(dataset_dir, clean=clean)
    write_data_yaml(dataset_dir)

    random.seed(seed)
    model = load_pseudo_model(pseudo_model_path)

    candidates: list[tuple[str, np.ndarray, BBox, BBox, str]] = []
    per_video_valid: dict[str, int] = {}

    videos = sorted(videos_dir.glob("*.mp4"))
    for video in videos:
        valid_for_video = 0
        samples = collect_frames_interval(video, interval_ms=frame_interval_ms)

        for ts, frame in samples:
            fish: BBox | None = None
            ruler: BBox | None = None

            if model is not None:
                fish, ruler = select_pair_from_model(
                    model=model,
                    frame=frame,
                    fish_conf_min=fish_conf_min,
                    ruler_conf_min=ruler_conf_min,
                )

            if fish is None or ruler is None:
                ruler = detect_ruler_heuristic(frame)
                if ruler is not None:
                    fish = detect_fish_heuristic(frame, ruler)

            if fish is None or ruler is None:
                continue
            if not validate_pair(frame, fish, ruler):
                continue

            stem = f"{video.stem}_t{ts:06d}"
            candidates.append((stem, frame, fish, ruler, video.stem))
            valid_for_video += 1

        per_video_valid[video.stem] = valid_for_video

    if not candidates:
        raise RuntimeError("No valid labeled samples were generated from the videos.")

    random.shuffle(candidates)
    val_size = max(1, int(len(candidates) * val_ratio))

    val_items = candidates[:val_size]
    train_items = candidates[val_size:]

    def write_split(items: list[tuple[str, np.ndarray, BBox, BBox, str]], split: str) -> None:
        for stem, frame, fish, ruler, _video_id in items:
            h, w = frame.shape[:2]
            img_path = dataset_dir / "images" / split / f"{stem}.jpg"
            lbl_path = dataset_dir / "labels" / split / f"{stem}.txt"

            cv2.imwrite(str(img_path), frame)
            lbl_path.write_text(
                "\n".join([to_yolo_line(0, fish, w, h), to_yolo_line(1, ruler, w, h)]) + "\n",
                encoding="utf-8",
            )

    write_split(train_items, "train")
    write_split(val_items, "val")

    used_videos = [video for video, count in per_video_valid.items() if count > 0]
    print(f"Videos found: {len(videos)} | Videos used: {len(used_videos)}")
    print("Per-video valid samples:")
    for name in sorted(per_video_valid):
        print(f"  - {name}: {per_video_valid[name]}")

    print(f"Total labeled images: {len(candidates)}")
    print(f"Train: {len(train_items)} | Val: {len(val_items)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset for fish+ruler YOLO training")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("/home/luan/IdeaProjects/FishingAPP/obsidian/fishing/video-modelo"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/luan/IdeaProjects/FishingAPP/dataset"),
    )
    parser.add_argument(
        "--pseudo-model",
        type=Path,
        default=Path("/home/luan/IdeaProjects/FishingAPP/fishing-ai-engine/ai/models/best.pt"),
    )
    parser.add_argument("--frame-interval-ms", type=int, default=250)
    parser.add_argument("--fish-conf-min", type=float, default=0.05)
    parser.add_argument("--ruler-conf-min", type=float, default=0.08)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Remove existing dataset directory before generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        videos_dir=args.videos_dir,
        dataset_dir=args.dataset_dir,
        frame_interval_ms=args.frame_interval_ms,
        val_ratio=args.val_ratio,
        seed=args.seed,
        clean=args.clean,
        pseudo_model_path=args.pseudo_model,
        fish_conf_min=args.fish_conf_min,
        ruler_conf_min=args.ruler_conf_min,
    )


if __name__ == "__main__":
    main()
