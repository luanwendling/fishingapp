"""Measurement logic: pixel-to-cm and fish length estimation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import bbox_width


@dataclass(frozen=True)
class FrameMeasurement:
    """Per-frame measurement details."""

    length_cm: float
    fish_confidence: float
    ruler_confidence: float
    fish_length_px: float
    ruler_width_px: float


class MeasurementEngine:
    """Compute fish length using ruler width as scale."""

    def __init__(self, ruler_length_cm: float, top_k_frames: int) -> None:
        self.ruler_length_cm = ruler_length_cm
        self.top_k_frames = top_k_frames

    @staticmethod
    def _largest_ruler_by_horizontal_width(detections: list[dict]) -> dict | None:
        rulers = [d for d in detections if d.get("class") == "ruler"]
        if not rulers:
            return None
        return max(rulers, key=lambda d: bbox_width(tuple(d["bbox"])))

    @staticmethod
    def _best_fish_by_confidence(detections: list[dict]) -> dict | None:
        fish = [d for d in detections if d.get("class") == "fish"]
        if not fish:
            return None
        return max(fish, key=lambda d: float(d.get("confidence", 0.0)))

    @staticmethod
    def _expand_bbox_horizontally(
        bbox: tuple[float, float, float, float],
        img_w: int,
        margin_ratio: float = 0.05,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        width = max(0.0, x2 - x1)
        margin = width * margin_ratio
        ex1 = int(max(0.0, x1 - margin))
        ex2 = int(min(float(img_w), x2 + margin))
        return ex1, int(y1), ex2, int(y2)

    @staticmethod
    def _is_ruler_vertical(ruler_bbox: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = [float(v) for v in ruler_bbox]
        return (y2 - y1) > (x2 - x1)

    @staticmethod
    def _extract_primary_fish_contour(
        frame: np.ndarray,
        expanded_bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, tuple[int, int]] | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = expanded_bbox

        x1 = max(0, min(x1, w - 1))
        x2 = max(1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(1, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        sat_mask = cv2.inRange(hsv[:, :, 1], 30, 255)
        dark_mask = cv2.inRange(gray, 0, 170)
        edges = cv2.Canny(gray, 40, 120)

        mask = cv2.bitwise_or(sat_mask, dark_mask)
        mask = cv2.bitwise_or(mask, edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        roi_area = float(crop.shape[0] * crop.shape[1])
        best: tuple[float, np.ndarray] | None = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.01 * roi_area:
                continue
            cx, cy, cw, ch = cv2.boundingRect(contour)
            aspect = max(cw, ch) / (min(cw, ch) + 1e-6)
            if aspect < 1.2:
                continue
            score = (area / roi_area) * 0.7 + min(1.0, aspect / 4.0) * 0.3
            if best is None or score > best[0]:
                best = (score, contour)

        if best is None:
            return None
        return best[1], (x1, y1)

    @staticmethod
    def _compute_real_fish_span(
        frame: np.ndarray,
        expanded_bbox: tuple[int, int, int, int],
        vertical_axis: bool,
    ) -> tuple[float, float] | None:
        extracted = MeasurementEngine._extract_primary_fish_contour(frame, expanded_bbox)
        if extracted is None:
            return None

        contour, (x1, y1) = extracted
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if vertical_axis:
            start = float(y1 + cy)
            end = float(y1 + cy + ch)
        else:
            start = float(x1 + cx)
            end = float(x1 + cx + cw)
        if end <= start:
            return None
        return start, end

    @staticmethod
    def _compute_real_fish_major_axis(
        frame: np.ndarray,
        expanded_bbox: tuple[int, int, int, int],
    ) -> float | None:
        extracted = MeasurementEngine._extract_primary_fish_contour(frame, expanded_bbox)
        if extracted is None:
            return None
        contour, _origin = extracted
        (_, _), (rw, rh), _angle = cv2.minAreaRect(contour)
        major = float(max(rw, rh))
        if major <= 0:
            return None
        return major

    @staticmethod
    def _estimate_tail_extent_from_mask(
        frame: np.ndarray,
        ruler_bbox: tuple[float, float, float, float],
        fish_bbox: tuple[float, float, float, float],
        ruler_vertical: bool,
    ) -> float | None:
        """Estimate fish extent from zero wall to furthest fish pixel on ruler axis."""
        h, w = frame.shape[:2]
        rx1, ry1, rx2, ry2 = [int(v) for v in ruler_bbox]
        rx1 = max(0, min(rx1, w - 1))
        ry1 = max(0, min(ry1, h - 1))
        rx2 = max(1, min(rx2, w))
        ry2 = max(1, min(ry2, h))
        if rx2 <= rx1 or ry2 <= ry1:
            return None

        fx1, fy1, fx2, fy2 = [float(v) for v in fish_bbox]
        # ROI around fish + ruler to capture tail fin tips that detector may miss.
        pad_x = int(0.15 * (fx2 - fx1))
        pad_y = int(0.15 * (fy2 - fy1))
        x1 = max(rx1, int(max(0, fx1 - pad_x)))
        y1 = max(ry1, int(max(0, fy1 - pad_y)))
        x2 = min(rx2, int(min(w, fx2 + pad_x)))
        y2 = min(ry2, int(min(h, fy2 + pad_y)))
        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Broad fish mask: yellow/green body + dark fins + texture.
        yellow1 = cv2.inRange(hsv, (15, 35, 35), (45, 255, 255))
        greenish = cv2.inRange(hsv, (35, 20, 20), (95, 255, 255))
        dark = cv2.inRange(gray, 0, 150)
        edges = cv2.Canny(gray, 40, 120)
        mask = cv2.bitwise_or(yellow1, greenish)
        mask = cv2.bitwise_or(mask, dark)
        mask = cv2.bitwise_or(mask, edges)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        roi_area = float((y2 - y1) * (x2 - x1))
        best: tuple[float, np.ndarray] | None = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.01 * roi_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            score = (area / roi_area) * 0.8 + min(1.0, aspect / 4.0) * 0.2
            if best is None or score > best[0]:
                best = (score, contour)

        if best is None:
            return None

        contour = best[1]
        pts = contour.reshape(-1, 2)
        if pts.size == 0:
            return None

        if ruler_vertical:
            # zero at top wall
            furthest_local = float(np.max(pts[:, 1]))
            furthest_global = float(y1 + furthest_local)
            zero_ref = min(float(ry1), float(fy1))
        else:
            furthest_local = float(np.max(pts[:, 0]))
            furthest_global = float(x1 + furthest_local)
            zero_ref = min(float(rx1), float(fx1))

        extent = max(0.0, furthest_global - zero_ref)
        return extent if extent > 0 else None

    def _calculate_pixels_per_cm_from_ticks(
        self,
        frame: np.ndarray,
        ruler_bbox: tuple[float, float, float, float],
    ) -> float | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in ruler_bbox]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(1, min(x2, w))
        y2 = max(1, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        ruler = frame[y1:y2, x1:x2]
        if ruler.size == 0:
            return None

        # Converter para escala de cinza
        gray = cv2.cvtColor(ruler, cv2.COLOR_BGR2GRAY)

        # Suavização leve
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Realçar marcas escuras
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Adapt signal axis to ruler orientation.
        ruler_vertical = ruler.shape[0] > ruler.shape[1]
        if ruler_vertical:
            # For vertical ruler, ticks repeat along Y.
            projection = np.mean(normalized, axis=1)
        else:
            # For horizontal ruler, ticks repeat along X.
            projection = np.mean(normalized, axis=0)

        # Inverter sinal (marcas são mais escuras)
        projection = np.max(projection) - projection

        # Remover tendência (centralizar)
        projection = projection - np.mean(projection)

        # Autocorrelação
        autocorr = np.correlate(projection, projection, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        # Ignorar primeiros pixels (lag 0)
        min_lag = 5
        autocorr[:min_lag] = 0

        signal_len = projection.shape[0]
        # Encontrar pico dominante em faixa plausivel de marcas de 1 cm.
        min_cm_lag = max(5, int(signal_len * 0.03))
        max_cm_lag = max(min_cm_lag + 1, int(signal_len * 0.20))
        max_cm_lag = min(max_cm_lag, len(autocorr) - 1)

        if max_cm_lag <= min_cm_lag:
            return None

        window = autocorr[min_cm_lag : max_cm_lag + 1]
        peak_index = int(np.argmax(window)) + min_cm_lag

        if peak_index <= 0:
            return None

        pixels_per_cm = float(peak_index)

        # Sanity check
        if pixels_per_cm < 5 or pixels_per_cm > (signal_len / 2):
            return None

        return pixels_per_cm

    @staticmethod
    def _ocr_red_number(roi_bgr: np.ndarray) -> int | None:
        """Try to OCR a ruler number from a small red blob ROI."""
        if roi_bgr.size == 0:
            return None

        import shutil

        if shutil.which("tesseract") is None:
            return None

        try:
            import pytesseract
        except Exception:
            return None

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            text = pytesseract.image_to_string(
                bw,
                config="--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789",
            )
        except Exception:
            return None
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return None

        try:
            value = int(digits)
        except ValueError:
            return None

        # Heuristic normalization for ruler major marks (20, 30, 40, ...).
        if 2 <= value <= 9:
            value *= 10
        if value < 0 or value > 120:
            return None
        if value % 10 != 0:
            value = int(round(value / 10.0) * 10)
        return value

    def _estimate_length_from_visible_number(
        self,
        frame: np.ndarray,
        ruler_bbox: tuple[float, float, float, float],
        fish_bbox: tuple[float, float, float, float],
    ) -> float | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in ruler_bbox]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(1, min(x2, w))
        y2 = max(1, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None

        ruler = frame[y1:y2, x1:x2]
        if ruler.size == 0:
            return None

        rh, rw = ruler.shape[:2]
        hsv = cv2.cvtColor(ruler, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 70, 50], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([170, 70, 50], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        number_candidates: list[tuple[float, int]] = []
        min_area = max(200, int(0.0025 * rw * rh))
        min_h = max(10, int(0.18 * rh))

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bh < min_h or bw < 6:
                continue
            if bw > int(0.35 * rw) or bh > int(0.8 * rh):
                continue

            # Expand small OCR crop around red blob.
            pad_x = max(3, int(0.25 * bw))
            pad_y = max(3, int(0.20 * bh))
            cx1 = max(0, bx - pad_x)
            cy1 = max(0, by - pad_y)
            cx2 = min(rw, bx + bw + pad_x)
            cy2 = min(rh, by + bh + pad_y)

            roi = ruler[cy1:cy2, cx1:cx2]
            value = self._ocr_red_number(roi)
            if value is None:
                continue

            center_x = bx + (bw / 2.0)
            number_candidates.append((center_x, value))

        if len(number_candidates) < 2:
            return None

        number_candidates.sort(key=lambda item: item[0])

        # Estimate local px/cm from consecutive known numbers (10 cm apart).
        px_per_cm_candidates: list[float] = []
        for i in range(len(number_candidates) - 1):
            x_a, v_a = number_candidates[i]
            x_b, v_b = number_candidates[i + 1]
            dv = v_b - v_a
            dx = x_b - x_a
            if dx <= 0:
                continue
            if abs(dv) == 10:
                px_per_cm_candidates.append(dx / 10.0)

        if len(px_per_cm_candidates) < 2:
            return None
        px_arr = np.array(px_per_cm_candidates, dtype=float)
        if float(np.mean(px_arr)) <= 0:
            return None
        if float(np.std(px_arr) / np.mean(px_arr)) > 0.25:
            return None
        pixels_per_cm = float(np.median(px_arr))
        if pixels_per_cm <= 0:
            return None

        # Cross-check local numeric calibration against tick-based calibration.
        tick_ppcm = self._calculate_pixels_per_cm_from_ticks(frame, ruler_bbox)
        if tick_ppcm is not None and tick_ppcm > 0:
            ratio = pixels_per_cm / tick_ppcm
            if ratio < 0.7 or ratio > 1.3:
                return None

        fish_tail_x = float(fish_bbox[2])
        tail_relative_x = fish_tail_x - float(x1)

        # Closest number at or left of tail.
        left_candidates = [item for item in number_candidates if item[0] <= tail_relative_x]
        if left_candidates:
            anchor_x, anchor_value = min(left_candidates, key=lambda item: abs(tail_relative_x - item[0]))
        else:
            anchor_x, anchor_value = min(number_candidates, key=lambda item: abs(tail_relative_x - item[0]))

        displacement_px = tail_relative_x - anchor_x
        displacement_cm = displacement_px / pixels_per_cm
        length_cm = float(anchor_value + displacement_cm)

        if length_cm <= 0 or length_cm > 150:
            return None
        return length_cm

    def measure_from_detections(
        self,
        frames: list[np.ndarray],
        frame_detections: list[list[dict]],
    ) -> list[FrameMeasurement]:
        measures: list[FrameMeasurement] = []

        for idx, (frame, detections) in enumerate(zip(frames, frame_detections)):
            fish_det = self._best_fish_by_confidence(detections)
            ruler_det = self._largest_ruler_by_horizontal_width(detections)
            if fish_det is None or ruler_det is None:
                continue

            h, w = frame.shape[:2]
            fish_bbox = tuple(float(v) for v in fish_det["bbox"])
            ruler_bbox = tuple(float(v) for v in ruler_det["bbox"])
            ruler_vertical = self._is_ruler_vertical(ruler_bbox)
            # Skip detections with tail touching frame border (likely truncated fish).
            border_margin_x = max(2, int(0.01 * w))
            border_margin_y = max(2, int(0.01 * h))
            if ruler_vertical:
                if fish_bbox[3] >= (h - border_margin_y):
                    continue
            else:
                if fish_bbox[2] >= (w - border_margin_x):
                    continue

            expanded = self._expand_bbox_horizontally(fish_bbox, img_w=w, margin_ratio=0.05)
            if ruler_vertical:
                # Add 5% safety margin along vertical axis when ruler is vertical.
                fy1 = int(max(0.0, fish_bbox[1] - 0.05 * (fish_bbox[3] - fish_bbox[1])))
                fy2 = int(min(float(h), fish_bbox[3] + 0.05 * (fish_bbox[3] - fish_bbox[1])))
                expanded = (expanded[0], fy1, expanded[2], fy2)

            real_span = self._compute_real_fish_span(frame, expanded, vertical_axis=ruler_vertical)
            if ruler_vertical:
                expanded_span = float(max(0, expanded[3] - expanded[1]))
            else:
                expanded_span = float(max(0, expanded[2] - expanded[0]))
            if expanded_span <= 0:
                continue

            if real_span is None:
                # Fallback to expanded bbox when contour segmentation is unstable.
                fish_length_px = expanded_span
            else:
                contour_span = float(max(0.0, real_span[1] - real_span[0]))
                fish_length_px = contour_span if contour_span >= (0.6 * expanded_span) else expanded_span

            # Video-level calibration aid: include major-axis estimate for tilted fish.
            major_axis_px = self._compute_real_fish_major_axis(frame, expanded)
            if major_axis_px is not None:
                fish_length_px = max(fish_length_px, 0.9 * major_axis_px)
            ruler_width_px = bbox_width(tuple(ruler_det["bbox"]))
            if ruler_vertical:
                ruler_major_px = max(0.0, float(ruler_bbox[3] - ruler_bbox[1]))
            else:
                ruler_major_px = max(0.0, float(ruler_bbox[2] - ruler_bbox[0]))
            if fish_length_px <= 0.0 or ruler_width_px <= 0.0:
                continue

            # Zero-anchored tail displacement on ruler axis (mouth at zero wall).
            if ruler_vertical:
                zero_ref = min(float(ruler_bbox[1]), float(fish_bbox[1]))
                fish_anchor_px = max(0.0, float(fish_bbox[3] - zero_ref))
            else:
                zero_ref = min(float(ruler_bbox[0]), float(fish_bbox[0]))
                fish_anchor_px = max(0.0, float(fish_bbox[2] - zero_ref))
            fish_length_px = max(fish_length_px, fish_anchor_px)
            tail_extent_px = self._estimate_tail_extent_from_mask(
                frame=frame,
                ruler_bbox=ruler_bbox,
                fish_bbox=fish_bbox,
                ruler_vertical=ruler_vertical,
            )
            if tail_extent_px is not None:
                fish_length_px = max(fish_length_px, tail_extent_px)

            length_cm = self._estimate_length_from_visible_number(
                frame=frame,
                ruler_bbox=ruler_bbox,
                fish_bbox=fish_bbox,
            )
            if length_cm is None:
                pixels_per_cm = self._calculate_pixels_per_cm_from_ticks(frame, ruler_bbox)
                if pixels_per_cm is None:
                    # Fallback to fixed-ruler conversion if tick extraction is weak.
                    pixels_per_cm = ruler_major_px / self.ruler_length_cm
                if pixels_per_cm <= 0:
                    continue
                length_cm = fish_length_px / pixels_per_cm

            measures.append(
                FrameMeasurement(
                    length_cm=float(length_cm),
                    fish_confidence=float(fish_det.get("confidence", 0.0)),
                    ruler_confidence=float(ruler_det.get("confidence", 0.0)),
                    fish_length_px=float(fish_length_px),
                    ruler_width_px=float(ruler_width_px),
                )
            )

        return measures

    def average_best_frames(self, measurements: list[FrameMeasurement]) -> tuple[float, list[FrameMeasurement]]:
        if not measurements:
            return 0.0, []

        lengths = np.array([m.length_cm for m in measurements], dtype=float)
        median_length = float(np.median(lengths))

        ranked = sorted(
            measurements,
            key=lambda m: (
                abs(m.length_cm - median_length),
                -((m.fish_confidence + m.ruler_confidence) / 2.0),
            ),
        )

        selected = ranked[: self.top_k_frames]
        avg_length = float(np.mean([m.length_cm for m in selected]))
        return avg_length, selected
