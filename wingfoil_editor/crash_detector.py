from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from wingfoil_editor.video_io import SampledFrame


@dataclass
class FrameMetrics:
    timestamp: float
    motion: float
    chaos: float
    brightness_jump: float
    splash_ratio: float
    score: float


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - mean) / std


def compute_metrics(sampled_frames: List[SampledFrame]) -> List[FrameMetrics]:
    """Compute simple instability metrics between consecutive sampled frames."""
    if len(sampled_frames) < 2:
        return []

    motion_vals = []
    chaos_vals = []
    bright_jump_vals = []
    splash_vals = []
    timestamps = []

    prev_gray = None
    prev_brightness = None

    for sf in sampled_frames:
        gray = cv2.cvtColor(sf.frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)

        # Blur/chaos proxy: high-frequency content from Laplacian variance.
        # During chaotic crashes this often spikes due to spray, shake, and texture bursts.
        lap_var = float(cv2.Laplacian(gray_small, cv2.CV_64F).var())

        brightness = float(np.mean(gray_small))

        if prev_gray is None:
            prev_gray = gray_small
            prev_brightness = brightness
            continue

        # Motion magnitude proxy using mean absolute difference.
        diff = cv2.absdiff(gray_small, prev_gray)
        motion = float(np.mean(diff))

        # Brightness jump catches abrupt white-water splash and rapid exposure changes.
        brightness_jump = abs(brightness - float(prev_brightness))

        # Splash ratio: proportion of very bright pixels (possible foam/water spray flashes).
        splash_ratio = float(np.mean(gray_small > 220))

        motion_vals.append(motion)
        chaos_vals.append(lap_var)
        bright_jump_vals.append(brightness_jump)
        splash_vals.append(splash_ratio)
        timestamps.append(sf.timestamp)

        prev_gray = gray_small
        prev_brightness = brightness

    motion_z = _safe_zscore(np.array(motion_vals, dtype=np.float32))
    chaos_z = _safe_zscore(np.array(chaos_vals, dtype=np.float32))
    bright_z = _safe_zscore(np.array(bright_jump_vals, dtype=np.float32))
    splash_z = _safe_zscore(np.array(splash_vals, dtype=np.float32))

    metrics = []
    for i, t in enumerate(timestamps):
        # Weighted simple score; motion dominates, then visual chaos, then brightness/splash hints.
        score = (
            0.45 * motion_z[i]
            + 0.30 * chaos_z[i]
            + 0.15 * bright_z[i]
            + 0.10 * splash_z[i]
        )
        metrics.append(
            FrameMetrics(
                timestamp=t,
                motion=motion_vals[i],
                chaos=chaos_vals[i],
                brightness_jump=bright_jump_vals[i],
                splash_ratio=splash_vals[i],
                score=float(score),
            )
        )

    return metrics


def detect_event_timestamps(metrics: List[FrameMetrics], threshold_z: float = 1.8) -> List[FrameMetrics]:
    """Pick candidate events from score spikes above threshold."""
    return [m for m in metrics if m.score >= threshold_z]


def merge_nearby_events(candidates: List[FrameMetrics], merge_window_sec: float = 4.0) -> List[FrameMetrics]:
    """Merge nearby candidates into one event (keep highest score in cluster)."""
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda m: m.timestamp)
    merged = []
    cluster = [candidates[0]]

    for cur in candidates[1:]:
        if cur.timestamp - cluster[-1].timestamp <= merge_window_sec:
            cluster.append(cur)
        else:
            merged.append(max(cluster, key=lambda m: m.score))
            cluster = [cur]

    merged.append(max(cluster, key=lambda m: m.score))
    return merged
