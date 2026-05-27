from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SampledFrame:
    """A sampled frame and its timestamp (seconds) in the source video."""

    timestamp: float
    frame: np.ndarray


def iter_sampled_frames(video_path: str, sample_fps: float = 2.0):
    """
    Yield frames sampled at approximately `sample_fps` from `video_path`.

    We seek by source frame index to keep sampling stable even on variable frame rate
    videos where simple sequential reads can drift.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if total_frames > 0 else 0.0

    step_seconds = 1.0 / sample_fps
    t = 0.0

    while t <= duration:
        frame_index = int(round(t * src_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        yield SampledFrame(timestamp=t, frame=frame)
        t += step_seconds

    cap.release()
