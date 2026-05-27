from __future__ import annotations

import subprocess
from pathlib import Path


def export_clip(
    input_video: str,
    output_path: Path,
    event_time: float,
    pre_seconds: float = 5.0,
    post_seconds: float = 6.0,
):
    """Export a clip around event_time using ffmpeg (stream copy for speed)."""
    start = max(0.0, event_time - pre_seconds)
    duration = pre_seconds + post_seconds

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        input_video,
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        str(output_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg clip export failed for {output_path.name}:\n{proc.stderr}"
        )
