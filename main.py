from __future__ import annotations

import argparse
from pathlib import Path

from wingfoil_editor.clip_exporter import export_clip
from wingfoil_editor.crash_detector import (
    compute_metrics,
    detect_event_timestamps,
    merge_nearby_events,
)
from wingfoil_editor.events import Event, write_events_json
from wingfoil_editor.video_io import iter_sampled_frames


def analyze_video(input_path: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled = list(iter_sampled_frames(input_path, sample_fps=2.0))
    metrics = compute_metrics(sampled)

    candidates = detect_event_timestamps(metrics, threshold_z=1.8)
    merged = merge_nearby_events(candidates, merge_window_sec=4.0)

    events = []
    for idx, event in enumerate(merged, start=1):
        clip_name = f"event_{idx:03d}_{event.timestamp:.2f}s.mp4"
        clip_path = out_dir / clip_name

        export_clip(
            input_video=input_path,
            output_path=clip_path,
            event_time=event.timestamp,
            pre_seconds=5.0,
            post_seconds=6.0,
        )

        # Convert z-score-like signal into a bounded 0-1 confidence for readability.
        confidence = max(0.0, min(1.0, (event.score - 1.0) / 3.0))
        events.append(
            Event(
                timestamp=round(event.timestamp, 3),
                confidence=round(confidence, 3),
                clip_file=clip_name,
            )
        )

    write_events_json(events, out_dir / "events.json")

    print(f"Analyzed: {input_path}")
    print(f"Sampled frames: {len(sampled)}")
    print(f"Detected events: {len(events)}")
    print(f"Output folder: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wingfoil Insta360 crash event detector MVP")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Analyze a video and export event clips")
    p_analyze.add_argument("--input", required=True, help="Path to input MP4 file")
    p_analyze.add_argument("--output", required=True, help="Path to output folder")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        analyze_video(args.input, args.output)


if __name__ == "__main__":
    main()
