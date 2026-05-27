# Wingfoil Insta360 Crash Detector (MVP)

Simple Python MVP that analyzes an input MP4, detects likely crash moments using lightweight OpenCV metrics, and exports short clips around each detected event with `ffmpeg`.

## Features

- Samples video at **2 FPS** for fast analysis.
- Computes simple instability signals between sampled frames:
  - frame difference magnitude (motion)
  - blur/chaos estimate (Laplacian variance)
  - brightness and splash-like flashes
- Detects event spikes and merges nearby events.
- Exports clips from **5 seconds before** to **6 seconds after** each event.
- Writes `events.json` containing timestamps, confidence scores, and clip filenames.

## Project Structure

```
wingfoil-ai-editor/
  README.md
  requirements.txt
  main.py
  wingfoil_editor/
    __init__.py
    video_io.py
    crash_detector.py
    clip_exporter.py
    events.py
  samples/
  out/
```

## Requirements

- Python 3.10+
- `ffmpeg` available on your PATH

Install Python deps:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py analyze --input input.mp4 --output out/
```

Example:

```bash
python main.py analyze --input samples/session.mp4 --output out/
```

Output in `out/`:

- `event_001_123.50s.mp4` (and more clips)
- `events.json`

## Notes

- This is a heuristic baseline (no deep learning).
- Depending on camera mounting and water conditions, tune thresholds in `wingfoil_editor/crash_detector.py`:
  - `threshold_z` (detection sensitivity)
  - `merge_window_sec` (event consolidation)
