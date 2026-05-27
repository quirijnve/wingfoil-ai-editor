from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass
class Event:
    timestamp: float
    confidence: float
    clip_file: str


def write_events_json(events: List[Event], output_path: Path):
    payload = {
        "events": [asdict(e) for e in events],
        "count": len(events),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
