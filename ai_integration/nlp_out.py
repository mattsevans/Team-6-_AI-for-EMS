# nlp_out.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Optional

class NlpOutbox:
    """
    Append-only JSONL publisher for Zora outputs.

    Single fixed output file (you are the ONLY writer), e.g.:
        NlpOutbox(Path("MasterFileOutput.jsonl"))

    Each line is a compact JSON object:
        {"incident_id":"...", "text":"...", ["time":"..."], ["role":"..."]}

    Notes:
    - `incident_id` is included in each line for consumers.
    - `event_id` is accepted but intentionally ignored.
    """

    def __init__(self, file_path: Path):
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        *,
        incident_id: str,
        text: str,
        # Ignored (accepted for API compatibility with the edge-loop callback)
        event_id: Optional[str] = None,
        timestamp_iso: Optional[str] = None,
        role: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> Path:
        """Append ONE JSON line to the fixed outbox file."""
        record = {
            "incident_id": incident_id,
            "text": text,
        }
        if timestamp_iso:
            record["time"] = timestamp_iso
        if role:
            record["role"] = role

        line = json.dumps(record, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())  # optional durability
            except OSError:
                pass
        return self.path
