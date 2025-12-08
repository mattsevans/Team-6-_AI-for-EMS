# edge_loop.py
from __future__ import annotations
import os
import time
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Any, Literal
import re
from datetime import datetime, timezone
# =========================
# Config & dependency types
# =========================

@dataclass(frozen=True)
class LoopConfig:
    """
    Paths and simple settings for the edge loop.
    NOTE: For stream mode we only need `transcript_path` and `outbox_dir`.
    """
    # Stream (JSONL-style) input: a single, fixed file that is appended to
    transcript_path: Path

    # Outbox for TTS/UI messages (unchanged)
    outbox_dir: Path

    # (Kept for compatibility; not used in stream mode)
    inbox_dir: Path = Path(".unused_inbox")
    processed_dir: Path = Path(".unused_processed")
    processed_err_dir: Path = Path(".unused_processed_err")

    # --- API pacing ---
    max_rps: float = 2.0  # Global max requests/sec across extractor and QA

    # --- Loop lifetime controls ---
    idle_timeout_sec: float = 15.0   # Failsafe: if no new completed line for this long → close incident
    poll_interval_sec: float = 0.5    # When no new line, how often to re-check


@dataclass
class CloudDeps:
    """
    Callbacks that push data to Firestore (Firebase).
    All functions should be FAST and safe to call many times (idempotent).
    """
    init_incident: Callable[[str, str], None]
    append_conversation: Callable[
        [str, str, Literal["EMS", "other", "zora"], str], None
    ]  # accepts **kwargs in implementations
    upsert_profile_snapshots: Callable[[str, str], None]  # accepts **kwargs in implementations
    mark_complete: Callable[[str, str, Literal["explicit", "idle"]], None]

def _cloud_noop(*args, **kwargs) -> None:
    return

NOOP_CLOUD = CloudDeps(
    init_incident=lambda incident_id, started_time_iso: None,
    append_conversation=lambda incident_id, event_id, role, timestamp_iso, **kwargs: None,
    upsert_profile_snapshots=lambda incident_id, timestamp_iso, **kwargs: None,
    mark_complete=lambda incident_id, close_time_iso, close_reason: None,
)


@dataclass(frozen=True)
class Deps:
    """
    All external functions the loop needs. Injecting them keeps this file small and testable.
    """
    normalize_incoming_message: Callable[[str], object]
    ask_zora: Callable[[dict, dict, str], str]
    extract_patient_info: Callable[[str, str, str], dict]
    smart_update_profiles: Callable[[dict, dict, str, dict, str], None]
    rollback_last_entry: Callable[[dict], str]
    on_incident_close: Callable[[dict, dict], None]
    cloud: CloudDeps = field(default_factory=lambda: NOOP_CLOUD)
    publish_zora: Optional[Callable[[str, str, str, str, str], None]] = None

# =========================
# Small internal helpers
# =========================

def _rate_limit(last_call_ts: Optional[float], max_rps: float) -> float:
    """Simple request pacing: ensures we don't exceed max requests per second."""
    if max_rps <= 0:
        return time.time()
    min_interval = 1.0 / max_rps
    now = time.time()
    if last_call_ts is not None:
        elapsed = now - last_call_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
    return time.time()

def _normalize_command(s: str) -> str:
    """Normalize a Zora command string for simple matching."""
    if s is None:
        return ""
    cmd = s.lower().strip()
    if cmd.endswith("."):
        cmd = cmd[:-1].strip()
    return cmd

# compile once for speed; matches one or more leading fillers (case-insensitive)
_LEADING_FILLERS_RE = re.compile(
    r'^(?:\s*(?:okay|ok|uh|um|alright|copy|roger|yeah|yep|yup|hey|so|well)\b[,\s]*)+',
    flags=re.IGNORECASE
)

def _strip_leading_fillers(s: str) -> str:
    """
    Remove leading chit-chat/filler tokens only if they occur at the start.
    Examples:
      'Okay, Zora, start timer' -> 'Zora, start timer'
      'um   patient name is Dan' -> 'patient name is Dan'
      'ok' -> '' (becomes empty => skip)
    """
    if not s:
        return s
    return _LEADING_FILLERS_RE.sub('', s).lstrip()


# =========================
# Stream tailer (line-based)
# =========================

class StreamTail:
    """
    Minimal line tailer for a single, ever-growing file.

    Rules implemented per your design:
    - At start, we ignore any pre-existing lines by initializing the pointer
      to the current total line count (we start a NEW incident at the next line).
    - We only process a line once a FOLLOWING newline/blank line arrives
      (prevents partial-line processing).
    - We track content-line index (for event_id) and detect file truncation.

    Assumptions:
    - Each input "record" is a single line like: "[HH:MM:SS] ROLE: text"
    - A blank line after a record is acceptable; it also flushes the previous line.
    """
    def flush_pending(self) -> Optional[tuple[int, str]]:
        """Force-emit the buffered line (if any)."""
        if self._buffer is None:
            return None
        idx = self._next_index
        self._next_index += 1
        line = self._buffer
        self._buffer = None
        return idx, line

    def __init__(self, path: Path):
        self.path = path
        self._fh = None
        self._last_size = 0
        self._buffer: Optional[str] = None  # previous line waiting to be confirmed
        self._next_index = 0                # next content-line index to assign
        self._started = False
        self._processed_at_least_one = False

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # utf-8-sig will swallow a BOM if one ever shows up at file start
        self._fh = open(self.path, "r", encoding="utf-8-sig")
        self._fh.seek(0, os.SEEK_END)  # always position at EOF on open

    def _ensure_open(self) -> bool:
        if self._fh is not None:
            return True
        if not self.path.exists():
            return False  # caller will sleep and retry
        self._open()
        return True

    def _reset_on_truncate(self):
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None
        self._buffer = None
        self._next_index = 0
        self._started = False
        self._processed_at_least_one = False
        self._open()
        self._last_size = self._fh.tell()  # ← record new EOF

    def _skip_existing_lines_once(self):
        """
        Begin reading from EOF and treat only newly appended lines as input.
        Do NOT rewind/scan; this avoids racing with a writer during startup
        and ignores any pre-existing content left from a previous run.
        """
        self._ensure_open()
        self._fh.seek(0, os.SEEK_END)
        self._last_size = self._fh.tell()
        self._next_index = 0
        self._started = True

    def next_completed_line(self, poll_interval_sec: float) -> Optional[tuple[int, str]]:
        """
        Returns (line_index, line_text) when a line is confirmed complete by the arrival
        of a subsequent newline/blank line. Returns None if no new completed line yet.

        Behavior:
        - If the stream file doesn't exist yet, we wait (do NOT create it).
        - On first open (or after truncate/rotation), we start at EOF (ignore old content).
        - A line is emitted only when a subsequent line (including blank) arrives.
        """
        # If the stream file does not exist yet, wait until partner creates it.
        if not self.path.exists():
            # Ensure we don't hold any stale file handles/state
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None
            self._buffer = None
            self._started = False
            self._last_size = 0
            time.sleep(poll_interval_sec)
            return None

        # Open (at EOF) if not already open
        self._ensure_open()

        # Detect file truncation/rotation
        try:
            cur_size = os.path.getsize(self.path)
        except FileNotFoundError:
            # File vanished between exists() and getsize(); clear state and wait
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
            self._fh = None
            self._buffer = None
            self._started = False
            self._last_size = 0
            time.sleep(poll_interval_sec)
            return None

        if cur_size < self._last_size:
            # Truncated/cleared → drop the handle and state.
            # We'll reopen at EOF on a future call once the writer recreates/grows it.
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
            self._fh = None
            self._buffer = None
            self._next_index = 0
            self._started = False
            self._processed_at_least_one = False
            self._last_size = 0
            time.sleep(poll_interval_sec)
            return None

        self._last_size = cur_size

        # On first run after open/reset, anchor at EOF and ignore pre-existing content
        if not self._started:
            self._skip_existing_lines_once()
            return None

        # Read any new lines appended since last call
        self._fh.seek(0, os.SEEK_CUR)  # ensure pointer stable
        new_line = self._fh.readline()

        # Nothing new written
        if new_line == "":
            time.sleep(poll_interval_sec)
            return None

        # Normalize line endings; preserve empty/blank lines as delimiters
        line = new_line.rstrip("\r\n")

        # If we have a buffered (previous) line and now see ANY line (even blank),
        # we can safely emit the buffered one as a completed record.
        if self._buffer is not None:
            completed = self._buffer
            idx = self._next_index
            self._next_index += 1
            # Set new buffer to current line ONLY if it's non-blank; else drop it.
            self._buffer = line if line.strip() else None
            return idx, completed

        # No buffered line yet: store this line if it's non-blank; otherwise ignore.
        if line.strip():
            self._buffer = line
        # If it's blank, we just ignore (could be multiple blank lines)
        return None

# =========================
# Main edge loop entrypoint
# =========================

def run_assistant(
    ems_profile: Dict,
    patwit_profile: Dict,
    cfg: LoopConfig,
    deps: Deps,
    *,
    process_one: bool = False,
) -> None:
    """
    Central loop: now tails a single stream file (`cfg.transcript_path`) and
    keeps all downstream logic identical (adapter → QA/extractor → merge → cloud → outbox).
    """
    # ---- Cloud/incident state ----
    incident_id: Optional[str] = None
    incident_started_iso: Optional[str] = None
    incident_initialized: bool = False

    # Ensure essential dirs exist
    cfg.outbox_dir.mkdir(parents=True, exist_ok=True)
    cfg.transcript_path.parent.mkdir(parents=True, exist_ok=True)

    last_llm_call_ts: Optional[float] = None
    last_activity_ts: float = time.time()

    # Fallback sequence for event ids when the tailer doesn't provide an index
    next_seq = 0
    def _finalize_and_exit(reason: str) -> None:
        """Call the close hook once, then return."""
        try:
            closed_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
            meta = {
                "incident_id": incident_id,
                "started_iso": incident_started_iso,
                "closed_iso": closed_iso,
                "reason": reason,
            }
            try:
                # New (preferred): 3-arg close hook
                deps.on_incident_close(ems_profile, patwit_profile, meta)  # type: ignore[arg-type]
            except TypeError:
                # Back-compat: old 2-arg hook
                deps.on_incident_close(ems_profile, patwit_profile)  # type: ignore[misc]
        except Exception as e:
            print(f"[edge][WARN] on_incident_close failed: {e}")
        print(f"[edge] Incident closed ({reason}).")

    tail = StreamTail(cfg.transcript_path)

    # --- Anchor at EOF immediately (avoids startup race) ---
    # Only do this if the file already exists; we never create it ourselves.
    try:
        if cfg.transcript_path.exists():
            tail._skip_existing_lines_once()   # positions at current EOF
    except Exception as e:
        print(f"[edge][WARN] initial tail anchor failed: {e}")

    while True:
        # Pull next completed line from the stream (None if nothing new yet)
        # Pull next completed line from the stream (None if nothing new yet)
        nxt = tail.next_completed_line(cfg.poll_interval_sec)

        # Unpack & compute line_index
        line_text = None
        line_index = None

        if nxt is None:
            line_text = None
        else:
            # StreamTail returns (index, text)
            if isinstance(nxt, tuple) and len(nxt) == 2:
                line_index, line_text = nxt
            else:
                # Fallback: treat as a single line payload
                line_text = str(nxt)
                line_index = next_seq
                next_seq += 1

        # Idle timeout if nothing completes or just a blank delimiter line
        if line_text is None or not str(line_text).strip():
            idle_for = time.time() - last_activity_ts
            if idle_for >= cfg.idle_timeout_sec:
                # Try to flush a pending line once before closing
                flushed = tail.flush_pending()
                if flushed:
                    line_index, line_text = flushed
                    # fall through to normal processing of this last line
                else:
                    # No pending line → announce + close incident due to idle
                    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

                    if incident_initialized and incident_id:
                        admin_event_id = f"evt-{next_seq:06d}-admin"
                        close_msg = "Incident closing due to timeout."

                        print(f"[edge] Idle timeout reached ({idle_for:.1f}s) — announcing and closing incident.")

                        # Log to Firestore
                        try:
                            deps.cloud.append_conversation(
                                incident_id=incident_id,
                                event_id=admin_event_id,
                                role="zora",
                                timestamp_iso=now_iso,
                                normalized_text=close_msg,
                            )
                        except Exception as e:
                            print(f"[edge][cloud][WARN] append_conversation (idle-close) failed for {admin_event_id}: {e}")

                        # Mark complete (idle)
                        try:
                            deps.cloud.mark_complete(
                                incident_id=incident_id,
                                close_time_iso=now_iso,
                                close_reason="idle",
                            )
                        except Exception as e:
                            print(f"[edge][cloud][WARN] mark_complete (idle) failed: {e}")

                        # Publish to MasterFileOutput.jsonl
                        if deps.publish_zora:
                            try:
                                deps.publish_zora(
                                    incident_id,
                                    admin_event_id,
                                    now_iso,
                                    "zora",
                                    close_msg,
                                )
                            except Exception as e:
                                print(f"[edge][WARN] publish_zora (idle-close) failed: {e}")

                    _finalize_and_exit(reason=f"idle for {idle_for:.1f}s")
                    return
            else:
                continue

        # From here on, use these:
        raw = str(line_text)
        base_name = f"evt-{line_index:06d}"  # used in event_ids / logs


        try:
            # Adapter converts raw text to a structured event (or raises ValueError)
            event = deps.normalize_incoming_message(raw)

            # ---- Initialize incident on FIRST valid line ----
            if not incident_initialized:
                incident_id = f"incident_{event.timestamp.strftime('%Y%m%d_%H%M%S')}"
                incident_started_iso = event.timestamp.isoformat()
                try:
                    deps.cloud.init_incident(incident_id, incident_started_iso)
                    incident_initialized = True
                    print(f"[edge] Incident init: {incident_id} (started {incident_started_iso})")
                except Exception as e:
                    print(f"[edge][cloud][WARN] init_incident failed: {e}")

            # Pull normalized fields
            event_time_hhmmss = event.timestamp.strftime("%H:%M:%S")
            role: str = event.role
            text: str = event.text
            zora_prompt_raw = getattr(event, "zora_prompt", None)

            # ---- Cloud log input line ----
            event_id = base_name
            timestamp_iso = event.timestamp.isoformat()
            role_for_cloud = "EMS" if event.role.upper() == "EMS" else "other"
            try:
                deps.cloud.append_conversation(
                    incident_id=incident_id,
                    event_id=event_id,
                    role=role_for_cloud,
                    timestamp_iso=timestamp_iso,
                    raw_input=getattr(event, "raw_input", None),
                    normalized_text=event.text,
                )
            except Exception as e:
                print(f"[edge][cloud][WARN] append_conversation failed for {event_id}: {e}")

            # --- Strip only LEADING chit-chat/fillers before agent processing ---
            # We keep Firestore logging above intact (logs original text).
            stripped_text = _strip_leading_fillers(text)

            if stripped_text != text:
                print(f"[edge] Leading filler removed: {base_name} -> '{text[:32]}' -> '{stripped_text[:32]}'")

            # If the whole line was just filler, skip agent processing (but keep it logged)
            if not stripped_text:
                last_activity_ts = time.time()
                print(f"[edge] Skipped (filler-only): {base_name}")
                if process_one:
                    return
                continue

            # Use the stripped text for the rest of the pipeline
            text = stripped_text

            # If the adapter didn’t detect a Zora command due to leading filler,
            # recover it here for EMS lines (handles 'Zora...' with optional comma).
            if role.upper() == "EMS" and not zora_prompt_raw:
                m = re.match(r'^\s*zora[:,]?\s*(.*)\s*$', text, flags=re.IGNORECASE)
                if m:
                    zora_prompt_raw = m.group(1)


            # -------------------------------
            # EMS "Zora, ..." COMMAND ROUTER
            # -------------------------------
            if zora_prompt_raw and role.upper() == "EMS":
                cmd = _normalize_command(zora_prompt_raw)

                if cmd in {"end incident"}:
                    last_activity_ts = time.time()
                    print(f"[edge] Received end-incident command: {base_name}")

                    # Cloud: log admin and mark complete
                    admin_event_id = f"{base_name}-admin"
                    confirm_text = "Incident closed by EMS."
                    try:
                        deps.cloud.append_conversation(
                            incident_id=incident_id,
                            event_id=admin_event_id,
                            role="zora",
                            timestamp_iso=event.timestamp.isoformat(),
                            normalized_text=confirm_text,
                        )
                    except Exception as e:
                        print(f"[edge][cloud][WARN] append_conversation (admin-end) failed for {admin_event_id}: {e}")

                    try:
                        close_time_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
                        deps.cloud.mark_complete(
                            incident_id=incident_id,
                            close_time_iso=close_time_iso,
                            close_reason="explicit",
                        )
                    except Exception as e:
                        print(f"[edge][cloud][WARN] mark_complete failed: {e}")

                    # nlp_out bus
                    if deps.publish_zora and incident_initialized and incident_id:
                        deps.publish_zora(
                            incident_id,
                            f"{base_name}-admin",
                            event.timestamp.isoformat(),
                            "zora",
                            confirm_text,
                        )

                    _finalize_and_exit(reason="explicit end incident command")
                    return

                elif cmd in {"delete the last entry", "delete last entry"}:
                    status_msg = deps.rollback_last_entry(ems_profile)

                    # Cloud: log rollback + snapshot
                    admin_event_id = f"{base_name}-admin"
                    try:
                        deps.cloud.append_conversation(
                            incident_id=incident_id,
                            event_id=admin_event_id,
                            role="zora",
                            timestamp_iso=event.timestamp.isoformat(),
                            normalized_text=status_msg,
                        )
                    except Exception as e:
                        print(f"[edge][cloud][WARN] append_conversation (admin-rollback) failed for {admin_event_id}: {e}")

                    try:
                        deps.cloud.upsert_profile_snapshots(
                            incident_id=incident_id,
                            timestamp_iso=event.timestamp.isoformat(),
                            ems_snapshot=ems_profile,
                        )
                    except Exception as e:
                        print(f"[edge][cloud][WARN] upsert_profile_snapshots (rollback) failed: {e}")

                    if deps.publish_zora and incident_initialized and incident_id:
                        deps.publish_zora(
                            incident_id,
                            f"{base_name}-admin",
                            event.timestamp.isoformat(),
                            "zora",
                            status_msg,
                        )

                    last_activity_ts = time.time()
                    print(f"[edge] Rollback handled: {base_name} -> {status_msg}")
                    if process_one:
                        return
                    continue

                else:
                    # QA PATH
                    last_llm_call_ts = _rate_limit(last_llm_call_ts, cfg.max_rps)
                    reply_text = deps.ask_zora(ems_profile, patwit_profile, zora_prompt_raw)

                    # Cloud: log reply
                    zora_event_id = f"{base_name}-reply"
                    try:
                        deps.cloud.append_conversation(
                            incident_id=incident_id,
                            event_id=zora_event_id,
                            role="zora",
                            timestamp_iso=event.timestamp.isoformat(),
                            normalized_text=reply_text,
                        )
                    except Exception as e:
                        print(f"[edge][cloud][WARN] append_conversation (reply) failed for {zora_event_id}: {e}")

                    if deps.publish_zora and incident_initialized and incident_id:
                        deps.publish_zora(
                            incident_id,
                            f"{base_name}-reply",
                            event.timestamp.isoformat(),
                            "zora",
                            reply_text,
                        )

                    last_activity_ts = time.time()
                    print(f"[edge] QA handled: {base_name} (prompt='{zora_prompt_raw[:48]}...')")
                    if process_one:
                        return
                    continue

            # -----------------------
            # EXTRACTOR / MERGE PATH
            # -----------------------
            last_llm_call_ts = _rate_limit(last_llm_call_ts, cfg.max_rps)
            updates: Dict = deps.extract_patient_info(role, event_time_hhmmss, text)  # may be {}
            if updates:
                deps.smart_update_profiles(
                    ems_profile, patwit_profile, role, updates, event_time_hhmmss
                )
                print(f"[edge] Merged updates: {base_name} -> {list(updates.keys())}")

                # Cloud: push snapshots after merge
                try:
                    if role.upper() == "EMS":
                        deps.cloud.upsert_profile_snapshots(
                            incident_id=incident_id,
                            timestamp_iso=event.timestamp.isoformat(),
                            ems_snapshot=ems_profile,
                        )
                    else:
                        deps.cloud.upsert_profile_snapshots(
                            incident_id=incident_id,
                            timestamp_iso=event.timestamp.isoformat(),
                            patwit_snapshot=patwit_profile,
                        )
                except Exception as e:
                    print(f"[edge][cloud][WARN] upsert_profile_snapshots (extract) failed: {e}")
            else:
                print(f"[edge] No updates: {base_name}")

            last_activity_ts = time.time()
            print(f"[edge] Processed: {base_name}")

        except Exception as e:
            # We don't move files in stream mode; just log the error and continue.
            print(f"[edge][ERR] {base_name}: {type(e).__name__}: {e}")

        if process_one:
            return
