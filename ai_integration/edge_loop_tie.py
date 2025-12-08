# edge_loop_tie.py
from pathlib import Path
from typing import Dict, Any
import os
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", r"C:\Users\thoma\OneDrive\Capstone\Code\GoogleCloudKey.json")

from dotenv import load_dotenv
load_dotenv()  # loads .env from current working directory

import sys, subprocess, shlex, threading, traceback

# --- nlp_out addition ---
from nlp_out import NlpOutbox

from edge_loop import run_assistant, LoopConfig, CloudDeps, Deps
import cloud_io

# ---- your existing adapters/agents/merger/rollback imports ----
from edge_input_adapter import normalize_incoming_message, RoleResolver
from qa_agent import ask_zora
from extractor_agent import extract_patient_info
from edge_data_ex import smart_update_profiles, rollback_last_entry

# Path to email_report.py sitting next to this file
REPORT_SCRIPT = Path(__file__).with_name("email_report.py")

def on_incident_close(ems, patwit, meta=None):
    """
    After an incident ends, generate & email the transfer report by invoking email_report.py.
    - Non-blocking: runs in a worker thread.
    - No extra Firestore/outbox messages (per your request).
    - Inherits env (GOOGLE_APPLICATION_CREDENTIALS / GCP_SA_KEY_JSON_B64, etc.).
    """
    incident_id = (meta or {}).get("incident_id")

    # Optional: allow a CHROME_PATH env override for headless print-to-PDF
    # Allow override; default to Chromium path in Linux containers if unset
    chrome_path = os.environ.get("CHROME_PATH") or ("/usr/bin/chromium" if os.name == "posix" else None)

    def _worker():
        try:
            # Build CLI
            cmd = [
                sys.executable, str(REPORT_SCRIPT),
                "--incident-id", str(incident_id or ""),
                "--outdir", "reports",
                "--oauth-method", "console",  # stays headless/terminal-friendly
            ]
            if chrome_path:
                cmd += ["--chrome-path", chrome_path]

            # Run with a sane timeout (adjust if your render + OAuth can take longer)
            subprocess.run(cmd, check=True, timeout=180)
        except Exception as e:
            # No Firestore/outbox logs by design; emit console trace for debugging
            print("[tie][ERR] email_report worker failed:", e)
            traceback.print_exc()

    # Use a NON-daemon worker so the process won’t exit before report finishes
    threading.Thread(target=_worker, daemon=False).start()


def main():
    # Stream input: single fixed file continuously appended by your partner
    transcript_path = Path("MasterFileInput.jsonl")  # <- set to the actual path partner writes

    # Not Used Anymore
    outbox = Path("mailbox/outbox")

    cfg = LoopConfig(
        transcript_path=transcript_path,
        outbox_dir=outbox,
        # The rest are unused in stream mode but kept for compatibility
        inbox_dir=Path(".unused_inbox"),
        processed_dir=Path(".unused_processed"),
        processed_err_dir=Path(".unused_processed_err"),
        max_rps=2.0,
        idle_timeout_sec=120.0,
        poll_interval_sec=0.25,
    )

    print(f"[tie] Tailing stream: {transcript_path.resolve()}")

    # Cloud client config (project_id optional; ADC usually infers from key)
    cloud_cfg = cloud_io.CloudIOConfig(project_id=None)

    cloud = CloudDeps(
        init_incident=lambda incident_id, started_iso: cloud_io.init_incident(
            cloud_cfg, incident_id, started_iso
        ),
        append_conversation=lambda incident_id, event_id, role, timestamp_iso, **kw: cloud_io.append_conversation(
            cloud_cfg, incident_id, event_id, role, timestamp_iso=timestamp_iso, **kw
        ),
        upsert_profile_snapshots=lambda incident_id, timestamp_iso, **kw: cloud_io.upsert_profile_snapshots(
            cloud_cfg, incident_id, timestamp_iso=timestamp_iso, **kw
        ),
        mark_complete=lambda incident_id, close_time_iso, close_reason: cloud_io.mark_complete(
            cloud_cfg, incident_id, close_time_iso=close_time_iso, close_reason=close_reason
        ),
    )

    # Role resolver instance (persisted)
    resolver = RoleResolver()

    # Adapter wrapper: matches the edge_loop signature (single-arg raw line)
    def _norm(raw: str):
        return normalize_incoming_message(raw, resolver)

    # nlp_out bus for QA replies (optional)
    nlp_bus = NlpOutbox(Path("MasterFileOutput.jsonl"))

    def _publish_zora_cb(incident_id: str, event_id: str, timestamp_iso: str, role: str, text: str) -> None:
        nlp_bus.publish(
            incident_id=incident_id,
            text=text,
        )

    deps = Deps(
        normalize_incoming_message=_norm,
        ask_zora=ask_zora,
        extract_patient_info=extract_patient_info,
        smart_update_profiles=smart_update_profiles,
        rollback_last_entry=rollback_last_entry,
        on_incident_close=on_incident_close,
        cloud=cloud,
        publish_zora=_publish_zora_cb,
    )

    # Fresh in-memory profiles per incident
    ems_profile: Dict[str, Any] = {}
    patwit_profile: Dict[str, Any] = {}

    # Run the loop (Ctrl+C to stop)
    run_assistant(ems_profile, patwit_profile, cfg, deps, process_one=False)

if __name__ == "__main__":
    main()
