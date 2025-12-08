# generate_transfer_report.py
import os
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple

from google.cloud import firestore
from google.oauth2 import service_account
from jinja2 import Environment, FileSystemLoader, select_autoescape

from dotenv import load_dotenv
load_dotenv()  # loads .env from current working directory

CREDS_PATH = r"C:\Users\thoma\OneDrive\Capstone\Code\GoogleCloudKey.json"
DASH = "—"


# PatWit → EMS identity fields eligible for one-way upgrade at render time
IDENTITY_KEYS = [
    "name", "alt_description", "age", "sex", "weight", "blood_type",
    "date_of_birth", "social_security_number", "address",
    "patient_contact_info", "next_of_kin_name", "next_of_kin_phone_number",
]

def _is_empty(v):
    """Empty if None, '', or [] (lists only)."""
    if v is None: return True
    if isinstance(v, str) and v.strip() == "": return True
    if isinstance(v, list) and len(v) == 0: return True
    return False

def _normalize_for_ems(key, value):
    """Keep PHI intact; coerce alt_description to list if needed."""
    if key == "alt_description" and isinstance(value, str):
        return [value]
    return value

def overlay_identity_for_report(ems: dict, pat: dict) -> dict:
    """
    Return a COPY of EMS with missing identity fields filled from PatWit (if present).
    No Firestore writes. Render-time only.
    """
    out = dict(ems or {})
    p = pat or {}
    for key in IDENTITY_KEYS:
        if _is_empty(out.get(key)) and not _is_empty(p.get(key)):
            out[key] = _normalize_for_ems(key, p[key])
    return out


# ----------------------------- Time helpers ----------------------------- #
def to_hhmmss(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) >= 8 and value[2] == ":" and value[5] == ":":
            return value[:8]
        try:
            dt = datetime.fromisoformat(value)
            return dt.strftime("%H:%M:%S")
        except Exception:
            return value
    return value

def sort_by_time(entries: Any) -> Any:
    if not isinstance(entries, list):
        return entries
    def key(e):
        t = (e or {}).get("time")
        t_norm = to_hhmmss(t) if isinstance(t, str) else None
        return (0, t_norm) if isinstance(t_norm, str) and len(t_norm) == 8 else (1, "")
    return sorted(entries, key=key)

def _split_incident_datetime(started_time: Any) -> Tuple[str, str]:
    """
    Returns (date_str 'DD Month YYYY', time_str 'HH:MM:SS').
    Gracefully degrades if only time is available or parsing fails.
    """
    if isinstance(started_time, str):
        # Already looks like HH:MM:SS?
        if len(started_time) >= 8 and started_time[2] == ":" and started_time[5] == ":":
            return ("—", started_time[:8])
        try:
            dt = datetime.fromisoformat(started_time)
            return (dt.strftime("%d %B %Y"), dt.strftime("%H:%M:%S"))
        except Exception:
            pass
    return ("—", to_hhmmss(started_time) if isinstance(started_time, str) else "—")

# ------------------------ Placeholder normalization --------------------- #
def dash_scalar(v: Any) -> Any:
    if v is None:
        return DASH
    if isinstance(v, str) and v.strip() == "":
        return DASH
    return v

def dashify_object_shallow(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            out[k] = v
        else:
            out[k] = dash_scalar(v)
    return out

# ----------------------------- Firestore I/O ---------------------------- #
def get_most_recent_incident_id(db: firestore.Client) -> str:
    """
    Robust picker: prefer parent 'started_time'; fall back to 'incident_meta.started_at'.
    """
    patients = db.collection("patients")

    # Try new schema: started_time
    try:
        q = patients.order_by("started_time", direction=firestore.Query.DESCENDING).limit(1)
        for doc in q.stream():
            return doc.id
    except Exception:
        pass

    # Fall back: old schema: incident_meta.started_at
    try:
        q = patients.order_by("incident_meta.started_at", direction=firestore.Query.DESCENDING).limit(1)
        for doc in q.stream():
            return doc.id
    except Exception:
        pass

    # Last resort: just grab any doc (unordered)
    any_doc = patients.limit(1).get()
    if any_doc:
        return any_doc[0].id

    raise ValueError("No incidents found in Firestore.")

def fetch_incident_meta(db: firestore.Client, incident_id: str) -> Dict[str, Any]:
    snap = db.collection("patients").document(incident_id).get()
    if not snap.exists:
        raise ValueError(f"No Firestore doc for incident: {incident_id}")
    data = snap.to_dict() or {}

    # Normalize keys (support both old/new)
    started_time = data.get("started_time")
    if started_time is None:
        # old shape
        im = data.get("incident_meta") or {}
        started_time = im.get("started_at") or im.get("started_time")

    return {
        "incident_id": data.get("incident_id", incident_id),
        "started_time": started_time,
    }

def fetch_profile(db: firestore.Client, incident_id: str, which: str) -> Dict[str, Any]:
    snap = (
        db.collection("patients")
          .document(incident_id)
          .collection("profiles")
          .document(which)
          .get()
    )
    return snap.to_dict() or {} if snap.exists else {}

# ------------------------- Prep for rendering --------------------------- #
def normalize_and_sort_logs(ems: Dict[str, Any]) -> None:
    if isinstance(ems.get("vitals_log"), list):
        norm_vitals = []
        for entry in ems["vitals_log"]:
            if isinstance(entry, dict):
                e = dict(entry)
                if "time" in e:
                    e["time"] = to_hhmmss(e["time"])
                norm_vitals.append(e)
        ems["vitals_log"] = sort_by_time(norm_vitals)

    if isinstance(ems.get("meds_log"), list):
        norm_meds = []
        for entry in ems["meds_log"]:
            if isinstance(entry, dict):
                e = dict(entry)
                if "time" in e:
                    e["time"] = to_hhmmss(e["time"])
                norm_meds.append(e)
        ems["meds_log"] = sort_by_time(norm_meds)

def prepare_for_render(
    ems_profile: Dict[str, Any],
    patwit_profile: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    - Sort EMS time-based logs.
    - Overlay PatWit identity into EMS ONLY for missing fields (render-time).
    - Apply placeholders to first-level scalars.
    """
    ems = dict(ems_profile or {})
    pat = dict(patwit_profile or {})

    normalize_and_sort_logs(ems)

    # >>> render-time upgrade only (no DB writes) <<<
    ems = overlay_identity_for_report(ems, pat)

    ems_dash = dashify_object_shallow(ems)
    pat_dash = dashify_object_shallow(pat)

    return ems_dash, pat_dash

# ------------------------------- Render -------------------------------- #
def render_html_report(template_path: str, ems_profile: Dict[str, Any], patwit_profile: Dict[str, Any], incident_meta: Dict[str, Any]) -> str:
    template_dir, template_name = os.path.split(os.path.abspath(template_path))
    ems_ready, pat_ready = prepare_for_render(ems_profile, patwit_profile)

    env = Environment(
        loader=FileSystemLoader(template_dir or "."),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["dash"] = dash_scalar

    template = env.get_template(template_name)

    date_str, time_str = _split_incident_datetime(incident_meta.get("started_time"))
    html = template.render(
        **ems_ready,
        patwit=pat_ready,
        incident={
            "incident_id": incident_meta.get("incident_id"),
            "started_time": incident_meta.get("started_time"),
            "incident_date": date_str,
            "start_time_hms": time_str,
        },
    )
    
    return html

# --------------------------------- CLI --------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Generate transfer-of-care report (HTML).")
    parser.add_argument("--template", required=True, help="Path to the Jinja2 HTML template (e.g., patient_care_report_template.html)")
    parser.add_argument("--outdir", default=".", help="Output directory for the rendered HTML (default: current dir)")
    parser.add_argument("--incident-id", default=None, help="Incident ID to render. If omitted, picks most recent.")
    args = parser.parse_args()

    # Use project_id from the service account to avoid wrong-project errors
    creds = service_account.Credentials.from_service_account_file(CREDS_PATH)
    project_id = getattr(creds, "project_id", None)
    if not project_id:
        raise RuntimeError("Service account JSON has no project_id. Please update CREDS_PATH or set GOOGLE_CLOUD_PROJECT.")

    db = firestore.Client(project=project_id, credentials=creds)

    incident_id = args.incident_id or get_most_recent_incident_id(db)
    incident_meta  = fetch_incident_meta(db, incident_id)
    ems_profile    = fetch_profile(db, incident_id, "ems_profile")
    patwit_profile = fetch_profile(db, incident_id, "patwit_profile")

    html = render_html_report(args.template, ems_profile, patwit_profile, incident_meta)

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"report_{incident_id}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Report generated: {out_path}")

if __name__ == "__main__":
    main()
