# email_report.py
# Render HTML -> PDF -> create Gmail draft with PDF attached.
# Subject: "'<Name or IncidentID>' Patient Care Report"
# Reads Firestore service account key from:
#   1) GCP_SA_KEY_JSON_B64 (base64-encoded JSON), or
#   2) GOOGLE_APPLICATION_CREDENTIALS / GCP_SA_KEY_PATH (file path), or
#   3) fallback to generate_transfer_report.CREDS_PATH

import os
import sys
import argparse
import json
import base64
import mimetypes
import subprocess
from pathlib import Path
from typing import Optional
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

from dotenv import load_dotenv
load_dotenv()  # picks up .env beside this script or from CWD

import generate_transfer_report as gtr

from google.cloud import firestore
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# -------------------- Config --------------------
DEFAULT_TEMPLATE = "patient_care_report_template.html"
DEFAULT_OUTDIR = "./reports"

GMAIL_CLIENT_SECRET = "gmail_client_secret.json"
GMAIL_TOKEN = "gmail_token.json"
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]

COMMON_CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

# -------------------- Env-based Firestore auth --------------------
def firestore_client_from_env() -> firestore.Client:
    """
    Priority:
      1) GCP_SA_KEY_JSON_B64 (base64-encoded full SA JSON)
      2) GOOGLE_APPLICATION_CREDENTIALS or GCP_SA_KEY_PATH (file path)
      3) gtr.CREDS_PATH fallback
    """
    b64 = os.getenv("GCP_SA_KEY_JSON_B64")
    key_path = (
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GCP_SA_KEY_PATH")
        or getattr(gtr, "CREDS_PATH", None)
    )

    if b64:
        info = json.loads(base64.b64decode(b64).decode("utf-8"))
        creds = service_account.Credentials.from_service_account_info(info)
    elif key_path:
        creds = service_account.Credentials.from_service_account_file(key_path)
    else:
        raise RuntimeError(
            "No Firestore credentials found. Set GCP_SA_KEY_JSON_B64 or GOOGLE_APPLICATION_CREDENTIALS / GCP_SA_KEY_PATH, "
            "or ensure generate_transfer_report.CREDS_PATH is valid."
        )

    project_id = getattr(creds, "project_id", None)
    if not project_id:
        raise RuntimeError("Service account JSON missing project_id.")
    return firestore.Client(project=project_id, credentials=creds)

# -------------------- Helpers --------------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def find_chrome(explicit_path: Optional[str]) -> str:
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    for p in COMMON_CHROME_PATHS:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Chrome/Edge not found. Install Chrome/Edge or pass --chrome-path to the executable."
    )

# def html_to_pdf(chrome_path: str, html_path: str, pdf_path: str):
#     url = "file:///" + os.path.abspath(html_path).replace("\\", "/")
#     cmd = [
#         chrome_path,
#         "--headless=new",
#         "--disable-gpu",
#         f"--print-to-pdf={os.path.abspath(pdf_path)}",
#         "--print-to-pdf-no-header",
#         url,
#     ]
#     subprocess.run(cmd, check=True)

def html_to_pdf(chrome_path: str, html_path: str, pdf_path: str):
    # Build file URL and absolute paths (keeps Windows + Linux behavior)
    abs_html = os.path.abspath(html_path)
    abs_pdf = os.path.abspath(pdf_path)
    url = "file:///" + abs_html.replace("\\", "/")

    # Add flags only when needed (e.g., running as root in a Linux container)
    extra_flags = []
    try:
        if os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() == 0:
            extra_flags += ["--no-sandbox", "--disable-dev-shm-usage"]
    except Exception:
        pass

    # Optional: allow extra args via env (space-separated), e.g. CHROME_EXTRA_ARGS="--some-flag"
    env_extra = os.environ.get("CHROME_EXTRA_ARGS", "").strip()
    if env_extra:
        extra_flags += [arg for arg in env_extra.split() if arg]

    cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        *extra_flags,
        f"--print-to-pdf={abs_pdf}",
        "--print-to-pdf-no-header",
        url,
    ]
    subprocess.run(cmd, check=True)


def gmail_service(oauth_method: str = "console"):
    if not os.path.exists(GMAIL_CLIENT_SECRET):
        raise FileNotFoundError(
            f"Missing {GMAIL_CLIENT_SECRET}. Download an OAuth client (Desktop) from Google Cloud Console."
        )
    creds = None
    if os.path.exists(GMAIL_TOKEN):
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN, GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CLIENT_SECRET, GMAIL_SCOPES)

            # "console" path uses local server without opening a browser (prints URL + waits for paste)
            if oauth_method == "local":
                creds = flow.run_local_server(port=0)
            else:  # "console"
                creds = flow.run_local_server(port=0, open_browser=False)

        with open(GMAIL_TOKEN, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)

def make_mime_with_attachment(subject: str, body_text: str, attachment_path: str) -> str:
    message = MIMEMultipart()
    message["Subject"] = subject  # leave "To" blank

    message.attach(MIMEText(body_text, "plain"))

    ctype, encoding = mimetypes.guess_type(attachment_path)
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)

    with open(attachment_path, "rb") as f:
        part = MIMEBase(maintype, subtype)
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(attachment_path)}"')
    message.attach(part)

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    return raw

def create_gmail_draft(service, raw_message_b64url: str):
    draft = {"message": {"raw": raw_message_b64url}}
    return service.users().drafts().create(userId="me", body=draft).execute()

def _clean_str_or_none(v) -> Optional[str]:
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return None

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Render report → PDF → Gmail draft with PDF attached.")
    parser.add_argument("--incident-id", default=None, help="Incident to render; omit to pick most recent.")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="Path to Jinja2 HTML template.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Directory for HTML/PDF outputs.")
    parser.add_argument("--chrome-path", default=None, help="Path to Chrome/Edge executable (optional).")
    parser.add_argument("--oauth-method", choices=["console", "local"], default="console",
                        help="Gmail OAuth method. Use 'console' for headless; 'local' opens a browser.")
    parser.add_argument("--body", default="Hello Recipient, \n\n EMS Team6 is in response of an emergency call. \n\n Attached is the Patient Care Report, containing crucial patient information.\n\nPlease prepare your staff for patient arrival.",
                        help="Email body text.")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    # Firestore via env-based credentials (supports .env)
    db = firestore_client_from_env()

    # Choose incident
    incident_id = args.incident_id or gtr.get_most_recent_incident_id(db)

    # Render HTML using your renderer (includes PatWit overlay at render-time)
    incident_meta  = gtr.fetch_incident_meta(db, incident_id)
    ems_profile    = gtr.fetch_profile(db, incident_id, "ems_profile")
    patwit_profile = gtr.fetch_profile(db, incident_id, "patwit_profile")
    html = gtr.render_html_report(args.template, ems_profile, patwit_profile, incident_meta)

    html_path = os.path.join(args.outdir, f"report_{incident_id}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Print to PDF via headless Chrome/Edge
    chrome_path = find_chrome(args.chrome_path)
    pdf_path = os.path.join(args.outdir, f"report_{incident_id}.pdf")
    html_to_pdf(chrome_path, html_path, pdf_path)

    # Subject: "'<Name or IncidentID>' Patient Care Report"
    ems_for_subject = gtr.overlay_identity_for_report(dict(ems_profile or {}), dict(patwit_profile or {}))
    candidate_name = _clean_str_or_none(ems_for_subject.get("name"))
    display = candidate_name or incident_id
    subject = f"'{display}' Patient Care Report"

    # Create Gmail draft with attachment
    service = gmail_service(oauth_method=args.oauth_method)
    raw = make_mime_with_attachment(subject, args.body, pdf_path)
    draft = create_gmail_draft(service, raw)

    print("Gmail draft created.")
    print(f"   Subject: {subject}")
    print(f"   Attachment: {pdf_path}")
    print(f"   Draft ID: {draft.get('id')}  (check Gmail Drafts)")

if __name__ == "__main__":
    main()
