import os
from datetime import datetime
from google.cloud import firestore
from jinja2 import Environment, FileSystemLoader
import json  # For optional debug printing

# --- TIME FORMATTING FUNCTION ---

def format_time(value):
    """
    Formats an ISO timestamp string to 'HH:MM:SS'.
    Example: '2025-04-11T15:49:30.416087' ➔ '15:49:30'
    """
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime("%H:%M:%S")  # ✅ Includes hours, minutes, seconds
    except Exception:
        return value  # fallback, in case format is wrong



# --- CONFIGURATION --

def get_most_recent_incident_id():
    """Fetches the most recent completed incident ID from Firestore."""
    db = firestore.Client()
    patients_ref = db.collection("patients")
    query = patients_ref \
        .where("incident_meta.status", "==", "complete") \
        .order_by("incident_meta.started_at", direction=firestore.Query.DESCENDING) \
        .limit(1)
    docs = query.stream()
    for doc in docs:
        return doc.id
    raise ValueError("No incidents found in Firestore.")

# File paths
template_path = "transfer_of_care_template.html"  # Path to your Jinja2 HTML template

# --- FIELD DEFINITIONS ---

# List of expected fields in the patient profile
expected_patient_fields = [
    "identification",               # Contains 'name' and 'alt_description'
    "age", "sex", "weight", "blood_type", "date_of_birth", "social_security_number",
    "address", "patient_contact_info", "next_of_kin_name", "next_of_kin_phone_number",
    "blood_pressure", "heart_rate", "respiratory_rate", "oxygen_saturation",
    "glucose_level", "gcs_score",
    "injury", "symptoms", "meds", "interventions", "airway_status",
    "scene_info", "bystander_info", "patient_behavior", "notes",
    "transport_destination", "transport_mode", "transport_time_pickup", "transport_time_dropoff",
    "timeline_events", "allergies", "medical_history", "last_intake",
    "advance_directive", "language_barrier", "mental_status",
    "vitals_log", "meds_log"
]

# --- FIRESTORE FETCH FUNCTION ---

def fetch_patient_info(incident_id):
    """Fetches the patient_profile section from Firestore for the given incident ID."""
    db = firestore.Client()
    doc_ref = db.collection("patients").document(incident_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("patient_profile", {})
    else:
        raise ValueError(f"No Firestore document found for ID: {incident_id}")

# --- DATA EXPANSION FUNCTION ---

def expand_patient_info_for_report(patient_info):
    """
    Prepares patient profile data for the report:
    - Ensures all fields are present.
    - Applies clean formatting.
    - Formats time in vitals_log and meds_log.
    """
    report_ready = {}

    for field in expected_patient_fields:
        if field not in patient_info:
            report_ready[field] = "—"  # Placeholder for missing fields

        elif isinstance(patient_info[field], dict):
            # Handle nested fields like 'identification'
            report_ready[field] = {
                k: (v if v not in [None, [], ""] else "—")
                for k, v in patient_info[field].items()
            }

        elif isinstance(patient_info[field], list):
            if field in ["vitals_log", "meds_log"]:
                # ✅ Clean time formatting for logs
                cleaned_list = []
                for entry in patient_info[field]:
                    if isinstance(entry, dict) and "time" in entry:
                        entry = entry.copy()  # Avoid modifying original
                        entry["time"] = format_time(entry["time"])
                    cleaned_list.append(entry)
                report_ready[field] = cleaned_list
            else:
                # Regular lists as comma-separated strings
                report_ready[field] = ", ".join(map(str, patient_info[field])) if patient_info[field] else "—"

        else:
            # Normal value handling
            report_ready[field] = patient_info[field] if patient_info[field] not in [None, ""] else "—"

    return report_ready

# --- HTML RENDERING FUNCTION ---

def render_html_report(data, template_file, output_file):
    """Renders the HTML template with patient data and saves the output file."""
    # Initialize Jinja2 environment
    env = Environment(loader=FileSystemLoader("."))  # Templates in current directory
    template = env.get_template(template_file)  # Load template file

    # Render the template with the provided data
    rendered = template.render(**data)

    # Write the output HTML file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(rendered)
    print(f"✅ Report generated: {output_file}")

# --- MAIN SCRIPT EXECUTION ---

if __name__ == "__main__":
    try:
        # Authenticate with your service account
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GoogleCloudKey.json"

        # Fetch the latest incident ID
        incident_id = get_most_recent_incident_id()

        # Define output path for the report
        output_path = f"report_{incident_id}.html"

        # Fetch and prepare patient data
        raw_data = fetch_patient_info(incident_id)
        filled_data = expand_patient_info_for_report(raw_data)

        # Generate the report
        render_html_report(filled_data, template_path, output_path)

    except Exception as e:
        print("❌ Error generating report:", e)
