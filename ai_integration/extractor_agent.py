# extractor_agent.py


import json
from typing import Dict, Any, List, Set
import re
import openai  # assumes you've set OPENAI_API_KEY

from dotenv import load_dotenv
load_dotenv()  # loads .env from the current working directory
#openai.api_key = "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"


# ---- short keys (LLM output) -> full field names (server) ----
SHORT_KEY_MAP: Dict[str, str] = {
    "nm": "name",
    "adesc": "alt_description",
    "age": "age",
    "sex": "sex",
    "wt": "weight",
    "bt": "blood_type",
    "dob": "date_of_birth",
    "ssn": "social_security_number",
    "addr": "address",
    "contact": "patient_contact_info",
    "nok_name": "next_of_kin_name",
    "nok_phone": "next_of_kin_phone_number",

    "bp": "blood_pressure",
    "hr": "heart_rate",
    "rr": "respiratory_rate",
    "spo2": "oxygen_saturation",
    "glu": "glucose_level",
    "gcs": "gcs_score",
    "wv": "witness_vitals",

    "inj": "injury",
    "sx": "symptoms",
    "airway": "airway_status",
    "mental": "mental_status",
    "scene": "scene_info",
    "behavior": "patient_behavior",
    "alg": "allergies",
    "hx": "medical_history",
    "loi": "last_oral_intake",
    "ad": "advance_directive",

    "ivx": "interventions",
    "meds": "meds",

    "dest": "transport_destination",
    "tmode": "transport_mode",
    "t_pickup": "transport_time_pickup",
    "t_drop": "transport_time_dropoff",

    "te": "timeline_events"  # EMS-only: we add the time when any vital present
}

ALLOW_KEYS: Dict[str, Set[str]] = {
    "EMS": {
        "nm","adesc","age","sex","wt","bt","dob","ssn","addr","contact","nok_name","nok_phone",
        "bp","hr","rr","spo2","glu","gcs",
        "inj","sx","airway","mental","scene","behavior","alg","hx","loi","ad",
        "ivx","meds",
        "dest","tmode","t_pickup","t_drop"
        # 'te' is set by server logic; LLM does not need to output it
    },
    "other": {
        "nm","adesc","age","sex","wt","bt","dob","ssn","addr","contact","nok_name","nok_phone", 'wv',
        "inj","sx","airway","mental","behavior","scene","alg","hx","loi","ad",
        "ivx"  # witness/self-aid only
        # vitals/meds/transport/timeline excluded
    }
}

    # "You extract structured EMS data from ONE line of transcript.\n"
    # "ROLE=EMS. Output ONLY a JSON object of updates using SHORT KEYS."
    # "If the line has no useful data, output {}.\n"
    # "Canonical units: bp='SYS/DIA', hr/rr=int, spo2=int 0-100, glu=int mg/dL, gcs='15' or 'E4 V5 M6'."
    # f"Allowed short keys: {allow}."
    # "Meds only when clearly administered (gave/administered/pushed/placed/started);"
    # "interventions are procedures (not meds). "
    # "Do not include transport or identity fields unless explicitly stated."
    # "Lines can include multiple useful items—extract all that apply in one JSON. Example: input: “His name is John Miller and he is wearing a blue jacket. Output: {'nm':'John Miller','adesc':['blue jacket']}"
    # "Airway is only for states like patent, compromised, obstructed, maintained with OPA/NPA. Device (such as non-rebreather/NRB, nasal cannula, CPAP) placement belongs in ivx."
    # "Do not guess blood_type. Only fill if explicitly stated like ‘Type O positive’."
    # "scene_info = environment/location only; injury = bodily harm. Do not cross-fill."
    # "alt_description = visible descriptors only (clothing, hair, marks). Never put actions, locations, injuries, or scene details here."
    # "If a DNR/DNI bracelet, card, paperwork is identified, or a verval statement is given, set ad (advance directive) to a concise value like 'DNR present'"
    # "Return ONLY JSON, no text."

# ---- minimal, role-conditioned system prompt (short, cheap) ----
def _system_prompt(role: str) -> str:
    if role == "EMS":
        allow = ", ".join(sorted(ALLOW_KEYS["EMS"]))
        return (
            "You are a medical assistant that extracts patient data from EMS speech during emergency responses."
            "The line passed to you is spoken by EMS personel and is part of a conversation."
            "You are to search the line for any useful data that can be used to populate a post incident report."
            f"The following shares the fields that are available to be filled and their short keys: {allow}."
            "The fields all pertain to the patient. For example adesc corresponds to the patients physical description(clothing, hair, marks), loi corresponds to the last thing the patient ate or drank, etc."
            "Meds correspond to medications EMS has clearly administered to patients. ivx are procedures EMS has taken, i.e. CPR."
            "Output ONLY a JSON object of updates using SHORT KEYS."
            "Lines can include multiple useful items—extract all that apply in one JSON. For example, 'Adult male on the ground, bleeding from the scalp' should extract sex and inj."
            "If the line has no useful data for the report, output {}.\n"
        )
    else:
        allow = ", ".join(sorted(ALLOW_KEYS["other"]))
        return (
            "You are a medical assistant that extracts patient data during emergency responses."
            "The line passed to you is spoken by either a current EMS patient or a witness who has additional info about the ermergency response."
            "You are to search the line for any useful data that can be used to populate a post incident report."
            f"The following shares the fields that are available to be filled and their short keys: {allow}."
            "The fields all pertain to the patient. For example adesc corresponds to the patients physical description(clothing, hair, marks), loi corresponds to the last thing the patient ate or drank, etc."
            "ivx are procedures/actions that the patient or a witness has performed, i.e. EpiPen injection, CPR."
            "Output ONLY a JSON object of updates using SHORT KEYS."
            "Lines can include multiple useful items—extract all that apply in one JSON. For example, 'Adult male on the ground, bleeding from the scalp' should extract sex and inj."
            "If the line has no useful data for the report, output {}.\n"
        )

    # "You extract structured witness/patient info from ONE line of transcript.\n"
    # "ROLE=other. Output ONLY a JSON object of updates using SHORT KEYS. "
    # "If the line has no useful data, output {}.\n"
    # f"Allowed short keys: {allow}. "
    # "Interventions here are witness/self-aid only. Example: 'used EpiPen' "
    # "Lines can include multiple useful items—extract all that apply in one JSON. Example: input: “His name is John Miller and he is wearing a blue jacket.' Output: {'nm':'John Miller','adesc':['blue jacket']}"
    # "If any eating/drinking is stated (had coffee at 8 am), output to loi."
    # "If a witness mentions vitals casually (e.g., ‘blood pressure around 110 over 70 earlier’), output a short paraphrase to wv."
    # "Do not guess blood_type. Only fill if explicitly stated like ‘Type O positive’."
    # "scene_info = environment/location only; injury = bodily harm. Do not cross-fill."
    # "alt_description = visible descriptors only (clothing, hair, marks). Never put actions, locations, injuries, or scene details here."
    # "If patient or bystander provides a verbal statement pertaining to advanced directive, add a consise value to short key 'ad'."
    # "Return ONLY JSON, no text."














# ---- tiny validation helpers (server-side guardrails) ----
_BP_RE = re.compile(r"^\d{2,3}/\d{2,3}$")

def _validate_and_expand(role: str, time_hhmmss: str, updates_short: Dict[str, Any]) -> Dict[str, Any]:
    """Drop disallowed keys, enforce types/units, map to full names, add timeline_events for EMS vitals."""
    allowed = ALLOW_KEYS[role]
    cleaned_short: Dict[str, Any] = {}

    # Keys that should always be lists (after validation/normalization)
    LIST_KEYS = {
        "bp","hr","rr","spo2","glu","gcs",      # vitals
        "inj","sx","ivx","meds", "wv",                # clinical/actions
        "adesc", "scene","behavior","alg","hx"   # context
    }

    for k, v in (updates_short or {}).items():
        if k not in allowed:
            continue

        # --- Basic type normalization/validation ---
        if k == "bp":
            vals = v if isinstance(v, list) else [v]
            vals_ok = [s for s in vals if isinstance(s, str) and _BP_RE.match(s)]
            if vals_ok:
                cleaned_short[k] = vals_ok

        elif k in {"hr", "rr", "spo2", "glu"}:
            vals = v if isinstance(v, list) else [v]
            out: List[int] = []
            for x in vals:
                try:
                    n = int(x)
                    if k == "spo2" and not (0 <= n <= 100):
                        continue
                    out.append(n)
                except Exception:
                    continue
            if out:
                cleaned_short[k] = out

        elif k == "gcs":
            vals = v if isinstance(v, list) else [v]
            out = [str(x) for x in vals if isinstance(x, (str, int))]
            if out:
                cleaned_short[k] = out

        else:
            # For generic list-valued fields: coerce scalar -> single-element list
            if k in LIST_KEYS:
                cleaned_short[k] = v if isinstance(v, list) else [v]
            else:
                cleaned_short[k] = v

    # EMS timeline rule: if any vital accepted, add te=[time]
    if role == "EMS" and any(k in cleaned_short for k in ("bp", "hr", "rr", "spo2", "glu", "gcs")):
        cleaned_short["te"] = [time_hhmmss]

    # Map short -> full names
    expanded: Dict[str, Any] = {}
    for sk, val in cleaned_short.items():
        full = SHORT_KEY_MAP.get(sk)
        if full:
            expanded[full] = val

    return expanded


# ---- main entry (single-agent, role-conditioned, no profile context) ----
def extract_patient_info(role: str, time_hhmmss: str, text: str) -> Dict[str, Any]:
    """
    Inputs:
      role: 'EMS' or 'other'
      time_hhmmss: 'HH:MM:SS' (adapter time)
      text: raw line text (single line)
    Output:
      dict of updates using FULL field names (already validated & role-filtered).
      Returns {} if nothing to update.
    """
    role = "EMS" if role.upper() == "EMS" else "other"
    if not (text or "").strip():
        return {}

    messages = [
        {"role": "system", "content": _system_prompt(role)},
        {"role": "user", "content": text.strip()},
    ]

    try:
        resp = openai.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=messages,
            max_tokens=180  # tiny JSON only
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Ensure we only parse JSON
        updates_short = json.loads(raw) if raw else {}
    except Exception as e:
        # Fail-safe: abstain on any LLM/JSON error
        # print(f"Extractor LLM error: {e}")
        return {}

    # Server-side allow-list + unit/type checks + timeline rule + key expansion
    return _validate_and_expand(role, time_hhmmss, updates_short)
