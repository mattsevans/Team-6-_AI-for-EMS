from datetime import datetime
from typing import Dict, Any, List
from typing import Optional, Callable

patwit_profile = {
    "name": None,
    "alt_description": [],

    "age": None,
    "sex": None,
    "weight": None,
    "blood_type": None,
    "date_of_birth": None,
    "social_security_number": None,
    "address": None,
    "patient_contact_info": None,
    "next_of_kin_name": None,
    "next_of_kin_phone_number": None,

    "injury": [],
    "symptoms": [],
    "airway_status": None,
    "mental_status": None,
    "patient_behavior": [],
    "witness_vitals": [],

    "scene_info": [],
    "allergies": [],
    "medical_history": [],
    "last_oral_intake": None,
    "advance_directive": None,

    "interventions": []
}

ems_profile = {
    "name": None,
    "alt_description": [],

    "age": None,
    "sex": None,
    "weight": None,
    "blood_type": None,
    "date_of_birth": None,
    "social_security_number": None,
    "address": None,
    "patient_contact_info": None,
    "next_of_kin_name": None,
    "next_of_kin_phone_number": None,

    "blood_pressure": [],
    "heart_rate": [],
    "respiratory_rate": [],
    "oxygen_saturation": [],
    "glucose_level": [],
    "gcs_score": [],

    "injury": [],
    "symptoms": [],
    "meds": [],
    "interventions": [],
    "airway_status": None,

    "scene_info": [],

    "patient_behavior": [],

    "transport_destination": None,
    "transport_mode": None,
    "transport_time_pickup": None,
    "transport_time_dropoff": None,

    "timeline_events": [],

    "allergies": [],
    "medical_history": [],
    "last_oral_intake": None,
    "advance_directive": None,
    "mental_status": None,

    "vitals_log": [],
    "meds_log": []
}


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

# --------------------- Change-log helper  --------------------
def _log_change(ems_profile: dict, *, t: str, field: str, op: str, value):
    """
    Append a reversible change to the EMS profile's unified _change_log.
    Each entry captures enough info to roll back later.

    t     : "HH:MM:SS"
    field : which EMS field was touched (e.g., "blood_pressure", "interventions", "meds")
    op    : "append" (list add), "set" (scalar overwrite), or "log" (entry added to a *_log)
    value : the new value appended/set/logged (store shallow copy)
    """
    ems_profile.setdefault("_change_log", []).append({
        "t": t,
        "field": field,
        "op": op,
        "value": value,
    })

# ---- Rollback API expected by edge_loop.Deps ----
def rollback_last_entry(ems_profile: dict) -> str:
    """
    Roll back the most recent EMS-side change, preserving field shapes:
      - list fields -> [] when cleared
      - dict fields -> {} when cleared
      - scalars     -> None when cleared
    Priority 1: unified _change_log; Priority 2: meds_log; Priority 3: vitals_log.
    Returns a short human-readable status string.
    """
    VITAL_KEYS = ("blood_pressure", "heart_rate", "respiratory_rate",
                  "oxygen_saturation", "glucose_level", "gcs_score")

    def _clear_preserving_shape(field: str, logged_value):
        if isinstance(logged_value, list):
            ems_profile[field] = []
        elif isinstance(logged_value, dict):
            ems_profile[field] = {}
        else:
            ems_profile[field] = None

    # -------- Preferred: unified change log --------
    chg = (ems_profile.get("_change_log") or [])
    if chg:
        last = chg.pop()
        field = last.get("field")
        op = last.get("op")
        val = last.get("value")
        t = last.get("t", "?")

        try:
            if op == "append":
                seq = ems_profile.get(field, [])
                if isinstance(seq, list) and seq:
                    if seq[-1] == val:
                        seq.pop()
                        return f"Rolled back last '{field}' append from {t}."
                    if val in seq:
                        seq.remove(val)
                        return f"Rolled back '{field}' (removed one match) from {t}."
                return f"Nothing to rollback for '{field}'."

            elif op == "set":
                cur = ems_profile.get(field)
                if cur == val:
                    _clear_preserving_shape(field, val)
                    return f"Cleared '{field}' set at {t}."
                return f"'{field}' changed again since {t}; not modified."

            elif op == "log":
                # Logs are append-only arrays like meds_log or vitals_log
                log_list = ems_profile.get(field, [])
                if isinstance(log_list, list) and log_list:
                    tail = log_list[-1]
                    # Pop log entry (prefer matching time, but pop anyway if older profiles lack time)
                    if isinstance(tail, dict) and tail.get("time") == t:
                        log_list.pop()
                    else:
                        log_list.pop()

                    # ALSO undo side-effects on main fields when we know them
                    if field == "vitals_log" and isinstance(val, dict):
                        for k in VITAL_KEYS:
                            if k in val:
                                seq = ems_profile.get(k)
                                if isinstance(seq, list):
                                    if seq:
                                        # Prefer removing the exact value we logged; else remove last
                                        if seq[-1] == val[k]:
                                            seq.pop()
                                        else:
                                            if val[k] in seq:
                                                seq.remove(val[k])
                                            elif seq:
                                                seq.pop()
                                        if not seq:
                                            ems_profile[k] = []
                    elif field == "meds_log" and isinstance(val, dict):
                        med_name = val.get("medication") or val.get("med") or val.get("name")
                        meds_list = ems_profile.get("meds")
                        if isinstance(meds_list, list):
                            if med_name in meds_list:
                                meds_list.remove(med_name)
                            if not meds_list:
                                ems_profile["meds"] = []

                    return f"Rolled back last '{field}' entry at {t}."

                return f"No entries to rollback in '{field}'."

            return "Nothing to rollback (unknown change op)."

        except Exception as e:
            return f"Rollback error on '{field}': {e}"

    # -------- Fallback: meds_log -> vitals_log --------
    meds_log = ems_profile.get("meds_log", [])
    if isinstance(meds_log, list) and meds_log:
        removed = meds_log.pop()
        med_name = removed.get("medication") or removed.get("med") or removed.get("name")
        meds_list = ems_profile.get("meds")
        if isinstance(meds_list, list):
            if med_name in meds_list:
                meds_list.remove(med_name)
            if not meds_list:
                ems_profile["meds"] = []
        t = removed.get("time", "?")
        return f"Rolled back medication logged at {t}."

    vitals_log = ems_profile.get("vitals_log", [])
    if isinstance(vitals_log, list) and vitals_log:
        removed = vitals_log.pop()
        for fld in VITAL_KEYS:
            seq = ems_profile.get(fld)
            if isinstance(seq, list):
                if seq:
                    seq.pop()
                if not seq:
                    ems_profile[fld] = []
        t = removed.get("time", "?")
        return f"Rolled back vitals logged at {t}."

    return "Nothing to rollback."

# ----- Original Helpers ------

def _extend_unique(dst: List[Any], src: List[Any]) -> None:
    """
    Append items from source to destination if not already present (exact-match de-dup).
    Keeps original order for existing items.
    """
    seen = set(dst)
    for x in src:
        if x not in seen:
            dst.append(x)
            seen.add(x)

def _merge_into(target: Dict[str, Any],
                updates: Dict[str, Any],
                list_dedupe: bool = True,
                on_change: Optional[Callable[[str, str, Any], None]] = None) -> None:
    """
    Generic merge into a target dict:
      - lists: extend (optionally de-dup),
      - dicts: shallow update,
      - scalars: overwrite.

    on_change (optional):
      A callback invoked only when the target actually changes.
      Signature: on_change(field: str, op: str, value: Any)
        - op == "append" for list item appends
        - op == "set"    for scalar/dict replacements or updates
    """
    for k, v in updates.items():
        cur = target.get(k, None)

        # ---------- List handling ----------
        if isinstance(v, list):
            if isinstance(cur, list):
                # Append items; optionally de-dup. Fire on_change per appended item.
                if list_dedupe:
                    for item in v:
                        if item not in cur:
                            cur.append(item)
                            if on_change:
                                on_change(k, "append", item)
                else:
                    for item in v:
                        cur.append(item)
                        if on_change:
                            on_change(k, "append", item)
            else:
                # Replace non-list (or missing) with a new list copy
                target[k] = list(v)
                if on_change:
                    # Treat as a set since we replaced the whole field with a list
                    on_change(k, "set", target[k])

        # ---------- Dict handling ----------
        elif isinstance(v, dict):
            if isinstance(cur, dict):
                # Shallow merge; only fire if anything actually changed
                changed = False
                for dk, dv in v.items():
                    if cur.get(dk) != dv:
                        changed = True
                        cur[dk] = dv
                if changed and on_change:
                    on_change(k, "set", cur)
            else:
                # Replace non-dict (or missing) with a dict copy
                target[k] = dict(v)
                if on_change:
                    on_change(k, "set", target[k])

        # ---------- Scalar (or other types) ----------
        else:
            if cur != v:
                target[k] = v
                if on_change:
                    on_change(k, "set", v)
            # else: no actual change → no callback


# -------------------------------------------------------------------------
# Main merge function (EMS + PatWit aware)
# -------------------------------------------------------------------------

def smart_update_profiles(
    ems_profile: Dict[str, Any],
    patwit_profile: Dict[str, Any],
    role: str,
    updates: Dict[str, Any],
    event_time_hhmmss: str,
) -> None:
    """
    Merge extractor updates into the correct profile (EMS or PatWit), and
    maintain EMS logs with accurate per-line timestamps.

    Parameters
    ----------
    ems_profile : dict
        The mutable EMS profile dict to be updated in place.
    patwit_profile : dict
        The mutable PatWit profile dict to be updated in place.
    role : str
        The normalized role from the adapter: "EMS" or "other".
    updates : dict
        Field updates produced by the extractor (already using FULL field names).
        Example: {"blood_pressure": ["120/80"], "interventions": ["IV cannulation"]}
    event_time_hhmmss : str
        The exact [HH:MM:SS] from the adapter for THIS line. This is used as
        the authoritative timestamp for logs (vitals_log, meds_log).

    Behavior
    --------
    - Routes updates by role:
        * EMS  -> ems_profile
        * other -> patwit_profile
    - Lists extend (de-dup), dicts shallow-merge, scalars overwrite.
    - EMS-only logging:
        * If THIS update contains any vitals (bp/hr/rr/spo2/glu/gcs),
          append ONE entry to ems_profile['vitals_log'] with those values and
          time = event_time_hhmmss (adapter is authoritative).
        * If THIS update contains meds, append one entry per medication to
          ems_profile['meds_log'] with time = event_time_hhmmss.
    - PatWit (role "other"):
        * Will not store true vitals (bp/hr/rr/spo2/glu/gcs) nor meds;
          those are ignored here.
        * Keeps 'witness_vitals' (list of short strings) if provided.

    Notes
    -----
    - We intentionally DROP any 'timeline_events' field to avoid redundancy.
      The adapter timestamp is the single source of truth.
    """

    # Choose merge target based on role
    target = ems_profile if (role or "").upper() == "EMS" else patwit_profile

    # Key groups for behavior control
    VITAL_KEYS = {"blood_pressure", "heart_rate", "respiratory_rate",
                  "oxygen_saturation", "glucose_level", "gcs_score"}

    # Fields that are list-like and should be coerced to list when merging
    LIST_FIELDS_ALWAYS = {
        # vitals (EMS profile)
        "blood_pressure", "heart_rate", "respiratory_rate", "oxygen_saturation", "glucose_level", "gcs_score",
        # shared clinical/context lists
        "injury", "symptoms", "interventions", "scene_info", "patient_behavior", "allergies", "medical_history",
        "alt_description",
        # PatWit-only list for casual mentions of vitals
        "witness_vitals",
        # meds list (EMS profile)
        "meds",
    }

    # ---------------------------------------------------------------------
    # 0) Defensive filtering
    #    - Drop any 'timeline_events' to avoid redundancy with adapter time.
    #    - For PatWit: do NOT allow true vitals or meds to land in the profile.
    #      (Keep 'witness_vitals' intact.)
    # ---------------------------------------------------------------------
    updates = {k: v for k, v in (updates or {}).items() if k != "timeline_events"}

    if target is patwit_profile:
        updates = {
            k: v for k, v in updates.items()
            if k not in VITAL_KEYS and k != "meds"
        }

    # ---------------------------------------------------------------------
    # 1) Normalize list shapes for known list fields to simplify merging
    # ---------------------------------------------------------------------
    normalized_updates: Dict[str, Any] = {}
    for k, v in updates.items():
        if k in LIST_FIELDS_ALWAYS:
            if v is None:
                normalized_updates[k] = []
            elif isinstance(v, list):
                normalized_updates[k] = v
            else:
                normalized_updates[k] = [v]
        else:
            normalized_updates[k] = v

    # ---------------------------------------------------------------------
    # 2) Merge normalized updates into target profile (de-dup lists)
    # ---------------------------------------------------------------------
    if target is ems_profile:
        # For EMS: log field-level list appends and scalar/dict sets
        def _on_change(field: str, op: str, value: Any) -> None:
            _log_change(ems_profile, t=event_time_hhmmss, field=field, op=op, value=value)
        _merge_into(target, normalized_updates, list_dedupe=True, on_change=_on_change)
    else:
        # For PatWit: no change-log (rollback only applies to EMS)
        _merge_into(target, normalized_updates, list_dedupe=True, on_change=None)
    # ---------------------------------------------------------------------
    # 3) EMS-only logs: vitals_log and meds_log with adapter line time
    # ---------------------------------------------------------------------
    if target is ems_profile:
        # 3a) If THIS update has any vitals, append ONE unified snapshot
        if any(k in normalized_updates for k in VITAL_KEYS):
            vitals_entry: Dict[str, Any] = {"time": event_time_hhmmss}  # adapter time is authoritative
            for k in VITAL_KEYS:
                if k in normalized_updates:
                    val = normalized_updates[k]
                    vitals_entry[k] = (val[-1] if isinstance(val, list) and val else val)
            ems_profile.setdefault("vitals_log", []).append(vitals_entry)
            _log_change(ems_profile, t=event_time_hhmmss, field="vitals_log", op="log", value=dict(vitals_entry))

        # 3b) If THIS update has meds, append one entry per medication with the adapter line time
        if "meds" in normalized_updates and normalized_updates["meds"]:
            meds_list: List[Any] = normalized_updates["meds"] if isinstance(normalized_updates["meds"], list) else [normalized_updates["meds"]]
            for med in meds_list:
                entry = {"time": event_time_hhmmss, "medication": med}
                ems_profile.setdefault("meds_log", []).append(entry)
                _log_change(ems_profile, t=event_time_hhmmss, field="meds_log", op="log", value=dict(entry))

    # Done: profiles are modified in place.
