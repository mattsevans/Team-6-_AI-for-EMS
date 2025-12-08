# edge_input_adapter.py
# ------------------------------------------------------------
# Purpose:
#   Normalize incoming lines in the strict format:
#       [HH:MM:SS] ROLE: text
#
#   - Extract timestamp from the required leading [HH:MM:SS].
#   - Extract role and text (split at the first colon).
#   - Collapse roles to two buckets: 'EMS' or 'other' (only explicit 'EMS' → 'EMS').
#   - Detect Zora commands with get_zora_prompt, but ONLY allow commands from EMS.
#   - Generate incident_id from the FIRST event timestamp:
#       incident_YYYYMMDD_HHMMSS
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict
import re

# ------------------ Incident ID manager ---------------------

class IncidentIdManager:
    """
    Holds a single incident_id for the current run/session.
    First call to ensure_from_timestamp(ts) sets:
      incident_YYYYMMDD_HHMMSS  (based on that timestamp)
    """
    def __init__(self):
        self._incident_id: Optional[str] = None

    @property
    def value(self) -> Optional[str]:
        return self._incident_id

    def ensure_from_timestamp(self, ts: datetime) -> str:
        if self._incident_id is None:
            self._incident_id = ts.strftime("incident_%Y%m%d_%H%M%S")
            print(f"Incident ID: {self._incident_id}")
        return self._incident_id


# ------------------ Normalized event ------------------------

@dataclass
class NormalizedEvent:
    """
    A uniform representation of one incoming line.
    - role: final role label ('EMS' or 'other')
    - text: message content (without role or timestamp)
    - timestamp: datetime for the line (parsed from [HH:MM:SS])
    - raw_input: original line for auditing
    - zora_prompt: string after 'zora' if present AND role == 'EMS'; else None
    """
    role: str
    text: str
    timestamp: datetime
    raw_input: str
    zora_prompt: Optional[str] = None


# ------------------ Role resolver ---------------------------

class RoleResolver:
    """
    Resolves roles into two buckets: 'EMS' or 'other'.
    Simple rule: only explicit 'EMS' (case-insensitive) → 'EMS'; everything else → 'other'.
    """
    def __init__(self, default_session_key: str = "session-1"):
        self._session_key = default_session_key
        self._cache: Dict[str, Dict[str, str]] = {}  # kept in case you reintroduce mapping later

    def set_session(self, session_key: str) -> None:
        self._session_key = session_key

    def resolve(self, role: str, text: str) -> str:
        """
        Returns:
            'EMS' if raw role is exactly 'EMS' (case-insensitive), else 'other'.

        Notes:
            - 'text' is currently unused but kept for easy future extension.
        """
        if (role or "").strip().upper() == "EMS":
            return "EMS"
        return "other"


# ------------------ Zora command detection ------------------

def get_zora_prompt(text: str) -> Optional[str]:
    """
    Behavior:
    - Find 'zora' anywhere (case-insensitive).
    - Return everything AFTER 'zora' (trim simple punctuation/space).
    - If not present, return None.
    """
    lowered = text.lower()
    if "zora" in lowered:
        idx = lowered.find("zora")
        return text[idx + len("zora"):].strip(" ,.:")
    return None


# ------------------ Strict parsing helpers ------------------

def _parse_bracketed_hms_required(raw_line: str) -> tuple[datetime, str]:
    """
    Expect a required prefix: [HH:MM:SS]
    Returns (timestamp, remainder_after_bracket).
    Raises ValueError if the required format is not present.
    """
    s = raw_line.strip()
    if not (s.startswith("[") and ("]" in s)):
        raise ValueError("Input must start with [HH:MM:SS]")

    closing = s.find("]")
    inside = s[1:closing].strip()
    h, m, sec = inside.split(":")  # will raise if not exactly 3 parts
    # Anchor at today's date, set time from the bracket
    today = datetime.now().replace(microsecond=0)
    ts = today.replace(hour=0, minute=0, second=0) + timedelta(
        hours=int(h), minutes=int(m), seconds=int(sec)
    )
    remainder = s[closing + 1:].strip()
    return ts, remainder


def _split_role_and_text_required(s: str) -> tuple[str, str]:
    """
    Split on the FIRST colon to get 'ROLE: text'.
    If no colon exists, treat as invalid (raise ValueError) to keep behavior strict.
    """
    if ":" not in s:
        raise ValueError("Input must contain 'ROLE: text' after the timestamp.")
    role, text = s.split(":", 1)
    return role.strip(), text.strip()

# ------------------ Text pre-normalization (edge) ------------------

# Basic number words up to 90; "hundred" supported for 3–4 word combos.
_NUM_WORDS = {
    "zero":0, "oh":0, "one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9,
    "ten":10, "eleven":11, "twelve":12, "thirteen":13, "fourteen":14, "fifteen":15, "sixteen":16,
    "seventeen":17, "eighteen":18, "nineteen":19,
    "twenty":20, "thirty":30, "forty":40, "fifty":50, "sixty":60, "seventy":70, "eighty":80, "ninety":90,
    "hundred":100
}
_IGNORE_TOKENS = {"and"}  # allow 'and' inside sequences

# Helpers to collapse whitespace
def _collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _words_to_int(tokens: list[str]) -> int | None:
    """
    Convert small sequences of number words to an int (conservative).
    'oh' counts as 0 only when sandwiched between number words (e.g., 'one oh two' -> 102).
    """
    if not tokens:
        return None
    if any(t not in _NUM_WORDS and t not in _IGNORE_TOKENS and t != "-" for t in tokens):
        return None

    # Normalize hyphenation and drop 'and'
    flat: list[str] = []
    for t in tokens:
        if t in _IGNORE_TOKENS or t == "-":
            continue
        if "-" in t:
            parts = [p for p in t.split("-") if p]
            if any(p not in _NUM_WORDS for p in parts):
                return None
            flat.extend(parts)
        else:
            flat.append(t)

    if not flat:
        return None

    # Enforce 'oh' only when sandwiched (general case)
    if "oh" in flat:
        idxs = [i for i, w in enumerate(flat) if w == "oh"]
        for i in idxs:
            if i == 0 or i == len(flat) - 1:
                return None  # 'oh' leading or trailing -> not a number
            if flat[i-1] not in _NUM_WORDS or flat[i+1] not in _NUM_WORDS:
                return None

    # A) Explicit 'hundred'
    if "hundred" in flat:
        try:
            i = flat.index("hundred")
        except ValueError:
            return None
        if i == 0:
            return None
        left = _NUM_WORDS.get(flat[i-1], None)
        if left is None or left == 0:
            return None
        right = sum(_NUM_WORDS.get(x, 0) for x in flat[i+1:])
        return left * 100 + right

    # B) EMS shorthand: 'one sixteen' -> 116 ; 'one oh two' -> 102 ; 'one twenty four' -> 124
    if flat[0] == "one" and len(flat) >= 2:
        second = _NUM_WORDS.get(flat[1], None)
        if second is not None:
            # if 'oh', require a tail digit to be present (i.e., 'one oh two')
            if flat[1] == "oh" and len(flat) < 3:
                return None
            tail = sum(_NUM_WORDS.get(x, 0) for x in flat[2:]) if len(flat) >= 3 else 0
            return 100 + second + tail

    # C) 'three twenty four' -> 324 ; include teens as second part
    if len(flat) >= 2 and 2 <= _NUM_WORDS.get(flat[0], -1) <= 9:
        tens = _NUM_WORDS.get(flat[1], -1)
        if tens in (20,30,40,50,60,70,80,90) or 10 <= tens <= 19:
            ones = _NUM_WORDS.get(flat[2], 0) if len(flat) >= 3 else 0
            return _NUM_WORDS[flat[0]] * 100 + tens + ones

    # D) Plain sums like 'twenty two'
    total = 0
    for w in flat:
        val = _NUM_WORDS.get(w)
        if val is None:
            return None
        total += val
    return total

def _replace_number_word_sequences(s: str) -> str:
    """
    Replace short sequences of number words with digits.
    Conservative: up to 4 tokens, only number words/hyphens.
    """
    def repl(m: re.Match) -> str:
        seq = m.group(0).lower()
        toks = [t for t in re.findall(r"[a-z]+|-", seq)]
        val = _words_to_int(toks)
        return str(val) if val is not None else seq

    # Match runs of up to 4 tokens composed of number words / hyphens
    pattern = r"\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)(?:-|\s+)){0,3}(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b"
    return re.sub(pattern, repl, s, flags=re.IGNORECASE)

def _conditional_replace_number_word_sequences(s: str) -> str:
    # Only apply in vitals/clinical contexts; skip birthdays, phones, etc.
    if re.search(
        r"\b("
        r"heart\s*rate|hr|pulse|pr|"
        r"blood\s*pressure|bp|"
        r"resp(iratory)?\s*rate|rr|"
        r"spo2|o2|oxygen|"
        r"glucose|blood\s*sugar|"
        r"temperature|temp|t(core)?|"
        r"etco2|end[-\s]*tidal"
        r")\b",
        s, flags=re.IGNORECASE
    ):
        return _replace_number_word_sequences(s)
    return s

# Local matcher for short number-word sequences (context-only; includes oh|and)
_LOCAL_NUMWORD_RE = re.compile(
    r"(?:(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|and)(?:-|\s+)){0,3}"
    r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety)"
    r"", re.IGNORECASE
)

def _parse_local_numwords(seq: str) -> int | None:
    toks = re.findall(r"[a-z]+|-", seq.lower())
    return _words_to_int(toks)  # uses your “oh only when sandwiched” logic

def _normalize_bp_words(s: str) -> str:
    # First: word-form BP like "one oh two over sixty six"
    def repl_words(m: re.Match) -> str:
        sys_w, dia_w = m.group(1), m.group(2)
        sys_v = _parse_local_numwords(sys_w)
        dia_v = _parse_local_numwords(dia_w)
        if sys_v is not None and dia_v is not None:
            return f"{sys_v}/{dia_v}"
        return m.group(0)

    s = re.sub(
        rf"\b({_LOCAL_NUMWORD_RE.pattern})\s+over\s+({_LOCAL_NUMWORD_RE.pattern})\b",
        repl_words, s, flags=re.IGNORECASE
    )

    # Second: already-digit form "128 over 76" → "128/76"
    def repl_digits(m: re.Match) -> str:
        return f"{int(m.group(1))}/{int(m.group(2))}"
    return re.sub(r"\b(\d{2,3})\s+over\s+(\d{2,3})\b", repl_digits, s, flags=re.IGNORECASE)

# Accept "heart rate …" / "hr …" (you can add aliases like pulse/pr if you want)
_HR_PREFIX = r"(?:heart\s*rate|hr|pulse|pr)"

def _normalize_hr_words(s: str) -> str:
    def repl(m: re.Match) -> str:
        prefix, seq = m.group(1), m.group(2)
        val = _parse_local_numwords(seq)
        return f"{prefix} {val}" if val is not None else m.group(0)

    # Examples: "Heart rate one sixteen", "HR is one oh two"
    return re.sub(
        rf"\b({_HR_PREFIX})\s*(?:is\s*)?({_LOCAL_NUMWORD_RE.pattern})\b",
        repl, s, flags=re.IGNORECASE
    )


# --- Time word handling (strict prefixes) ---
_TIME_WORD = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty)"
_TIME_CHUNK = rf"(?:{_TIME_WORD})(?:\s+{_TIME_WORD})?"
_TIME_PAIR  = rf"{_TIME_CHUNK}\s+{_TIME_CHUNK}"

# Only convert when preceded by these phrases
_TIME_PREFIX = r"(?:pickup\s+time|eta|etd|arrival\s*time|time\s+of\s+pickup)"

def _parse_time_chunk(tokens: list[str]) -> int | None:
    """Turn 1–2 number words into an integer 0..59. In time context, 'oh five' -> 5 is allowed."""
    if not tokens:
        return None
    flat: list[str] = []
    for t in tokens:
        flat.extend([p for p in t.split("-") if p])  # allow hyphenated forms
    vals = []
    for w in flat:
        if w not in _NUM_WORDS and w not in _IGNORE_TOKENS:
            return None
        if w in _IGNORE_TOKENS:
            continue
        vals.append(_NUM_WORDS.get(w, 0))
    if not vals:
        return None
    n = sum(vals)
    return n if 0 <= n <= 59 else None

def _normalize_time_word_pairs(s: str) -> str:
    """
    'pickup time is zero one forty four' -> 'pickup time 01:44'
    Run BEFORE global number-word replacement.
    """
    def repl(m: re.Match) -> str:
        head = m.group(1)        # includes the prefix + trailing spaces
        words = m.group(2).lower().split()
        for h_len in (2, 1):
            if h_len <= len(words) - 1:
                hh = _parse_time_chunk(words[:h_len])
                mm = _parse_time_chunk(words[h_len:])
                if hh is not None and mm is not None and 0 <= hh <= 23 and 0 <= mm <= 59:
                    return f"{head}{hh:02d}:{mm:02d}"
        return m.group(0)

    # Allow optional 'is ' between prefix and the two word-number chunks
    pattern = rf"\b((?:{_TIME_PREFIX})\s+(?:is\s+)?)({_TIME_PAIR})\b"
    return re.sub(pattern, repl, s, flags=re.IGNORECASE)

def _collapse_digit_time_pair(s: str) -> str:
    # Also allow optional 'is '
    return re.sub(
        rf"\b((?:{_TIME_PREFIX})\s+(?:is\s+)?)"
        r"(\d{1,2})\s+(\d{1,2})\b",
        lambda m: f"{m.group(1)}{int(m.group(2)):02d}:{int(m.group(3)):02d}",
        s, flags=re.IGNORECASE
    )

def _colonize_four_digit_time(s: str) -> str:
    # Also allow optional 'is '
    def repl(m: re.Match) -> str:
        head, blob = m.group(1), m.group(2)
        if len(blob) == 3:
            hh, mm = int(blob[0]), int(blob[1:])
        else:  # len == 4
            hh, mm = int(blob[:2]), int(blob[2:])
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{head}{hh:02d}:{mm:02d}"
        return m.group(0)
    return re.sub(
        rf"\b((?:{_TIME_PREFIX})\s+(?:is\s+)?)"
        r"(\d{3,4})\b",
        repl, s, flags=re.IGNORECASE
    )

def _normalize_percent_phrases(s: str) -> str:
    # "... eighty-nine percent" -> "89%"
    def repl(m: re.Match) -> str:
        num = m.group(1)
        # If it's digits already, keep; else convert number words we just normalized above
        try:
            return f"{int(num)}%"
        except ValueError:
            return f"{num}%"
    return re.sub(r"\b(\d{1,3})\s*percent\b", repl, s, flags=re.IGNORECASE)

def _normalize_o2_flow(s: str) -> str:
    # "fifteen liters per minute" (after number normalization) -> "15 L/min"
    return re.sub(
        r"\b(\d{1,2})\s+(?:liter|liters)\s+per\s+minute\b",
        r"\1 L/min",
        s,
        flags=re.IGNORECASE
    )


def _pre_normalize_text(text: str) -> str:
    """
    Safe, role-agnostic normalization for the cloud extractor:
      - collapse whitespace,
      - numbers→digits ("one hundred four" -> 104; "three twenty-four" -> 324),
      - percent phrases ("eighty-nine percent" -> "89%"),
      - BP words ("one twenty-eight over seventy-six" -> "128/76"),
      - O2 flow ("fifteen liters per minute" -> "15 L/min").
    Returns the normalized text string.
    """
    if not text:
        return text

    s = text

    # Time words first → HH:MM
    s = _normalize_time_word_pairs(s)
    s = _collapse_digit_time_pair(s)
    s = _colonize_four_digit_time(s)

    # Context-only domain normalizers that understand oh/and locally
    s = _normalize_hr_words(s)          # HR word-forms → digits
    s = _normalize_bp_words(s)          # BP word-forms/digits → 102/66

    # Generic number words (unchanged, no oh/and globally)
    s = _conditional_replace_number_word_sequences(s)

    # Other domain tweaks
    s = _normalize_percent_phrases(s)
    s = _normalize_o2_flow(s)
    
    # Final tidy
    s = _collapse_spaces(s)
    return s

# ------------------ Normalization entry point ---------------

def normalize_incoming_message(
    raw_line: str,
    role_resolver: RoleResolver,
) -> NormalizedEvent:
    """
    Convert a raw input line into a NormalizedEvent, assuming strict format:
        [HH:MM:SS] ROLE: text

    - Collapse role to 'EMS' or 'other'.
    - Extract Zora command only if final role is EMS.
    """
    ts, remainder = _parse_bracketed_hms_required(raw_line)
    role, text = _split_role_and_text_required(remainder)
    
    norm_text = _pre_normalize_text(text)

    # Use normalized text for role resolution & zora detection
    final_role = role_resolver.resolve(role=role, text=norm_text)
    candidate_prompt = get_zora_prompt(norm_text)
    zora_prompt = candidate_prompt if (candidate_prompt and final_role.upper() == "EMS") else None

    return NormalizedEvent(
        role=final_role,
        text=norm_text,
        timestamp=ts,
        raw_input=raw_line,
        zora_prompt=zora_prompt,
    )


# ------------------ Orchestration helper --------------------

def process_incoming_line(
    raw_line: str,
    role_resolver: RoleResolver,
    incident_mgr: IncidentIdManager,
) -> tuple[NormalizedEvent, str]:
    """
    - Normalize the line.
    - Ensure/create incident_id from the FIRST event timestamp.
    - Align the resolver's session key to incident_id.
    - Return (event, incident_id).
    """
    event = normalize_incoming_message(raw_line, role_resolver)
    incident_id = incident_mgr.ensure_from_timestamp(event.timestamp)
    role_resolver.set_session(incident_id)
    return event, incident_id
