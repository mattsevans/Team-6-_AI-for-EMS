# ===================== post_whisper_translate.py =====================
from __future__ import annotations
import os, re, queue, threading, subprocess, tempfile, shutil, time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import importlib


# ---------- Config via env ----------
TRANSLATION_ALWAYS_TO_EN = os.environ.get("TRANSLATION_ALWAYS_TO_EN", "1").lower() in {"1","true","yes","on"}
TRANSLATION_MODE_DEFAULT  = os.environ.get("TRANSLATION_MODE_DEFAULT", "off").lower() in {"on","true","1"}
TRANSLATION_LIB           = os.environ.get("TRANSLATION_LIB", "argos")   # "argos" recommended on Pi
TRANSLATION_MODE_PHRASES_START = os.environ.get("TRANSLATION_MODE_PHRASES_START",
    "start translation mode;enable translation mode;translation on; translate; turn on translation")
TRANSLATION_MODE_PHRASES_STOP  = os.environ.get("TRANSLATION_MODE_PHRASES_STOP",
    "stop translation mode;disable translation mode;translation off;turn off translation")
ARGOS_DATA_DIR = os.environ.get("ARGOS_DATA_DIR", "/usr/share/argos-translate")
TTS_SPEED_WPM  = int(os.environ.get("TTS_SPEED_WPM", "170"))

# ---------- Utils ----------
def _default_safe_print(*a, **k):  # used if main doesn’t inject a logger
    try: print(*a, **k)
    except Exception: pass

ISO6391_WHISPER_EQUIV = {
    "zh-cn":"zh", "zh-tw":"zh",
}

def normalize_lang(code: Optional[str]) -> Optional[str]:
    if not code: return None
    c = code.strip().lower()
    c = ISO6391_WHISPER_EQUIV.get(c, c)
    return c[:2]

def is_english(code: Optional[str]) -> bool:
    return normalize_lang(code) == "en"

def normalize_text_for_cmd(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_phrase_set(csv_like: str) -> List[str]:
    return [normalize_text_for_cmd(x) for x in csv_like.split(";") if x.strip()]

# ---------- State machine ----------
@dataclass
class TranslationState:
    mode_on: bool = TRANSLATION_MODE_DEFAULT
    target_non_en_lang: Optional[str] = None
    last_non_en_ring: deque[str] = field(default_factory=lambda: deque(maxlen=3))

    def update_non_en(self, lang: Optional[str]):
        L = normalize_lang(lang)
        if L and L != "en":
            self.last_non_en_ring.appendleft(L)
            if not self.target_non_en_lang:
                self.target_non_en_lang = L

    def pick_other_lang(self) -> Optional[str]:
        return self.target_non_en_lang or (self.last_non_en_ring[0] if self.last_non_en_ring else None)

# ---------- Translator (Argos) ----------

class Translator:
    def __init__(self, impl: str = TRANSLATION_LIB, safe_print=_default_safe_print):
        self.impl = impl
        self.safe_print = safe_print
        self._cache_pair = None
        self._cache_model = None

        if impl == "argos":
            try:
                if ARGOS_DATA_DIR:
                    os.environ.setdefault("ARGOS_TRANSLATE_PACKAGE_DIR", ARGOS_DATA_DIR)
                self._argos_package  = importlib.import_module("argostranslate.package")
                self._argos_translate = importlib.import_module("argostranslate.translate")
                # optional on some versions:
                # self._argos_translate.load_installed_languages()
            except Exception as e:
                self.safe_print(f"[WARN] Argos Translate not available: {e}")
                self.impl = "none"

    def _find_argos_pair(self, src: str, dst: str):
        if not src or not dst:
            return None  # <-- add this early exit
        # ✅ use translate.get_installed_languages()
        langs = self._argos_translate.get_installed_languages()

        def _match(code: str):
            for cand in (code, code.split("-")[0]):
                m = next((L for L in langs if L.code == cand), None)
                if m: return m
            return None

        from_lang = _match(src)
        if not from_lang:
            return None
        dst_primary = dst.split("-")[0]
        to_lang = _match(dst)
        if not to_lang:
            return None

        # API pattern per docs: from_lang.get_translation(to_lang)
        return from_lang.get_translation(to_lang)

    def _ensure_pair(self, src: str, dst: str):
        if self._cache_pair == (src, dst) and self._cache_model is not None:
            return True
        tx = self._find_argos_pair(src, dst)
        if not tx:
            self._cache_pair, self._cache_model = None, None
            return False
        self._cache_pair, self._cache_model = (src, dst), tx
        return True

    def translate(self, text: str, src: str, dst: str) -> str:
        if not text or not src or src == dst or self.impl != "argos":
            return text
        try:
            if not self._ensure_pair(src, dst):
                self.safe_print(f"[WARN] No Argos model for {src}->{dst}.")
                return text
            return self._cache_model.translate(text)
        except Exception as e:
            self.safe_print(f"[WARN] Argos translate error ({src}->{dst}): {e}")
            return text


# ---------- TTS ----------
class TTSQueue:
    """
    Simple TTS queue that:
      - Uses espeak-ng to synthesize speech into a temporary WAV file.
      - Plays that WAV via the external C++ DAC player (dac_audio_play).
      - Works for any language espeak-ng supports (en, es, zh, ...).
    """

    def __init__(self, speed=TTS_SPEED_WPM, safe_print=_default_safe_print):
        self.speed = speed
        self.q: "queue.Queue[Tuple[str,str]]" = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.safe_print = safe_print
        self.th = threading.Thread(target=self._worker, name="tts-worker", daemon=True)
        self.th.start()

        # Path to the C++ DAC player; override via env if needed
        self.dac_player = os.environ.get(
            "DAC_PLAYER_PATH",
            os.path.join(os.path.dirname(__file__), "dac_audio_play"),
        )

    # ---------- public API ----------

    def speak(self, text: str, lang: str = "en"):
        """Enqueue text for asynchronous TTS."""
        if not text:
            return
        try:
            self.q.put_nowait((text, normalize_lang(lang) or "en"))
        except queue.Full:
            self.safe_print("[WARN] TTS queue full; dropping output.")

    def close(self):
        self.stop_event.set()
        self.th.join(timeout=0.5)

    def speak_blocking(self, text: str, lang: str = "en"):
        """
        Public, synchronous TTS using espeak-ng + DAC.
        Blocks until playback finishes. This lets callers pause mic capture while TTS plays.
        """
        if not text:
            return
        lang = normalize_lang(lang) or "en"
        try:
            self._speak_via_dac(text, lang)
        except Exception as e:
            self.safe_print(f"[WARN] TTS (blocking) exception: {e}")

    # ---------- internals ----------

    def _speak_via_dac(self, txt: str, lang: str):
        """
        1) Use espeak-ng to synthesize txt -> temporary WAV file.
        2) Call the C++ DAC player (dac_audio_play) to play that WAV via MCP4822.
        """
        voice_map = {
            "en": "en-us",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "it": "it",
            "pt": "pt",
            "ru": "ru",
            "zh": "zh",
        }
        v = voice_map.get(lang, "en")

        # 1) Synthesize to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = tf.name
           
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = tf.name

        try:
            espeak_cmd = [
                "espeak-ng",
                "-v", v,
                "-s", str(self.speed),
                "-w", wav_path,  # write to WAV file
                txt,
            ]
            rc1 = subprocess.run(
                espeak_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if rc1.returncode != 0:
                msg = rc1.stderr.decode(errors="ignore")[:160]
                self.safe_print(f"[WARN] espeak-ng failed: {msg}")
                return

            # ---------- Save a unique copy of the WAV for debugging ----------
            log_dir = "/home/team6/NLP-System/AI_Wear_NLP/tts_logs"
            os.makedirs(log_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            debug_name = f"tts_{timestamp}_{threading.get_ident()}.wav"
            debug_copy = os.path.join(log_dir, debug_name)

            try:
                shutil.copy2(wav_path, debug_copy)
                self.safe_print(f"[TTS] saved copy to {debug_copy}")
            except Exception as e:
                self.safe_print(f"[WARN] could not save TTS WAV copy: {e}")
            # -----------------------------------------------------------------

            # 2) Play the WAV via the C++ DAC player
            dac_cmd = [
                self.dac_player,
                "--wav", wav_path,
                "--bus", os.environ.get("DAC_SPI_BUS", "0"),
                "--ce", os.environ.get("DAC_SPI_CE", "1"),
                "--speed", os.environ.get("DAC_SPI_SPEED", "1000000"),
            ]
            rc2 = subprocess.run(
                dac_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if rc2.returncode != 0:
                msg = rc2.stderr.decode(errors="ignore")[:160]
                self.safe_print(f"[WARN] DAC player failed: {msg}")

        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass
           
            pass


    def _worker(self):
        """Background worker for async TTS (same engine as speak_blocking)."""
        while not self.stop_event.is_set():
            try:
                text, lang = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._speak_via_dac(text, lang)
            except Exception as e:
                self.safe_print(f"[WARN] TTS exception: {e}")
            finally:
                self.q.task_done()



# ---------- Results ----------
@dataclass
class PostResult:
    text_raw: str
    detected_lang: Optional[str]
    text_en: str
    text_other: Optional[str]          # EN->Other only when mode_on and other lang known
    other_lang: Optional[str]          # which “other” was used (if any)
    mode_event: Optional[str] = None   # "enabled"/"disabled" when toggled, else None

# ---------- Processor ----------
class PostWhisperProcessor:
    def __init__(
        self,
        state: TranslationState,
        tts: TTSQueue,
        translator: Translator,
        safe_print=_default_safe_print,
        get_lang_mode=None,   # <-- NEW: callback that returns 0 or 1
    ):
        self.state = state
        self.tts = tts
        self.tx = translator
        self.safe_print = safe_print
        self.get_lang_mode = get_lang_mode
        self._start_phrases = build_phrase_set(TRANSLATION_MODE_PHRASES_START)
        self._stop_phrases  = build_phrase_set(TRANSLATION_MODE_PHRASES_STOP)

    # Extract detected language from whisper.cpp JSON (if present)
    def extract_lang(self, data: Dict) -> Optional[str]:
        # Try common locations across whisper.cpp JSON variants
        lang = (
            data.get("language")
            or (data.get("result") or {}).get("language")
            or (data.get("params") or {}).get("language")
        )

        # Segment/transcription-level fallbacks
        if not lang:
            segs = data.get("segments") or data.get("transcription") or []
            if isinstance(segs, list) and segs:
                lang = (segs[0] or {}).get("language")

        return normalize_lang(lang)


    #def extract_lang(self, data: Dict) -> Optional[str]:
     #   print("[extract_lang] from", __file__)
      #  lang = data.get("language")
      #  if not lang:
      #      segs = data.get("segments") or data.get("transcription") or []
      #      if segs and isinstance(segs, list):
      #          lang = segs[0].get("language")
      #  return normalize_lang(lang)

    def _is_start(self, text_norm: str) -> bool:
        return text_norm in self._start_phrases

    def _is_stop(self, text_norm: str) -> bool:
        return text_norm in self._stop_phrases

    def compute_en(self, text: str, detected: Optional[str]) -> str:
        if TRANSLATION_ALWAYS_TO_EN:
            return text if is_english(detected) else (self.tx.translate(text, detected, "en") or text)
        # If not forced, still try to have English available:
        return text if is_english(detected) else (self.tx.translate(text, detected, "en") or text)

    def handle_segment(self, speaker_id: int, text: str, whisper_json: Dict) -> PostResult:
        raw = (text or "").strip()
        det = self.extract_lang(whisper_json)
        en  = self.compute_en(raw, det)

        # Remember last non-EN for “other”
        self.state.update_non_en(det)

        # ----------------------------------------------------------
        # Button-controlled translation mode via environment variable
        # master_launcher.py will keep os.environ["LANGUAGE_MODE"]
        # set to "0" or "1":
        #   "1" => translation mode ON
        #   "0" => translation mode OFF
        # ----------------------------------------------------------
        env_mode = os.environ.get("LANGUAGE_MODE")
        if env_mode == "1" and not self.state.mode_on:
            self.state.mode_on = True
            self.safe_print("[MODE] Translation mode ON (button/env).")
        elif env_mode == "0" and self.state.mode_on:
            self.state.mode_on = False
            self.safe_print("[MODE] Translation mode OFF (button/env).")

        # If LANGUAGE_MODE is not set, or set to something else,
        # we fall back to the old voice-command behavior.
        if env_mode not in ("0", "1"):
            # Speaker 1 can toggle the mode via exact normalized phrases
            text_norm = normalize_text_for_cmd(raw)
            if speaker_id == 1 and text_norm:
                if self._is_start(text_norm) and not self.state.mode_on:
                    self.state.mode_on = True
                    self.tts.speak("Translation mode enabled.", "en")
                    return PostResult(raw, det, en, None, None, mode_event="enabled")
                if self._is_stop(text_norm) and self.state.mode_on:
                    self.state.mode_on = False
                    self.tts.speak("Translation mode disabled.", "en")
                    return PostResult(raw, det, en, None, None, mode_event="disabled")
        #print("[debug] keys:", list(whisper_json.keys()))

        # When mode ON, route speech:
        other_txt = None
        other_lang = None
        if self.state.mode_on:
            other_lang = self.state.pick_other_lang()
            if is_english(det):
                # English -> Other
                if other_lang and other_lang != "en":
                    other_txt = self.tx.translate(raw, "en", other_lang)
                    if other_txt:
                        self.tts.speak(other_txt, other_lang)
                else:
                    # No known target yet; speak English as a cue
                    self.tts.speak(raw, "en")
            else:
                # Non-English -> English
                self.tts.speak(en, "en")

        return PostResult(raw, det, en, other_txt, other_lang)
