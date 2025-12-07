# ===================== post_whisper_translate.py =====================
from __future__ import annotations
import os, re, queue, threading, subprocess, tempfile
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

# ---------- Config via env ----------
TRANSLATION_ALWAYS_TO_EN = os.environ.get("TRANSLATION_ALWAYS_TO_EN", "1").lower() in {"1","true","yes","on"}
TRANSLATION_MODE_DEFAULT  = os.environ.get("TRANSLATION_MODE_DEFAULT", "off").lower() in {"on","true","1"}
TRANSLATION_LIB           = os.environ.get("TRANSLATION_LIB", "argos")   # "argos" recommended on Pi
TRANSLATION_MODE_PHRASES_START = os.environ.get("TRANSLATION_MODE_PHRASES_START",
    "start translation mode;enable translation mode;translation on; translate; turn on translation")
TRANSLATION_MODE_PHRASES_STOP  = os.environ.get("TRANSLATION_MODE_PHRASES_STOP",
    "stop translation mode;disable translation mode;translation off;turn off translation")
ARGOS_DATA_DIR = os.environ.get("ARGOS_DATA_DIR", "/usr/share/argos-translate")

TTS_ENGINE     = os.environ.get("TTS_ENGINE", "espeak-ng")   # "espeak-ng" or "pico2wave"
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
        self._cache_pair: Optional[Tuple[str,str]] = None
        self._cache_model = None
        self.safe_print = safe_print
        if impl == "argos":
            try:
                os.environ.setdefault("ARGOS_TRANSLATE_PACKAGE_DIR", ARGOS_DATA_DIR)
                import argostranslate.package, argostranslate.translate  # noqa
                self._argos_pkg = __import__("argostranslate.package")
                self._argos_tx  = __import__("argostranslate.translate")
            except Exception as e:
                self.safe_print(f"[WARN] Argos Translate not available: {e}")
                self.impl = "none"

    def _load_argos_pair(self, src: str, dst: str):
        pair = (src, dst)
        if self._cache_pair == pair and self._cache_model is not None:
            return self._cache_model
        installed = self._argos_tx.get_installed_languages()
        srcL = next((l for l in installed if l.code == src), None)
        dstL = next((l for l in installed if l.code == dst), None)
        if not srcL or not dstL:
            return None
        try:
            mdl = srcL.get_translation(dstL)
        except Exception:
            mdl = None
        self._cache_pair = pair
        self._cache_model = mdl
        return mdl

    def translate(self, text: str, src: Optional[str], dst: str) -> Optional[str]:
        if not text: return ""
        if self.impl != "argos":  # identity/fallback
            return text if (normalize_lang(src) == normalize_lang(dst)) else None
        src = normalize_lang(src) if src else None
        dst = normalize_lang(dst)
        if not dst: return None
        if src == dst: return text
        if src:
            mdl = self._load_argos_pair(src, dst)
            if mdl:
                try: return mdl.translate(text)
                except Exception: return None
        # No src or missing model — best-effort: try a small list
        for guess in ["es","fr","de","it","pt","ru","zh"]:
            mdl = self._load_argos_pair(guess, dst)
            if mdl:
                try: return mdl.translate(text)
                except Exception: continue
        return None

# ---------- TTS ----------
class TTSQueue:
    def __init__(self, engine=TTS_ENGINE, speed=TTS_SPEED_WPM, safe_print=_default_safe_print):
        self.engine = engine
        self.speed = speed
        self.q: "queue.Queue[Tuple[str,str]]" = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.safe_print = safe_print
        self.th = threading.Thread(target=self._worker, name="tts-worker", daemon=True)
        self.th.start()

    def speak(self, text: str, lang: str = "en"):
        if not text: return
        try: self.q.put_nowait( (text, normalize_lang(lang) or "en") )
        except queue.Full: self.safe_print("[WARN] TTS queue full; dropping output.")

    def close(self):
        self.stop_event.set()
        self.th.join(timeout=0.5)

    def speak_blocking(self, text: str, lang: str = "en"):
        """
        Public, synchronous TTS using the SAME engine/device as the async queue.
        Blocks until playback finishes. This lets callers pause mic capture while TTS plays.
        """
        if not text:
            return
        lang = normalize_lang(lang) or "en"
        try:
            if self.engine == "pico2wave":
                rc = self._pico(text, lang)
            else:
                rc = self._espeak(text, lang)
            if rc.returncode != 0:
                self.safe_print(f"[WARN] TTS (blocking) failed: {rc.stderr.decode(errors='ignore')[:160]}")
        except Exception as e:
            self.safe_print(f"[WARN] TTS (blocking) exception: {e}")

    def _espeak(self, txt: str, lang: str):
        voice_map = {"en":"en-us","es":"es","fr":"fr","de":"de","it":"it","pt":"pt","ru":"ru","zh":"zh"}
        v = voice_map.get(lang, "en")
        return subprocess.run(["espeak-ng","-v",v,"-s",str(self.speed), txt],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _pico(self, txt: str, lang: str):
        voice_map = {"en":"en-US","es":"es-ES","fr":"fr-FR","de":"de-DE","it":"it-IT"}
        v = voice_map.get(lang, "en-US")
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "tts.wav")
            p1 = subprocess.run(["pico2wave","-l",v,"-w",wav, txt],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p1.returncode != 0: return p1
            return subprocess.run(["aplay", wav], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _worker(self):
        while not self.stop_event.is_set():
            try: text, lang = self.q.get(timeout=0.2)
            except queue.Empty: continue
            try:
                rc = self._pico(text, lang) if self.engine == "pico2wave" else self._espeak(text, lang)
                if rc.returncode != 0:
                    self.safe_print(f"[WARN] TTS failed: {rc.stderr.decode(errors='ignore')[:160]}")
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
    def __init__(self, state: TranslationState, tts: TTSQueue, translator: Translator, safe_print=_default_safe_print):
        self.state = state
        self.tts = tts
        self.tx = translator
        self.safe_print = safe_print
        self._start_phrases = build_phrase_set(TRANSLATION_MODE_PHRASES_START)
        self._stop_phrases  = build_phrase_set(TRANSLATION_MODE_PHRASES_STOP)

    # Extract detected language from whisper.cpp JSON (if present)
    def extract_lang(self, data: Dict) -> Optional[str]:
        lang = data.get("language")
        if not lang:
            segs = data.get("segments") or data.get("transcription") or []
            if segs and isinstance(segs, list):
                lang = segs[0].get("language")
        return normalize_lang(lang)

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
# =================== end: post_whisper_translate.py ===================
