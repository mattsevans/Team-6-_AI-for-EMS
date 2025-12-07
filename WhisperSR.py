"""
Real-time Whisper transcription with endpointed listening via speech_recognition.

What this does:
- Uses speech_recognition (SR) to manage the microphone, detect speech start/stop
  (endpointing), and hand you complete "utterances".
- On each utterance, writes a small in-memory WAV to a temp file and transcribes
  with your local Whisper model (single model load, CPU-friendly).
- Exposes and documents SR’s tuning knobs: energy threshold, dynamic energy, pause
  threshold, and non-speaking duration so you can dial in latency vs. completeness.

Why SR + Whisper:
- SR abstracts PyAudio setup and gives you endpointing so you aren't forced into
  rigid 5-second blocks.
- You keep your local Whisper (no API needed), and you still control model size
  for CPU vs. accuracy tradeoffs.

Prereqs:
  pip install speechrecognition pyaudio openai-whisper

Notes for EMS tuning:
- Lower pause_threshold -> faster cut-offs (less latency, more fragments)
- Higher pause_threshold -> longer phrases (more context, slightly more lag)
- dynamic_energy_threshold=True is robust to changing noise environments.
- Run the short ambient calibration at startup (adjust_for_ambient_noise) with
  the mic in a representative environment (e.g., inside rig / ER room).
"""

import io
import os
import sys
import time
import tempfile
import threading
from typing import Optional

import speech_recognition as sr
import whisper


# --------------------------- Config: Whisper ---------------------------
MODEL_NAME = "base"      # "tiny"/"base" for CPU, "small" if you can afford it
USE_FP16 = False         # Keep False on CPU
WHISPER_OPTIONS = dict( # Add/override as needed
    task="transcribe",   # or "translate" if you want English outputs from other languages
    fp16=USE_FP16,
    # language=None,     # Leave None for auto-detect; set e.g. "en", "es", "zh" to lock language
)
# ----------------------------------------------------------------------


# ---------------- Config: SpeechRecognition endpointing ----------------
# Initial static threshold for VAD energy. If dynamic_energy_threshold=True,
# SR will adjust this over time (good for variable noise).
ENERGY_THRESHOLD = 300

# Let SR auto-adapt the energy threshold. Great for on-the-go noise shifts.
DYNAMIC_ENERGY = True

# How long of silence marks the end of an utterance (seconds).
# Lower -> faster cutoffs (lower latency), Higher -> more complete phrases.
PAUSE_THRESHOLD = 1.5

# Minimum gap of silence SR treats as non-speech (seconds). Helps reject micro-pauses.
NON_SPEAKING_DURATION = 0.2

# Optional hard cap on an utterance length (seconds). None = unlimited (SR decides).
PHRASE_TIME_LIMIT: Optional[float] = 25

# Ambient calibration duration at startup (seconds). 0 to disable.
AMBIENT_CALIBRATION_SEC = 0.8
# ----------------------------------------------------------------------


# --------------- Printing & threading (safe, simple queue) -------------
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)
        sys.stdout.flush()
# ----------------------------------------------------------------------


def write_wav_temp(audio_data: sr.AudioData, target_rate: int = 16000) -> str:
    """
    Convert SR AudioData to a temp WAV file at `target_rate` for Whisper.
    We use a NamedTemporaryFile because whisper.load_audio expects a file path.
    The file is cleaned up after each transcription.
    """
    # SR can render WAV bytes at a specified sample rate:
    wav_bytes = audio_data.get_wav_data(convert_rate=target_rate, convert_width=2)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(wav_bytes)
        tmp.flush()
    finally:
        tmp.close()
    return tmp.name


class SRWhisperRunner:
    """
    Encapsulates the SR (for capture + endpointing) and Whisper (for decode).
    """

    def __init__(self):
        # Load Whisper once (avoid reload per utterance)
        safe_print("Loading Whisper model...")
        self.model = whisper.load_model(MODEL_NAME)

        # SR recognizer + mic
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = DYNAMIC_ENERGY
        self.recognizer.pause_threshold = PAUSE_THRESHOLD
        self.recognizer.non_speaking_duration = NON_SPEAKING_DURATION

        self.microphone = sr.Microphone()  # default input device; change device_index=... if needed
        self._stopper = None  # background listener handle

    def calibrate(self):
        if AMBIENT_CALIBRATION_SEC and AMBIENT_CALIBRATION_SEC > 0:
            safe_print(f"Calibrating mic for {AMBIENT_CALIBRATION_SEC:.1f}s ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_CALIBRATION_SEC)
            safe_print(f"Calibrated energy_threshold={self.recognizer.energy_threshold:.1f}")

    def start(self):
        """
        Start the background listener. SR will call _on_utterance for each segment.
        """
        if self._stopper is not None:
            return

        safe_print("Listening... (Ctrl+C to stop)")
        self._stopper = self.recognizer.listen_in_background(
            self.microphone,
            self._on_utterance,
            phrase_time_limit=PHRASE_TIME_LIMIT
        )

    def stop(self):
        """
        Stop the background listener and release the microphone.
        """
        if self._stopper is not None:
            self._stopper(wait_for_stop=False)
            self._stopper = None

    # ----------------------- Callback on utterance -----------------------
    def _on_utterance(self, recognizer: sr.Recognizer, audio: sr.AudioData):
        """
        This runs in a thread whenever SR thinks an utterance finished.
        Keep it lean: do minimal work and return. Heavy work (Whisper) is OK
        here but avoid blocking too long; in high-throughput settings, offload
        to a worker thread or queue.
        """
        try:
            wav_path = write_wav_temp(audio, target_rate=16000)
            # Transcribe with your already-loaded Whisper model
            result = self.model.transcribe(wav_path, **WHISPER_OPTIONS)
            text = (result.get("text") or "").strip()

            if text:
                safe_print(text)
            # Clean up temp file
            try:
                os.remove(wav_path)
            except OSError:
                pass

        except Exception as e:
            safe_print(f"[Transcribe error] {type(e).__name__}: {e}")
    # --------------------------------------------------------------------


def main():
    runner = SRWhisperRunner()
    runner.calibrate()
    runner.start()
    try:
        # Keep main thread alive while background thread runs.
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        safe_print("\nStopping...")
    finally:
        runner.stop()
        safe_print("Stopped.")


if __name__ == "__main__":
    main()
