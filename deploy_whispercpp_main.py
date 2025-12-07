import os
import sys
import time
import wave
import queue
import math
import json
import shutil
import subprocess
from typing import Optional
import requests
import io
import re
from queue import Queue, Empty

import datetime #provides access to current time
import pytz #assigns time zoneimport json

# --- Button / GPIO imports ---
try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None  # Allows running on non-Pi systems

BUTTON_PIN = 18            # BCM pin number (physical pin 12)
LONG_PRESS_SEC = 5.0       # Seconds required for "end incident" long press


import threading
from pathlib import Path
from typing import List, Dict
import tempfile
from collections import deque
from dataclasses import  dataclass, field
from typing import Optional, Tuple

import numpy as np
import pyaudio
import webrtcvad
from python_speech_features import mfcc

#post processing import
from deploy_post_whisper_translate import (
    TranslationState, TTSQueue, Translator, PostWhisperProcessor, TRANSLATION_MODE_DEFAULT
)

#input imports
from io_inputs.input_wav import WavSource
# Only import MicSource if you actually plug a mic in later.
# Keeping this import is fine; it won't run unless INPUT_MODE=mic.
try:
    from io_inputs.input_mic import MicSource
except Exception:
    MicSource = None  # allows running on a Pi with no audio stack yet

#raspi input mode
from spi_audio import SpiAudioIO
#
#  MCP3201 ADC streaming source
try:
    from adc_record_test import AdcSource
except Exception:
    AdcSource = None  # allows non-Pi environments to still import this module


class SpiInputMode:
    def __init__(self, sample_rate=16000, frame_ms=20):
        self.io = SpiAudioIO(sample_rate=sample_rate, frame_ms=frame_ms)

    def __enter__(self):
        self.io.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.io.stop()

    def generator(self):
        """Yields raw int16 frames suitable for VAD/feature extraction/Whisper."""
        for frame in self.io.frames():
            yield frame  # np.int16 mono

    def play(self, frame_i16):
        """Optional: send audio to speaker (e.g., TTS output)."""
        self.io.write_frame(frame_i16)

class SpiSource:
    """
    Adapter to look like your other *Source classes*.
    Yields int16 mono frames at 16 kHz (or whatever you set).
    """
    def __init__(self, sample_rate=16000, frame_ms=20):
        if SpiInputMode is None:
            raise RuntimeError("SpiInputMode unavailable. Did you copy input_mode_spi.py and install python3-spidev on the Pi?")
        self._mode = SpiInputMode(sample_rate=sample_rate, frame_ms=frame_ms)
        self._ctx = None

    def __enter__(self):
        self._ctx = self._mode.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            self._mode.__exit__(exc_type, exc, tb)
            self._ctx = None

    def frames(self):
        # match your pipeline's expectation (like WavSource.frames())
        for f in self._mode.generator():
            yield f

'''
INPUT_MODE = os.getenv("INPUT_MODE", "wav")  # "wav" (default for your testing) or "mic" or "spi" for raspi input
WAV_GLOB   = os.getenv("WAV_GLOB", "test_audio/*.wav")
WAV_LOOP   = os.getenv("WAV_LOOP", "0").lower() in {"1","true","yes"}
WAV_RT     = os.getenv("WAV_REALTIME", "1").lower() in {"1","true","yes"}

# Optional: silence TTS on a headless Pi while testing
DISABLE_TTS = os.getenv("DISABLE_TTS", "1").lower() in {"1","true","yes"}
'''
#print("[python exe]", sys.executable)
#print("[cwd]", os.getcwd())
#print("[sys.path 0]", sys.path[0])
#print("[sys.path]", sys.path)
#print("[main file]", __file__)

# Config 
RATE = 16000                 # 16 kHz mono
CHANNELS = 1
SAMPLE_WIDTH = 2             # 16-bit
FRAME_MS = 20                # VAD frame size (10, 20, or 30 ms); we pick 20 ms
FRAME_SAMPLES = int(RATE * FRAME_MS / 1000)      # 320 samples
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH       # 640 bytes

VAD_AGGRESSIVENESS = 3       # 0-3; higher is more aggressive (more "speech" filtered)
VAD_HANGOVER_MS = 1000        # keep speech active this long after last voiced frame
VAD_HANGOVER_FRAMES = int(VAD_HANGOVER_MS / FRAME_MS)

# Diarizer settings (Lite)
EMB_WIN_SEC = 1.5            # embedding window
EMB_HOP_SEC = 0.5
EMB_WIN_FRAMES = int(EMB_WIN_SEC * 1000 / FRAME_MS)   # 75 frames
EMB_HOP_FRAMES = int(EMB_HOP_SEC * 1000 / FRAME_MS)   # 25 frames
EMB_N_MFCC = 24              # MFCC dims (20–26 reasonable)
SPK_SIM_THRESHOLD = 0.22     # cosine similarity threshold to create new speaker
SPK_CHANGE_CONFIRM_HOPS = 2  # require this many consecutive hops agreeing on a new speaker

# Segmenting
BOUNDARY_OVERLAP_MS = 500    # add to next/prev when closing segments
BOUNDARY_OVERLAP_FRAMES = int(BOUNDARY_OVERLAP_MS / FRAME_MS)

# whisper.cpp (CLI) configuration
#Environment file setup
# You can override these with environment variables at runtime
WHISPER_CPP_BIN = os.environ.get("WHISPER_CPP_BIN", "whisper.cpp/build/bin/Release/whisper-cli.exe")
WHISPER_CPP_MODEL = os.environ.get("WHISPER_CPP_MODEL", "whisper.cpp/ggml-tiny.bin")
#Smaller
#WHISPER_CPP_MODEL = os.environ.get("WHISPER_CPP_MODEL", "whisper.cpp/models/ggml-base-q5_1.bin")

WHISPER_CPP_THREADS = int(os.environ.get("WHISPER_CPP_THREADS", "4"))
WHISPER_CPP_LANG = os.environ.get("WHISPER_CPP_LANG", "")  # "" = auto
WHISPER_CPP_JSON_FULL = os.environ.get("WHISPER_CPP_JSON_FULL", "0") in {"1","true","True"}

#path to a text file that contains your initial_prompt
WHISPER_CPP_PROMPT_PATH = os.environ.get("WHISPER_CPP_PROMPT_PATH", "ems_prompt.txt")  # e.g., C:\path\ems_prompt.txt
# Workers
NUM_WHISPER_WORKERS = 1

# Target enrollment of EMS Speaker
ENROLLED_TARGET_PATH = os.environ.get("ENROLLED_TARGET_PATH", "enrolled_target.npz")
TARGET_ID = int(os.environ.get("TARGET_ID", "1"))  # force a stable ID for the target
TARGET_ACCEPT_BIAS = float(os.environ.get("TARGET_ACCEPT_BIAS", "0.03"))  # small bias toward target

JSONL_PATH = "MasterFileInput.jsonl"
 
#CLEAR JSON
try:
    with open(JSONL_PATH, 'w') as f:
        pass
    print(f"File '{JSONL_PATH}' cleared and now contains an empty JSON object.")
except IOError as e:
    print(f"Error clearing file '{JSONL_PATH}': {e}")


#audio input configs
INPUT_MODE = os.getenv("INPUT_MODE", "adc")  # "wav" (default for your testing) or "mic"
WAV_GLOB   = os.getenv("WAV_GLOB", "test_audio/*.wav")
WAV_LOOP   = os.getenv("WAV_LOOP", "0").lower() in {"1","true","yes"}
WAV_RT     = os.getenv("WAV_REALTIME", "1").lower() in {"1","true","yes"}
# Optional: silence TTS on a headless Pi while testing
DISABLE_TTS = os.getenv("DISABLE_TTS", "1").lower() in {"1","true","yes"}

'''
try:
    with open(JSONL_PATH, 'w') as f:
        pass
    print(f"File '{JSONL_PATH}' cleared and now contains an empty JSON object.")
except IOError as e:
    print(f"Error clearing file '{JSONL_PATH}': {e}")

'''
# --- Pause flag (if you don’t already have pause/resume functions) ---
CAPTURE_PAUSED = False  # global flag the capture loop will check
# --- Global stop flag (for button-triggered shutdown) ---
STOP_REQUESTED = False   # when True, main loop will exit cleanly


def _init_button():
    """
    Set up the GPIO button. Safe to call even if RPi.GPIO isn't available.
    """
    if GPIO is None:
        print("[BUTTON] RPi.GPIO not available; running without button control.", flush=True)
        return

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def _wait_for_start_button():
    """
    Block at startup until the button is pressed once.
    If GPIO is unavailable, this returns immediately.
    """
    if GPIO is None:
        print("[BUTTON] No GPIO; skipping start-button wait.", flush=True)
        return

    print(f"[BUTTON] Waiting for press on GPIO {BUTTON_PIN} to start pipeline...", flush=True)
    print("  - Not pressed -> HIGH (1)")
    print("  - Pressed     -> LOW  (0)")

    # Wait until the button goes LOW once
    while True:
        state = GPIO.input(BUTTON_PIN)
        if state == GPIO.LOW:
            print("[BUTTON] Start button press detected; starting pipeline.\n", flush=True)
            break
        time.sleep(0.05)


def _button_long_press_watcher():
    """
    Background thread that watches for a LONG_PRESS_SEC press.
    When detected:
      - Appends '[HH:MM:SS] EMS: Zora end incident' to MasterFileOutput.jsonl
      - Sets STOP_REQUESTED = True so the main loop exits.
    """
    global STOP_REQUESTED

    if GPIO is None:
        return  # nothing to do

    pressed_start = None

    while not STOP_REQUESTED:
        state = GPIO.input(BUTTON_PIN)

        if state == GPIO.LOW:
            # Button is currently pressed
            if pressed_start is None:
                pressed_start = time.time()
            else:
                held = time.time() - pressed_start
                if held >= LONG_PRESS_SEC:
                    # Long press detected -> write end-incident line, request shutdown
                    try:
                        tz = pytz.timezone("America/Chicago")
                        now = datetime.datetime.now(tz)
                        # If you really want the fixed literal time "16:53:18", replace {now:%H:%M:%S} with "16:53:18".
                        line = f"[{now:%H:%M:%S}] EMS: Zora end incident"

                        # INBOX_PATH is the MasterFileOutput.jsonl path
                        with open(JSONL_PATH, "a", encoding="utf-8") as f:
                            f.write(line + "\n")

                        with open(INBOX_PATH, "a", encoding="utf-8") as f:
                            f.write("Incident Closed" + "\n")

                        print("[BUTTON] Long press detected: wrote 'Zora end incident' and requesting shutdown.", flush=True)
                    except Exception as e:
                        print(f"[BUTTON] Error writing end-incident line: {e}", flush=True)

                    STOP_REQUESTED = True
                    break
        else:
            # Not pressed; reset timer
            pressed_start = None

        time.sleep(0.05)


def _start_button_long_press_thread():
    """
    Spawn the background watcher thread (daemon).
    """
    if GPIO is None:
        return

    th = threading.Thread(target=_button_long_press_watcher, name="ButtonMonitor", daemon=True)
    th.start()


@dataclass
class PendingSegment:
    speaker_id: int
    pcm_bytes: bytearray
    t0_frame: int
    t1_frame: int
    reason: str = "merged"

def _capture_pause_adapter():
    # If you already have a real pause function, call it here instead.
    # Otherwise, this flag is enough—just make your capture loop skip work when True.
    global CAPTURE_PAUSED
    CAPTURE_PAUSED = True

def _capture_resume_adapter():
    global CAPTURE_PAUSED
    CAPTURE_PAUSED = False

# Will be bound to your *actual* TTSQueue instance at runtime
_speak_en_adapter = None

def _bind_inbox_speak(tts_queue):
    """Attach the inbox speaker to your existing TTSQueue so it uses the same voice/device."""
    global _speak_en_adapter
    def _speak_en_blocking(text: str):
        tts_queue.speak_blocking(text, "en")
    _speak_en_adapter = _speak_en_blocking

_inbox_speaker = None

def start_inbox_tts():
    """Start tailing inbox.jsonl and speaking lines as they’re appended."""
    global _inbox_speaker, _speak_en_adapter
    if _inbox_speaker is not None:
        return
    if _speak_en_adapter is None:
        raise RuntimeError("Call _bind_inbox_speak(tts) before start_inbox_tts().")

    _inbox_speaker = InboxSpeaker(
        inbox_path=INBOX_PATH,
        speak_fn=_speak_en_adapter,
        pause_fn=_capture_pause_adapter,
        resume_fn=_capture_resume_adapter,
        poll_ms=POLL_MS,
        start_at_eof=START_AT_EOF,
    )
    _inbox_speaker.start()

def stop_inbox_tts():
    global _inbox_speaker
    if _inbox_speaker:
        _inbox_speaker.stop()
        _inbox_speaker = None

def make_source():
    if INPUT_MODE == "mic":
        if MicSource is None:
            raise RuntimeError("MicSource unavailable (no mic/pyaudio). Set INPUT_MODE=wav for now.")
        return MicSource()

    elif INPUT_MODE == "spi":
        # Existing SPI-in path (SpiAudioIO). Leave behavior unchanged.
        sr = 16000 if 'SAMPLE_RATE' not in globals() else RATE
        fm = 20    if 'FRAME_MS'    not in globals() else FRAME_MS
        return SpiSource(sample_rate=sr, frame_ms=fm)

    elif INPUT_MODE == "adc":
        if AdcSource is None:
            raise RuntimeError(
                "AdcSource unavailable (could not import from adc_record_test.py, "
                "or spidev not installed?)."
            )

        # Default ADC environment:
        #   sample_rate = 44100 Hz
        #   frame_samples = 882 (≈20 ms)
        #   spi_speed = 1_000_000 Hz
        #   mode = 0
        #
        # Allow overrides via env vars if you want (but these defaults match your request).
        adc_sample_rate = int(os.getenv("ADC_SAMPLE_RATE", "44100"))
        adc_frame_samples = int(os.getenv("ADC_FRAME_SAMPLES", "882"))
        # Compute frame_ms from samples and sample rate (will be 20 ms with defaults).
        adc_frame_ms = int(1000 * adc_frame_samples / adc_sample_rate)

        spi_bus = int(os.getenv("ADC_SPI_BUS", "0"))
        spi_ce = int(os.getenv("ADC_SPI_CE", "0"))
        spi_mode = int(os.getenv("ADC_SPI_MODE", "0"))
        spi_speed = int(os.getenv("ADC_SPI_SPEED", "1000000"))
        dc_block = os.getenv("ADC_DC_BLOCK", "0").lower() in {"1", "true", "yes"}

        return AdcSource(
            sample_rate=adc_sample_rate,
            frame_ms=adc_frame_ms,
            spi_bus=spi_bus,
            spi_ce=spi_ce,
            spi_mode=spi_mode,
            spi_speed_hz=spi_speed,
            dc_block=dc_block,
        )

    else:
        return WavSource(wav_glob=WAV_GLOB, realtime=WAV_RT, loop=WAV_LOOP)

# Description: Safe print helper that forces immediate flush so logs appear in real time.
def safe_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# Description: Normalize raw PCM audio: remove DC offset, RMS-normalize (with a cap), and return int16 bytes.
def preprocess_pcm(pcm_bytes: bytes, target_rms=0.1, max_gain=8.0) -> bytes:
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    # DC remove
    x = x - np.mean(x)
    # RMS normalize
    rms = np.sqrt(np.mean(x * x) + 1e-12)
    if rms > 0:
        gain = min(max_gain, target_rms / rms)
        x = x * gain
    # Clip-safe back to int16
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()

# Description: Convert PCM bytes to a NumPy int16 array without copying.
def bytes_to_np_int16(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16)

# Description: Convert a NumPy array to int16 PCM bytes.
def np_int16_to_bytes(arr: np.ndarray) -> bytes:
    return arr.astype(np.int16).tobytes()

def resample_frame_to_diarizer_rate(frame_bytes: bytes) -> bytes:
    """
    Linearly resample a single 20 ms frame from the ADC native rate
    (e.g., 44.1 kHz, 882 samples) to the diarizer/Whisper rate
    (RATE, FRAME_SAMPLES – currently 16 kHz, 320 samples).

    - Input:  int16 mono PCM bytes at some fixed rate (per 20 ms)
    - Output: int16 mono PCM bytes with length FRAME_SAMPLES

    This assumes each input 'frame_bytes' is one contiguous 20 ms frame.
    """

    arr_in = bytes_to_np_int16(frame_bytes)
    in_len = arr_in.shape[0]

    # Target length is whatever the diarizer expects (16 kHz * 20 ms = 320).
    out_len = FRAME_SAMPLES

    # If it's already the correct length, just return as-is.
    if in_len == out_len:
        return frame_bytes

    # Convert to float for interpolation
    x = arr_in.astype(np.float32)

    # Handle degenerate cases safely
    if in_len <= 1:
        # Just repeat or truncate to target length
        y = np.full(out_len, x[0] if in_len == 1 else 0.0, dtype=np.float32)
    else:
        # Normalize both input and output grids to [0, 1]
        t_in = np.linspace(0.0, 1.0, num=in_len, endpoint=True, dtype=np.float32)
        t_out = np.linspace(0.0, 1.0, num=out_len, endpoint=True, dtype=np.float32)
        y = np.interp(t_out, t_in, x).astype(np.float32)

    # Clip back to int16 range
    y = np.clip(y, -32768.0, 32767.0).astype(np.int16)
    return y.tobytes()

# Description: Write raw PCM bytes to a WAV file with the expected format (rate/channels/sample width).
def write_wav(path: str, pcm_bytes: bytes, rate: int = RATE, channels: int = 1, sampwidth: int = SAMPLE_WIDTH):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)

# Description: Compute cosine similarity between two vectors with small epsilon to avoid divide-by-zero.
def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

def l2norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)



@dataclass
# Description: Represents a tracked speaker with an embedding centroid and count of updates.
class Speaker:
    id: int
    centroid: np.ndarray
    count: int = 1

    # Description: Initialize a speaker with a zeroed centroid of a given embedding dimension.
    def __init__(self, emb_dim):
        self.centroid = np.zeros(emb_dim, dtype=np.float32)
        self.count = 0

    # Description: Update the speaker centroid using an exponential moving average; optionally adapts step by similarity.
    def update(self, emb: np.ndarray, sim: Optional[float] = None) ->  None:
        """Update centroid with an exponential moving average.
        If sim (cosine similarity) is provided, use it to adapt alpha."""
        e = emb.astype(np.float32)
        n = np.linalg.norm(e) + 1e-8
        e = e / n

        if self.count == 0:
            self.centroid = e
        else:
            # base alpha from env or default
            base_alpha = float(os.environ.get("EMB_EMA_ALPHA", "0.15"))
            alpha = base_alpha
            if sim is not None:
                # map cosine sim [-1,1] -> [0,1]
                w = max(0.0, min(1.0, 0.5 * (sim + 1.0)))
                # allow 0.05..0.5 step depending on confidence
                alpha = 0.05 + 0.45 * w
            self.centroid = (1.0 - alpha) * self.centroid + alpha * e
            # keep centroid normalized
            self.centroid /= (np.linalg.norm(self.centroid) + 1e-8)
        self.count += 1


# Description: Lightweight online diarization pipeline that chunks mic audio by speech and assigns speaker IDs.
class LiteDiarizer:
    """
    VAD -> rolling voiced frames -> MFCC embeddings over sliding windows -> online centroid assignment.
    Emits (speaker_id, segment_bytes, t0_frames, t1_frames) when a segment closes
    due to speaker change or speech -> silence.
    """
    # Description: Set up VAD, buffers, speaker tracking, and an optional segment-closed callback.
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.frame_idx = 0

        # "Hold" keeps the current speaker label for a few frames to prevent flicker.
        self.HOLD_FRAMES = int(os.environ.get("DIAR_HOLD_FRAMES", "5"))  # ~5 frames @20 ms ≈ 100 ms
        self.hold_counter = 0
        # Also make sure these are initialized if you reference them elsewhere
        self.current_speaker = None          # active speaker label or None
        self.prev_embedding = None           # for centroid smoothing (if you use it)
        
        # ring buffers
        self.recent_frames = deque(maxlen=5 * 1000 // FRAME_MS)  # ~5 s lookback for overlap
        self.voiced_frames = []      # current active speech frames (bytes)
        self.voiced_mask = []        # 1/0 per frame (aligned to voiced_frames)

        # diarization state
        self.speakers: List[Speaker] = []
        self.current_spk: Optional[int] = None
        self.pending_spk: Optional[int] = None
        self.pending_spk_hops = 0

        # for embedding hops
        self.emb_cursor = 0  # counts frames within current voiced run

        # output callback (set by user)
        self.on_segment_closed = None  # fn(speaker_id, pcm_bytes, t0_frame, t1_frame)
        
        # thresholds / guards for speaker-birth and post-switch stability
        self.low_sim_runs = 0
        self.min_birth_hops = int(os.environ.get("DIAR_MIN_BIRTH_HOPS", "3"))
        self.hold_after_switch_hops = int(os.environ.get("DIAR_HOLD_AFTER_SWITCH_HOPS", "6"))

        #pre-roll config (keeps a short buffer before VAD triggers) 
        self.PRE_ROLL_MS = int(os.environ.get("VAD_PRE_ROLL_MS", "1000"))  # try 150–300 ms
        self.PRE_ROLL_FRAMES = max(0, int(self.PRE_ROLL_MS / FRAME_MS))
        self.in_segment = False  # track if we are currently inside a speech segment
    

        #preload an EMS target speaker 
        self.target = None
        if ENROLLED_TARGET_PATH and Path(ENROLLED_TARGET_PATH).exists():
            try:
                d = np.load(ENROLLED_TARGET_PATH)
                # raw centroid from file -> normalize now
                centroid = l2norm(d["centroid"])
                protos = d["prototypes"].astype(np.float32)
                # ensure prototypes are normalized (the file may already have them L2'd)
                protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9)
                mu = float(d.get("mu", 0.6))
                sig = float(d.get("sig", 0.05))
                self.target = {
                    "id": TARGET_ID,
                    "centroid": centroid,
                    "protos": protos,
                    "mu": mu,
                    "sig": sig,
                }
                # Also seed the speakers list with the target's centroid so ID is stable
                spk = Speaker(emb_dim=centroid.shape[0])
                spk.id = TARGET_ID
                spk.centroid = centroid.copy()
                spk.count = 1
                self.speakers.append(spk)
                safe_print(f"[INFO] Enrolled target loaded: id={TARGET_ID}, protos={protos.shape[0]}")
            except Exception as e:
                safe_print(f"[WARN] Could not load enrolled target: {e}")


    # Description: Ask WebRTC VAD whether this 20 ms frame likely contains speech.
    def _frame_is_speech(self, frame_bytes: bytes) -> bool:
        #WebRTC VAD wants 16-bit mono PCM at 8/16/32 kHz and frames of 10/20/30 ms
        #yes or no if there was speech detected in each frame
        return self.vad.is_speech(frame_bytes, RATE)

    # Description: Compute a fixed-size MFCC embedding over the recent audio window to represent voice timbre.
    def _compute_embedding(self, pcm_bytes_concat: bytes) -> np.ndarray:
        # Convert bytes -> float32 [-1,1]
        x = bytes_to_np_int16(pcm_bytes_concat).astype(np.float32) / 32768.0
        # MFCC over ~1s window; return mean across time as fixed-size embedding
        m = mfcc(
            x,
            samplerate=RATE,
            winlen=0.025,    # 25 ms
            winstep=0.010,   # 10 ms
            numcep=EMB_N_MFCC,
            nfilt=26,
            nfft=512,
            preemph=0.97,
            appendEnergy=True
        )
        emb = m.mean(axis=0)  # (EMB_N_MFCC,)
        return emb

    # Description: Choose or create the best-matching speaker ID for the given embedding; handles label stability and births.
    def _assign_speaker(self, emb: np.ndarray) -> int:
        if not self.speakers:
            # ===================== BEGIN PATCH: callsite alternative =====================
            spk = Speaker(emb_dim=emb.shape[0])
            spk.id = 1
            spk.centroid = emb.astype(np.float32).copy()
            spk.centroid /= (np.linalg.norm(spk.centroid) + 1e-8)
            spk.count = 1
# ====================== END PATCH: callsite alternative ======================
            self.speakers.append(spk)
            return spk.id
        
                # ----- Target-first multi-prototype check (if enrolled) -----
        if self.target is not None:
            emb_n = l2norm(emb)
            sims_t = self.target["protos"] @ emb_n  # cosine since rows are L2-normed
            sim_t = float(np.max(sims_t)) if sims_t.size else float(np.dot(self.target["centroid"], emb_n))

            # adaptive threshold around enrollment similarity stats (looser than others)
            t_accept = max(0.35, self.target["mu"] - 2.0*self.target["sig"]) + TARGET_ACCEPT_BIAS

            if sim_t >= t_accept:
                # assign to target, with *very slow* EMA centroid update only if confident
                # find the Speaker object that holds TARGET_ID (it exists—seeded in __init__)
                for s in self.speakers:
                    if s.id == self.target["id"]:
                        # only update if strong evidence (prevents drift)
                        if sim_t >= max(t_accept + 0.10, 0.60):
                            s.update(emb, sim_t)
                            # keep the cached centroid in self.target in sync (normalized already by s.update)
                            self.target["centroid"] = s.centroid.copy()
                            # nudge stats slowly
                            b = 0.02
                            self.target["mu"]  = (1-b)*self.target["mu"]  + b*sim_t
                            self.target["sig"] = (1-b)*self.target["sig"] + b*abs(sim_t - self.target["mu"])
                        return self.target["id"]
                # if somehow not found (shouldn't happen), fall through to normal path


        # find best match
        sims = [cosine_sim(emb, s.centroid) for s in self.speakers]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        best_spk = self.speakers[best_idx]

        # Per-speaker adaptive accept threshold
        adapt = max(0.35, best_spk.mu_sim - 2.0*best_spk.sig_sim) if hasattr(best_spk, 'mu_sim') else 0.40

        if best_sim >= adapt:
            self.low_sim_runs = 0
            hc = getattr(self, "hold_counter", 0)
            hc = max(0, hc - 1)
            self.hold_counter = hc            
            best_spk.update(emb, best_sim)
            return best_spk.id
        else:
            self.low_sim_runs += 1
            if self.hold_counter > 0:
                # still in cooldown after a switch → stick to best_spk
                best_spk.update(emb, best_sim*0.0)  # no centroid move
                return best_spk.id

            if self.low_sim_runs >= self.min_birth_hops:
                new_id = max(s.id for s in self.speakers) + 1
                spk = Speaker(emb_dim=emb.shape[0])
                spk.id = new_id
                spk.centroid = emb.astype(np.float32).copy()
                spk.centroid /= (np.linalg.norm(spk.centroid) + 1e-8)
                spk.count = 1
                self.speakers.append(spk)
                self.low_sim_runs = 0
                self.hold_counter = self.hold_after_switch_hops
                return spk.id
            else:
                # provisional stickiness to best_spk without updating centroid
                return best_spk.id
        

    # Description: Finalize and emit the current voiced segment (with a small overlap) and reset state.
    def _close_current_segment(self, reason: str):
        if not self.voiced_frames or self.current_spk is None:
            # nothing to emit
            self.voiced_frames.clear()
            self.voiced_mask.clear()
            self.emb_cursor = 0
            return

        # add overlap from recent_frames tail
        overlap = list(self.recent_frames)[-BOUNDARY_OVERLAP_FRAMES:]
        seg_frames = overlap + self.voiced_frames
        seg_pcm = b"".join(seg_frames)

        # compute times (approximate) in frames
        total_frames = len(seg_frames)
        t1 = self.frame_idx             # current global frame index
        t0 = max(0, t1 - total_frames)

        if self.on_segment_closed:
            self.on_segment_closed(self.current_spk, seg_pcm, t0, t1, reason)

        # reset
        self.voiced_frames.clear()
        self.voiced_mask.clear()
        self.emb_cursor = 0

    # Description: Consume 20 ms frames one-by-one: run VAD, manage segment boundaries, and trigger diarization hops.
    def process_frame(self, frame_bytes: bytes):
        """Push one 20 ms frame (FRAME_BYTES)"""
        assert len(frame_bytes) == FRAME_BYTES, "Expected exact 20ms frame bytes"

        if self.frame_idx % 50 == 0:
            arr = bytes_to_np_int16(frame_bytes)
            rms = float(np.sqrt((arr.astype(np.float32)**2).mean()))
            # safe_print(f"debug: rms={rms:.1f}")

        self.recent_frames.append(frame_bytes)
        is_speech = self._frame_is_speech(frame_bytes)
        # Maintain voiced run with hangover, with pre-roll on entry
        if is_speech:
            # If we weren't already in a segment, this is a speech "onset"
            if not self.in_segment:
                # Take the last PRE_ROLL_FRAMES from recent_frames and prepend them
                # Mark them as voiced (=1) so later trimming logic doesn't delete them
                if self.PRE_ROLL_FRAMES > 0:
                    pre = list(self.recent_frames)[-self.PRE_ROLL_FRAMES:]
                    if pre:
                        self.voiced_frames.extend(pre)
                        self.voiced_mask.extend([1] * len(pre))
                        self.emb_cursor += len(pre)
                self.in_segment = True

            # Always append the current speech frame
            self.voiced_frames.append(frame_bytes)
            self.voiced_mask.append(1)
            self.emb_cursor += 1

        else:
            # Only append non-speech if we are already inside a segment
            if self.in_segment and any(self.voiced_mask):
                self.voiced_frames.append(frame_bytes)
                self.voiced_mask.append(0)
                self.emb_cursor += 1

    # trailing hangover closing logic stays the same below

    # trailing hangover closing logic stays the same below

            tail = 0
            for v in reversed(self.voiced_mask):
                if v == 0:
                    tail += 1
                else:
                    break
            if tail >= VAD_HANGOVER_FRAMES and any(self.voiced_mask):
                self._close_current_segment(reason="silence")
                self.current_spk = None
                self.pending_spk = None
                self.pending_spk_hops = 0
                # after closing, drop lingering frames to avoid duplicate capture
                self.voiced_frames.clear()
                self.voiced_mask.clear()
                self.emb_cursor = 0

        # Diarization hop logic (only if we have enough audio in the current run)
        # Use only frames marked as speech to compute embedding window content
        # For robustness, we keep hop clock even with some non-speech inside.
        if len(self.voiced_frames) >= EMB_WIN_FRAMES and (self.emb_cursor - EMB_WIN_FRAMES) % EMB_HOP_FRAMES == 0:
            # Take last EMB_WIN_FRAMES worth of frames
            win_frames = self.voiced_frames[-EMB_WIN_FRAMES:]
            pcm_concat = b"".join(win_frames)

            emb = self._compute_embedding(pcm_concat)
            spk_id = self._assign_speaker(emb)

            if self.current_spk is None:
                self.current_spk = spk_id
                self.pending_spk = None
                self.pending_spk_hops = 0
            else:
                if spk_id != self.current_spk:
                    # require confirmation across several hops
                    if self.pending_spk is None or self.pending_spk != spk_id:
                        self.pending_spk = spk_id
                        self.pending_spk_hops = 1
                    else:
                        self.pending_spk_hops += 1

                    if self.pending_spk_hops >= SPK_CHANGE_CONFIRM_HOPS:
                        # close current segment and switch
                        self._close_current_segment(reason="speaker_change")
                        self.current_spk = spk_id
                        self.pending_spk = None
                        self.pending_spk_hops = 0
                else:
                    # stable; clear pending
                    self.pending_spk = None
                    self.pending_spk_hops = 0

        self.frame_idx += 1


######    Audio Capture
# Description: Minimal microphone wrapper to open the audio stream and read fixed-size frames.
class MicReader:
    # Description: Create a PyAudio instance; stream is opened later.
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None

    # Description: Open the input stream configured to produce 20 ms int16 frames at the target sample rate.
    def open(self):
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAME_SAMPLES
        )

    # Description: Read exactly one frame (20 ms) from the microphone without throwing on overflow.
    def read_frame(self) -> bytes:
        return self.stream.read(FRAME_SAMPLES, exception_on_overflow=False)

    # Description: Close the stream and terminate PyAudio safely.
    def close(self):
        try:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
        finally:
            self.pa.terminate()

# (Optional) Glossary plumbing — currently returns empty list; extend as needed
GLOSSARY_PATH = Path(os.environ.get("GLOSSARY_PATH", "ems_glossary_seed.csv"))

# Description: Placeholder to load domain terms for prompting; currently returns an empty list.
def load_glossary(path: Path = GLOSSARY_PATH) -> List[str]:
    try:
        # Implement your CSV loading here if you want to bias prompts later.
        return []
    except Exception:
        return []

# -------------------- whisper.cpp Worker(s) --------------------
@dataclass
# Description: A unit of work handed to the transcriber: who spoke, audio bytes, time bounds, and why the segment ended.
class Job:
    speaker_id: int
    pcm_bytes: bytes
    t0_frame: int
    t1_frame: int
    reason: str



#def append_jsonl(record: dict, path: str = JSONL_PATH):
 ##  with _JSONL_LOCK:
   #     with open(path, "a", encoding="utf-8") as f:
    #        f.write(json.dumps(record, ensure_ascii=False) + "\n")
     #       f.flush()
      #      os.fsync(f.fileno())

def append_jsonl(line, path=JSONL_PATH):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# Description: Background worker pool that writes temp WAVs, calls whisper.cpp CLI, parses JSON, and logs results.
class WhisperWorkers:
    """Transcription workers that call whisper.cpp's CLI and parse JSON output."""
    # Description: Start worker threads, read optional prompt file, and warn if binary/model paths look wrong.
    def __init__(self, num_workers=1, postproc: 'PostWhisperProcessor'=None):
        self.q = queue.Queue(maxsize=64)
        self.threads = []
        self.stop_event = threading.Event()
        self.postproc = postproc

        #load prompt text (optional)
        self.prompt_text = None
        if WHISPER_CPP_PROMPT_PATH:
            try:
                self.prompt_text = Path(WHISPER_CPP_PROMPT_PATH).read_text(encoding="utf-8").strip()
                if self.prompt_text:
                    safe_print(f"[INFO] Loaded initial prompt from: {WHISPER_CPP_PROMPT_PATH} "
                               f"({len(self.prompt_text.split())} words)")
            except Exception as e:
                safe_print(f"[WARN] Could not read prompt file '{WHISPER_CPP_PROMPT_PATH}': {e}")
                self.prompt_text = None

        # Simple checks to help at startup
        if not shutil.which(WHISPER_CPP_BIN) and not Path(WHISPER_CPP_BIN).exists():
            safe_print(f"[WARN] whisper.cpp binary not found at '{WHISPER_CPP_BIN}'. Set WHISPER_CPP_BIN or adjust path.")
        if not Path(WHISPER_CPP_MODEL).exists():
            safe_print(f"[WARN] whisper.cpp model file not found at '{WHISPER_CPP_MODEL}'. Set WHISPER_CPP_MODEL or download one.")

        for i in range(num_workers):
            th = threading.Thread(target=self._worker, name=f"whispercpp-{i}", daemon=True)
            th.start()
            self.threads.append(th)

    # Description: Try to enqueue a transcription Job; drops if the queue is full to avoid backpressure.
    def submit(self, job: Job):
        try:
            self.q.put_nowait(job)
        except queue.Full:
            safe_print("[WARN] Transcription queue full; dropping segment.")

    # Description: Build and run the whisper.cpp CLI command on a WAV file; return parsed JSON output.
    """def _run_whispercpp(self, wav_path: str, out_prefix: str) -> Dict:
        cmd = [
            WHISPER_CPP_BIN,
            "-m", WHISPER_CPP_MODEL,
            "-f", wav_path,
            "-t", str(WHISPER_CPP_THREADS),
            "-oj",                 #JSON output
            "-of", out_prefix,      #output file prefix
            "--beam-size", "1", #increase to improve accuracy. normal is 5
            "--temperature", "0" #increase to improve accuracy. normal is 1
        ]
        if WHISPER_CPP_LANG:
            cmd += ["-l", WHISPER_CPP_LANG]
        if self.prompt_text:
            cmd += ["--prompt", self.prompt_text]
            print("whisper prompted") # for testing if prompt worked
        if WHISPER_CPP_JSON_FULL:
            cmd += ["-ojf"]

     
        # Run
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"whisper.cpp failed (code {proc.returncode}): {proc.stderr.decode(errors='ignore')}")

        json_path = out_prefix + ".json"
        #print("[WHISPER CMD]", " ".join(cmd))
        #print("[WHISPER JSON PATH]", json_path)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"whisper.cpp produced no JSON at {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data"""
    
    def _run_whispercpp(self, wav_path: str, out_prefix: str) -> Dict:
        """
        Send a WAV file to a long-lived whisper.cpp HTTP server instead of
        spawning whisper-cli for each segment.
        """
        server_url = os.environ.get("WHISPER_SERVER_URL", "http://127.0.0.1:8080/inference")

        with open(wav_path, "rb") as f:
            files = {"file": ("seg.wav", f, "audio/wav")}
            data = {
                "temperature": "0.0",
                "temperature_inc": "0.0",  # no fallback; keep deterministic like your CLI flags
                "response_format": "json",
            }
            # You can also send language here if you want:
            if WHISPER_CPP_LANG:
                data["language"] = "auto"

            resp = requests.post(server_url, files=files, data=data, timeout=60)

        resp.raise_for_status()
        data = resp.json()
        return data


    # Description: Thread target that drains the job queue, runs preprocessing + whisper.cpp, and logs transcription.
    def _worker(self):
        while not self.stop_event.is_set():
            try:
                job: Job = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            
            # ---- HARD DURATION CAP (e.g., 28 seconds) ----
            # Compute current duration from PCM bytes
            bytes_per_second = RATE * CHANNELS * SAMPLE_WIDTH
            cur_sec = len(job.pcm_bytes) / float(bytes_per_second)

            # Safety cap so Whisper never sees audio > MAX_CLIP_SEC
            MAX_CLIP_SEC = 28.0  # keep in sync with value in main()

            if cur_sec > MAX_CLIP_SEC:
                safe_print(f"[WARN] Segment {cur_sec:.2f}s > {MAX_CLIP_SEC:.2f}s; trimming before Whisper.")

                # Trim PCM bytes
                max_bytes = int(MAX_CLIP_SEC * bytes_per_second)
                job.pcm_bytes = job.pcm_bytes[:max_bytes]

                # Adjust t1_frame so logging/RTF use the trimmed duration
                orig_frames = job.t1_frame - job.t0_frame
                max_frames = int(MAX_CLIP_SEC * 1000.0 / FRAME_MS)
                new_frames = min(orig_frames, max_frames)
                job.t1_frame = job.t0_frame + new_frames
            # -----------------------------------------------
                                         
            tmp_path = None #placeholder for temp Wav file
            try:
                # Write temp WAV and transcribe via whisper.cpp
                with tempfile.TemporaryDirectory() as td:
                    tmp_path = os.path.join(td, "seg.wav")
                    out_prefix = os.path.join(td, "out")
                    
                    tz = pytz.timezone("America/Chicago") #sets time zone
                    StartTime = datetime.datetime.now(tz) #finds start of data processing time

                    clean = preprocess_pcm(job.pcm_bytes)
                    write_wav(tmp_path, clean, rate=RATE, channels=CHANNELS, sampwidth=SAMPLE_WIDTH)
                    print("wav created")
                    # --- duration debug (add this right after write_wav) ---
                    dur_bytes = len(job.pcm_bytes) / (RATE * CHANNELS * SAMPLE_WIDTH)
                    with wave.open(tmp_path, "rb") as wf:
                        dur_wav = wf.getnframes() / wf.getframerate()
                    safe_print(f"[SEG] len(pcm_bytes)≈{dur_bytes:.3f}s | WAV file={dur_wav:.3f}s | frames={wf.getnframes()} @ {wf.getframerate()} Hz")
                    # -------------------------------------------------------
                    whisper_t0 = time.perf_counter()
                    data = self._run_whispercpp(tmp_path, out_prefix)
                    whisper_t1 = time.perf_counter()
                    safe_print(f"[DEBUG] whisper.cpp core time (this segment): {whisper_t1 - whisper_t0:.3f}s")
                    
                    #print("DEBUG pcm_bytes_len:", len(job.pcm_bytes))
                    
                    # Assemble text from possible shapes of the response
                    text = ""

                    # 1) If there's an explicit "transcription" list, use that
                    transcription = data.get("transcription")
                    if isinstance(transcription, list):
                        text = " ".join(s.get("text", "").strip() for s in transcription).strip()

                    # 2) Otherwise, try "segments" (what whisper-server likely returns)
                    if not text:
                        segs = data.get("segments")
                        if isinstance(segs, list):
                            text = " ".join(s.get("text", "").strip() for s in segs).strip()

                    # 3) Finally, fall back to top-level "text" if present
                    if not text:
                        top = data.get("text")
                        if isinstance(top, str):
                            text = top.strip()

                        # Optional: debug once to see server JSON shape
                        # safe_print("DEBUG whisper-server keys:", list(data.keys()))

                    if text:
                        if self.postproc:
                            meta = self.postproc.handle_segment(job.speaker_id, text, data)
                            # Your logging — keep it short to avoid overhead
                            src = meta.detected_lang or "unk"
                            #safe_print(f"[{job.t0_frame/50.0:07.2f}] Spk{job.speaker_id} (src={src}): {meta.text_en}")
                        # Convert frames to seconds for display
                        t0_sec = job.t0_frame * (FRAME_MS / 1000.0)
                        t1_sec = job.t1_frame * (FRAME_MS / 1000.0)                            
                        dt = datetime.datetime.now().astimezone()   # respects system/local TZ
                        
                        if "Zora and incident" in text:
                            text = text.replace("Zora and incident", "Zora end incident")

                        if "Zora and Incident" in text:
                            text = text.replace("Zora and Incident", "Zora end incident")

                        if "Zwa" in text:
                            text = text.replace("Zwa", "Zora")

                        # NEW: collapse all whitespace (including newlines) into single spaces
                        # so each Whisper output is a single line in the JSONL file.
                        text_one_line = " ".join(text.split())

                        if job.speaker_id == 1:
                            line = f"[{dt:%H:%M:%S}] EMS: {text_one_line}"
                        else: 
                            line = f"[{dt:%H:%M:%S}] Speaker{job.speaker_id-1}: {text_one_line}"
                        safe_print(line)

                        # Only append to JSONL if the *whisper text* is not just () or [] content
                        core = text_one_line.strip()
                        # Match strings like "(crowd cheering)" or "[BLANK_AUDIO]" with nothing else
                        is_bracketed = bool(re.fullmatch(r'[\(\[][^()\[\]]*[\)\]]', core))
                        if not is_bracketed:
                            append_jsonl(line)

                    AudioDuration = t1_sec - t0_sec
                    EndTime = datetime.datetime.now(tz) #finds end of data processing time
                    ProcTime = EndTime - StartTime #net difference time
                    RTF = ProcTime / AudioDuration #real time factor
                    print(f"Processing time: {ProcTime}")
                    print(f"RTF: {RTF}")

            except Exception as e:
                safe_print(f"[Transcribe error] {type(e).__name__}: {e}")
            finally:
                self.q.task_done()
                # temp files cleaned up by TemporaryDirectory

    # Description: Signal all worker threads to stop and join them briefly.
    def close(self):
        self.stop_event.set()
        for th in self.threads:
            th.join(timeout=0.5)


# =========================
# Inbox -> Speak (English) 
# =========================

INBOX_PATH = os.environ.get("INBOX_PATH", "MasterFileOutput.jsonl")
POLL_MS = int(os.environ.get("INBOX_POLL_MS", "150"))
CHUNK_SOFT_MAX_CHARS = int(os.environ.get("INBOX_CHUNK_SOFT", "250"))
CHUNK_HARD_MAX_CHARS = int(os.environ.get("INBOX_CHUNK_HARD", "350"))
START_AT_EOF = os.environ.get("INBOX_START_AT_EOF", "1") not in ("0", "false", "False")

_SENT_SPLIT_RE = re.compile(r'(?<=[\.!\?])\s+')

class InboxSpeaker:
    """
    Tails a newline-delimited *plain text* file (no keys/JSON) and speaks each new line in English.
    - Uses the SAME TTS engine + audio device you already use in post_whisper_translate (provided as a callable).
    - Pauses mic capture during TTS; resumes immediately after (no VAD grace).
    - Ignores existing lines at startup (tail -f semantics).
    """
    def __init__(self, inbox_path, speak_fn, pause_fn, resume_fn, 
                 poll_ms=POLL_MS, start_at_eof=START_AT_EOF):
        self.inbox_path = inbox_path
        self.speak_fn = speak_fn         # callable(text:str) -> None  (must block until audio done)
        self.pause_fn = pause_fn         # callable() -> None
        self.resume_fn = resume_fn       # callable() -> None
        self.poll_ms = poll_ms
        self.start_at_eof = start_at_eof

        self._q = Queue()
        self._stop = threading.Event()
        self._watcher_th = None
        self._speaker_th = None
        self._file_offset = 0
        self._carryover = b""
        self._enc = "utf-8"

    # ---------- public API ----------
    def start(self):
        self._stop.clear()
        # Initialize file offset
        try:
            size = os.path.getsize(self.inbox_path)
        except FileNotFoundError:
            # Create the file if it doesn't exist yet
            open(self.inbox_path, "ab").close()
            size = 0
        self._file_offset = size if self.start_at_eof else 0
        self._carryover = b""

        self._watcher_th = threading.Thread(target=self._watcher_loop, name="InboxWatcher", daemon=True)
        self._speaker_th = threading.Thread(target=self._speaker_loop, name="InboxSpeaker", daemon=True)
        self._watcher_th.start()
        self._speaker_th.start()
        print(f"TTS: inbox watcher started (tail={'EOF' if self.start_at_eof else 'BOF'}), poll={self.poll_ms}ms, path={self.inbox_path}")

    def stop(self, timeout=2.0):
        self._stop.set()
        if self._watcher_th:
            self._watcher_th.join(timeout=timeout)
        if self._speaker_th:
            self._speaker_th.join(timeout=timeout)
        # Safety: ensure capture is not left paused
        try:
            self.resume_fn()
        except Exception:
            pass
        print("TTS: inbox watcher stopped")

    # ---------- internals ----------
    def _watcher_loop(self):
        """
        Poll the file for appended bytes and enqueue complete lines (ending with \n).
        """
        while not self._stop.is_set():
            try:
                # Detect rotation/truncation
                try:
                    size = os.path.getsize(self.inbox_path)
                except FileNotFoundError:
                    # File was deleted; re-create and continue
                    open(self.inbox_path, "ab").close()
                    size = 0

                if size < self._file_offset:
                    # Truncated/rotated: jump to new EOF (skip old content)
                    self._file_offset = size
                    self._carryover = b""

                if size > self._file_offset:
                    # Read appended bytes
                    with open(self.inbox_path, "rb", buffering=0) as f:
                        f.seek(self._file_offset)
                        data = f.read(size - self._file_offset)
                    self._file_offset = size

                    # Prepend any partial carryover and split on newlines
                    buf = self._carryover + data
                    lines = buf.split(b"\n")

                    # All but last element are complete lines
                    for raw in lines[:-1]:
                        line_str = raw.decode(self._enc, errors="replace").strip()
                        if not line_str:
                            continue

                        # Extract the 4th quoted string (the message text) by splitting on "
                        # Example:
                        #   {"incident_id": "incident_20251201_165026", "text": "Save this info from here."}
                        #   parts = ['{', 'incident_id', ': ', 'incident_20251201_165026', ', ', 'text', ': ', 'Save this info from here.', '}']
                        #   -> parts[7] == 'Save this info from here.'
                        parts = line_str.split('"')
                        if len(parts) >= 8:
                            speak_text = parts[7].strip()
                        else:
                            # Fallback: if format is unexpected, just use the whole line
                            speak_text = line_str

                        if speak_text:
                            self._q.put(speak_text)
                            print(f"TTS: queued (+1) len={len(speak_text)} queue={self._q.qsize()}")

                    # Last element might be partial (no newline yet)
                    self._carryover = lines[-1]



                time.sleep(self.poll_ms / 1000.0)

            except Exception as e:
                # Non-fatal; keep trying
                print(f"TTS: watcher error: {e}")
                time.sleep(0.5)

    def _speaker_loop(self):
        """
        Consume queued lines FIFO; pause capture -> speak (with chunking) -> resume capture.
        """
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.25)
            except Empty:
                continue

            try:
                chunks = self._chunk_text(text, CHUNK_SOFT_MAX_CHARS, CHUNK_HARD_MAX_CHARS)
                print(f"TTS: speaking len={len(text)} chunks={len(chunks)}")
                t0 = time.perf_counter()
                self.pause_fn()
                print("TTS: capture_paused")

                for i, c in enumerate(chunks, 1):
                    t_chunk0 = time.perf_counter()
                    self.speak_fn(c)     # BLOCKING call until audio done
                    dt = time.perf_counter() - t_chunk0
                    print(f"TTS: chunk {i}/{len(chunks)} done ({dt:.2f}s)")

                self.resume_fn()
                print("TTS: capture_resumed")
                dt_total = time.perf_counter() - t0
                print(f"TTS: done ({dt_total:.2f}s total), queue={self._q.qsize()}")

            except Exception as e:
                # According to your plan: do not retry, just log.
                print(f"TTS: speak error (no retry): {e}")
            finally:
                self._q.task_done()

    def _chunk_text(self, text, soft_max, hard_max):
        """
        Split into sentence-like chunks; then enforce soft and hard limits.
        """
        # Sentence-ish split
        parts = []
        for piece in _SENT_SPLIT_RE.split(text.strip()):
            if not piece:
                continue
            parts.append(piece)

        if not parts:
            return []

        # Merge to respect soft_max, then hard wrap if needed
        merged = []
        cur = ""
        for p in parts:
            if not cur:
                cur = p
                continue
            if len(cur) + 1 + len(p) <= soft_max:
                cur = f"{cur} {p}"
            else:
                merged.append(cur)
                cur = p
        if cur:
            merged.append(cur)

        # Hard wrap any that exceed hard_max
        final = []
        for m in merged:
            if len(m) <= hard_max:
                final.append(m)
                continue
            # wrap on whitespace near hard_max
            s = m
            while len(s) > hard_max:
                cut = s.rfind(" ", 0, hard_max)
                if cut == -1:
                    cut = hard_max
                final.append(s[:cut].strip())
                s = s[cut:].lstrip()
            if s:
                final.append(s)
        return final

# ========================= Main ============================
# Description: End-to-end entry point: sets up translation/TTS, diarizer, workers, mic loop, and handles shutdown.
def main(get_lang_mode=None):
    global STOP_REQUESTED

    # --- Button startup: initialize GPIO and wait for first press ---
    _init_button()
    _wait_for_start_button()
    state = TranslationState(mode_on=TRANSLATION_MODE_DEFAULT)
    tts = TTSQueue(
        speed=int(os.environ.get("TTS_SPEED_WPM", "170")),
        safe_print=safe_print,
    )

    translator = Translator(
        impl=os.environ.get("TRANSLATION_LIB", "argos"),
        safe_print=safe_print,
    )

    # >>> This is the constructor line you asked about <<<
    # Put it EXACTLY here, right after creating `translator`:
    postproc = PostWhisperProcessor(
        state,
        tts,
        translator,
        safe_print=safe_print,
        get_lang_mode=get_lang_mode,   # <-- hook in the button state
    )

    # Keep the rest of main unchanged; just keep `postproc` in scope.
    # On shutdown:
    _bind_inbox_speak(tts)
    #print("INBOX abs:", os.path.abspath(INBOX_PATH))
    
    diar = LiteDiarizer()
    workers = WhisperWorkers(NUM_WHISPER_WORKERS, postproc=postproc)
    
    pending: Optional[PendingSegment] = None

    MAX_MERGED_SEC = 25.0   # target max length for each Whisper call
    MAX_GAP_SEC    = .75    # if silence gap is longer, start a new job
    MAX_GAP_FRAMES = int(MAX_GAP_SEC * 1000 / FRAME_MS)

    # Hard safety cap so Whisper never sees > ~30 s
    MAX_CLIP_SEC = 28.0
    # Description: Callback invoked whenever a speech segment ends; enqueues it for transcription.
    def on_segment_closed(speaker_id, pcm_bytes, t0, t1, reason):
        nonlocal pending

        # Duration of this micro-segment
        seg_frames = t1 - t0
        seg_sec = seg_frames * (FRAME_MS / 1000.0)

        # If nothing pending, start a new buffer
        if pending is None:
            pending = PendingSegment(
                speaker_id=speaker_id,
                pcm_bytes=bytearray(pcm_bytes),
                t0_frame=t0,
                t1_frame=t1,
                reason=reason,
            )
            return

        # Compute time gap from previous end to this start
        gap_frames = t0 - pending.t1_frame
        gap_sec = gap_frames * (FRAME_MS / 1000.0)

        # --- Compute merged length BEFORE deciding what to do ---
        merged_frames = (
            (pending.t1_frame - pending.t0_frame)
            + max(gap_frames, 0)
            + seg_frames
        )
        merged_sec = merged_frames * (FRAME_MS / 1000.0)

        # --- NEW: Hard flush if adding this would exceed MAX_CLIP_SEC ---
        if merged_sec >= MAX_CLIP_SEC:
            # Flush the pending segment now
            workers.submit(Job(
                pending.speaker_id,
                bytes(pending.pcm_bytes),
                pending.t0_frame,
                pending.t1_frame,
                pending.reason,
            ))

            # Start a NEW pending segment using only the current piece
            pending = PendingSegment(
                speaker_id=speaker_id,
                pcm_bytes=bytearray(pcm_bytes),
                t0_frame=t0,
                t1_frame=t1,
                reason=reason,
            )
            return

        # Otherwise follow normal merge rules
        can_merge = (
            speaker_id == pending.speaker_id and
            gap_frames >= 0 and
            gap_frames <= MAX_GAP_FRAMES and
            merged_sec <= MAX_MERGED_SEC
        )

        if can_merge:
            # Insert silence if needed
            if gap_frames > 0:
                silence = np.zeros(gap_frames * FRAME_SAMPLES, dtype=np.int16).tobytes()
                pending.pcm_bytes.extend(silence)

            pending.pcm_bytes.extend(pcm_bytes)
            pending.t1_frame = t1

        else:
            # Flush pending segment
            workers.submit(Job(
                pending.speaker_id,
                bytes(pending.pcm_bytes),
                pending.t0_frame,
                pending.t1_frame,
                pending.reason,
            ))

            # Start new pending
            pending = PendingSegment(
                speaker_id=speaker_id,
                pcm_bytes=bytearray(pcm_bytes),
                t0_frame=t0,
                t1_frame=t1,
                reason=reason,
            )


    diar.on_segment_closed = on_segment_closed

    src = make_source()
    safe_print(f"Input mode: {INPUT_MODE} | WAV_GLOB={WAV_GLOB} | realtime={WAV_RT} | loop={WAV_LOOP}")

    start_inbox_tts()
        # Start background watcher for long-press "end incident" + shutdown
    _start_button_long_press_thread()

    try:
        with src as s:
            for frame in s.frames():  # source-specific frames (ADC may be 44.1 kHz here)
                if STOP_REQUESTED:
                    safe_print("[MAIN] Stop requested; breaking capture loop.")
                    break
                if CAPTURE_PAUSED:
                    continue   # skip feeding frames while TTS is speaking

                # If input is from MCP3201 ADC (44.1 kHz), resample each 20 ms
                # frame down to the diarizer/Whisper rate (16 kHz).
                if INPUT_MODE == "adc":
                    frame = resample_frame_to_diarizer_rate(frame)

                # After this point, `frame` must be 16 kHz, 20 ms, int16 PCM bytes:
                #   len(frame) == FRAME_BYTES
                diar.process_frame(frame)
    except KeyboardInterrupt:
        safe_print("\nStopping...")

    finally:
        # No mic.close() here; handled by the context manager above
        # Flush any leftover merged segment
        try:
            if pending is not None and len(pending.pcm_bytes) > 0:
                workers.submit(Job(
                    pending.speaker_id,
                    bytes(pending.pcm_bytes),
                    pending.t0_frame,
                    pending.t1_frame,
                    pending.reason,
                ))
        except NameError:
            # pending not defined if something blew up super early
            pass

        workers.q.join()
        workers.close()
        stop_inbox_tts()
        tts.close()

        # GPIO cleanup
        if GPIO is not None:
            try:
                GPIO.cleanup()
            except Exception as e:
                safe_print(f"[BUTTON] GPIO cleanup error: {e}")

        safe_print("Stopped.")


if __name__ == "__main__":
    main()
