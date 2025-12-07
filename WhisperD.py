import os
import sys
import time
import wave
import queue
import math
import json

import threading
from pathlib import Path
from typing import List, Dict
import tempfile
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyaudio
import webrtcvad
from python_speech_features import mfcc
import whisper

# ===================== Config =====================
RATE = 16000                 # 16 kHz mono
CHANNELS = 1
SAMPLE_WIDTH = 2             # 16-bit
FRAME_MS = 20                # VAD frame size (10, 20, or 30 ms); we pick 20 ms
FRAME_SAMPLES = int(RATE * FRAME_MS / 1000)      # 320 samples
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH       # 640 bytes

VAD_AGGRESSIVENESS = 2       # 0-3; higher is more aggressive (more "speech" filtered)
VAD_HANGOVER_MS = 150        # keep speech active this long after last voiced frame
VAD_HANGOVER_FRAMES = int(VAD_HANGOVER_MS / FRAME_MS)

# Diarizer settings (Lite)
EMB_WIN_SEC = 1.0            # embedding window
EMB_HOP_SEC = 0.5
EMB_WIN_FRAMES = int(EMB_WIN_SEC * 1000 / FRAME_MS)   # 50 frames
EMB_HOP_FRAMES = int(EMB_HOP_SEC * 1000 / FRAME_MS)   # 25 frames
EMB_N_MFCC = 24              # MFCC dims (20–26 reasonable)
SPK_SIM_THRESHOLD = 0.65     # cosine similarity threshold to create new speaker
SPK_CHANGE_CONFIRM_HOPS = 3  # require this many consecutive hops agreeing on a new speaker

# Segmenting
BOUNDARY_OVERLAP_MS = 250    # add to next/prev when closing segments
BOUNDARY_OVERLAP_FRAMES = int(BOUNDARY_OVERLAP_MS / FRAME_MS)

# Whisper
WHISPER_MODEL = "base"       # "tiny" is safest on Pi; try "base" if CPU headroom
WHISPER_FP16 = False         # no GPU on Pi; keep False

# Workers
NUM_WHISPER_WORKERS = 1

# ==================================================

def safe_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def bytes_to_np_int16(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16)

def np_int16_to_bytes(arr: np.ndarray) -> bytes:
    return arr.astype(np.int16).tobytes()

def write_wav(path: str, pcm_bytes: bytes, rate: int = RATE, channels: int = 1, sampwidth: int = SAMPLE_WIDTH):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

@dataclass
class Speaker:
    id: int
    centroid: np.ndarray
    count: int = 1

    def update(self, emb: np.ndarray):
        # online centroid update
        self.count += 1
        self.centroid = self.centroid + (emb - self.centroid) / self.count

class LiteDiarizer:
    """
    VAD -> rolling voiced frames -> MFCC embeddings over sliding windows -> online centroid assignment.
    Emits (speaker_id, segment_bytes, t0_frames, t1_frames) when a segment closes
    due to speaker change or speech -> silence.
    """
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.frame_idx = 0

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

    def _frame_is_speech(self, frame_bytes: bytes) -> bool:
        # WebRTC VAD wants 16-bit mono PCM at 8/16/32 kHz and frames of 10/20/30 ms
        return self.vad.is_speech(frame_bytes, RATE)

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

    def _assign_speaker(self, emb: np.ndarray) -> int:
        if not self.speakers:
            spk = Speaker(id=1, centroid=emb.copy())
            self.speakers.append(spk)
            return spk.id

        # find best match
        sims = [cosine_sim(emb, s.centroid) for s in self.speakers]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        best_spk = self.speakers[best_idx]

        if best_sim < SPK_SIM_THRESHOLD:
            # create new
            new_id = max(s.id for s in self.speakers) + 1
            spk = Speaker(id=new_id, centroid=emb.copy())
            self.speakers.append(spk)
            return spk.id
        else:
            # update centroid for matched spk
            best_spk.update(emb)
            return best_spk.id

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

    def process_frame(self, frame_bytes: bytes):
        """Push one 20 ms frame (FRAME_BYTES)"""
        assert len(frame_bytes) == FRAME_BYTES, "Expected exact 20ms frame bytes"

        self.recent_frames.append(frame_bytes)
        is_speech = self._frame_is_speech(frame_bytes)

      
        # Maintain voiced run with hangover
        if is_speech:
            self.voiced_frames.append(frame_bytes)
            self.voiced_mask.append(1)
            self.emb_cursor += 1
        else:
            # still append, but mark as non-speech; used for hangover counting
            self.voiced_frames.append(frame_bytes)
            self.voiced_mask.append(0)
            self.emb_cursor += 1

            # if we've seen enough trailing non-speech frames, close segment
            # Count trailing zeros
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


# ---------------------- Audio Capture ----------------------
class MicReader:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None

    def open(self):
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAME_SAMPLES
        )

    def read_frame(self) -> bytes:
        return self.stream.read(FRAME_SAMPLES, exception_on_overflow=False)

    def close(self):
        try:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
        finally:
            self.pa.terminate()

GLOSSARY_PATH = Path(r"C:\Users\bolge\CAPSTONE_FOLDER\AI_Wear_NLP\ems_glossary_seed.csv")


def load_glossary(path: path) -> Dict[str, any]:
    
    
        return json.load(f)

def pick_scene_pack(glossary, incident_type="general", extras=None) -> List[str]:
    # A tiny curated seed per incident (extend as you like)
    seeds = {
        "cardiac": ["chest pain","12-lead ECG","STEMI","ASA","nitroglycerin","ROSC",
                    "ventricular fibrillation","ventricular tachycardia","defibrillation","cardioversion","EKG","NTG"],
        "trauma": ["trauma alert","tourniquet","hemostatic gauze","pelvic binder",
                   "tension pneumothorax","needle decompression","occlusive dressing","C-spine","traction splint","hemorrhage"],
        "respiratory": ["respiratory distress","asthma","COPD","albuterol","ipratropium","CPAP","PEEP","ETCO2","SpO2","wheezing"],
        "neuro": ["stroke","TIA","Glasgow Coma Scale","GCS","last known well","Cincinnati stroke scale","BE FAST","PERRL"],
        "overdose": ["overdose","opioid","naloxone","Narcan","respiratory depression","pinpoint pupils","bag valve mask","BVM"],
        "general": ["vital signs stable","blood pressure","heart rate","oxygen saturation","pulse ox","primary survey","SAMPLE","OPQRST"]
    }
    base = seeds.get(incident_type, seeds["general"])[:20]
    pool = [w for w in glossary if w not in base]
    chosen = base + pool[:max(0, 120 - len(base))]  # keep it compact
    if extras:
        chosen = (chosen + list(extras))[:160]
    return chosen

def build_initial_prompt(scene_terms: List[str]) -> str:
    # Short, whitespace-separated phrases bias tokenization well
    # Keep it under ~200 words for stability
    return " ".join(scene_terms[:200])

# -------------------- Whisper Worker(s) --------------------
@dataclass
class Job:
    speaker_id: int
    pcm_bytes: bytes
    t0_frame: int
    t1_frame: int
    reason: str

class WhisperWorkers:
    def __init__(self, num_workers=1):
        self.q = queue.Queue(maxsize=64)
        safe_print("Loading Whisper model...")
        self.model = whisper.load_model(WHISPER_MODEL)
        self.threads: List[threading.Thread] = []
        self.stop_event = threading.Event()
        for i in range(num_workers):
            th = threading.Thread(target=self._worker, name=f"whisper-{i}", daemon=True)
            th.start()
            self.threads.append(th)

    def submit(self, job: Job):
        try:
            self.q.put_nowait(job)
        except queue.Full:
            safe_print("[WARN] Transcription queue full; dropping segment.")

    def _worker(self):
        glossary = load_glossary()
        scene_terms = pick_scene_pack(glossary, "general", None)
        initial_prompt = build_initial_prompt(scene_terms)
        while not self.stop_event.is_set():
            try:
                job: Job = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                # Write temp WAV and transcribe
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp_path = tmp.name
                write_wav(tmp_path, job.pcm_bytes, rate=RATE, channels=CHANNELS, sampwidth=SAMPLE_WIDTH)
                result = self.model.transcribe(tmp_path, 
                        fp16=WHISPER_FP16, 
                        temperature=0.0,
                        beam_size=5,
                        no_speech_threshold=0.5,
                        logprob_threshold=-1.0,
                        compression_ratio_threshold=2.4,
                        condition_on_previous_text=False,   # we send discrete chunks; keeps bias consistent
                        initial_prompt=initial_prompt,
                        word_timestamps=False)
                text = (result.get("text") or "").strip()
                os.remove(tmp_path)
                if text:
                    # Convert frames to seconds for display
                    t0_sec = job.t0_frame * (FRAME_MS / 1000.0)
                    t1_sec = job.t1_frame * (FRAME_MS / 1000.0)
                    safe_print(f"[{t0_sec:07.2f}–{t1_sec:07.2f}] Spk{job.speaker_id}: {text}")
            except Exception as e:
                safe_print(f"[Transcribe error] {type(e).__name__}: {e}")
            finally:
                self.q.task_done()

    def close(self):
        self.stop_event.set()
        for th in self.threads:
            th.join(timeout=0.5)


# ========================= Main ============================
def main():
    diar = LiteDiarizer()
    workers = WhisperWorkers(NUM_WHISPER_WORKERS)

    def on_segment_closed(speaker_id, pcm_bytes, t0, t1, reason):
        # Add boundary overlap on both sides (already included pre-segment via recent_frames)
        # Here we just ship the bytes we have.
        workers.submit(Job(speaker_id, pcm_bytes, t0, t1, reason))

    diar.on_segment_closed = on_segment_closed

    mic = MicReader()
    mic.open()
    safe_print("Listening with lite diarization... (Ctrl+C to stop)")
    try:
        while True:
            frame = mic.read_frame()  # 20 ms 16k mono int16
            diar.process_frame(frame)
    except KeyboardInterrupt:
        safe_print("\nStopping...")
    finally:
        mic.close()
        workers.close()
        safe_print("Stopped.")

if __name__ == "__main__":
    main()
