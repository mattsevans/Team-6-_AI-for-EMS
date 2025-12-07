#!/usr/bin/env python3
"""
whisper_benchmark.py — simple one-file benchmark for whisper.cpp on Raspberry Pi.

- Uses a single WAV file
- Calls whisper.cpp once
- Measures wall-clock time
- Computes RTF = processing_time / audio_duration
"""

import os
import time
import subprocess
import wave
from pathlib import Path

# --------- CONFIG: EDIT THESE TO MATCH YOUR SETUP ---------

# Path to whisper.cpp binary (Release build) on the Pi
# If your binary is elsewhere, change this line.
WHISPER_CPP_BIN = os.environ.get(
    "WHISPER_CPP_BIN",
    "/home/team6/NLP-System/whisper.cpp/main"   # <-- adjust if needed
)

# Path to the model you want to test (tiny, base, etc.)
# You can also override this with the WHISPER_CPP_MODEL env var.
WHISPER_CPP_MODEL = os.environ.get(
    "WHISPER_CPP_MODEL",
    "/home/team6/NLP-System/whisper.cpp/models/ggml-tiny.en.bin"  # <-- change to base for base test
)

# Number of CPU threads to use
WHISPER_CPP_THREADS = int(os.environ.get("WHISPER_CPP_THREADS", "4"))

# WAV file to benchmark
TEST_WAV = os.environ.get(
    "WHISPER_BENCH_WAV",
    "/home/team6/NLP-System/AI_Wear_NLP/test_audio/recording8_16k.wav"  # <-- point to your test file
)

# Temporary output prefix (whisper.cpp will create files like this)
OUT_PREFIX = "/tmp/whisper_bench_out"

# Language (set to "en" for English)
LANG = os.environ.get("WHISPER_LANG", "en")

# ----------------------------------------------------------


def get_wav_duration_seconds(wav_path: str) -> float:
    """Return duration of WAV file in seconds."""
    with wave.open(wav_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def main():
    wav_path = Path(TEST_WAV)
    bin_path = Path(WHISPER_CPP_BIN)
    model_path = Path(WHISPER_CPP_MODEL)

    if not wav_path.exists():
        raise SystemExit(f"ERROR: WAV file not found: {wav_path}")

    if not (bin_path.exists() or shutil.which(str(bin_path))):
        # fall back to PATH search
        from shutil import which
        if which(str(bin_path)) is None:
            raise SystemExit(f"ERROR: whisper.cpp binary not found: {bin_path}")

    if not model_path.exists():
        raise SystemExit(f"ERROR: Model file not found: {model_path}")

    audio_sec = get_wav_duration_seconds(str(wav_path))

    print("=== Whisper.cpp Benchmark ===")
    print(f"Binary : {bin_path}")
    print(f"Model  : {model_path}")
    print(f"WAV    : {wav_path}")
    print(f"Threads: {WHISPER_CPP_THREADS}")
    print(f"Audio duration: {audio_sec:.3f} s")
    print()

    cmd = [
        str(bin_path),
        "-m", str(model_path),
        "-f", str(wav_path),
        "-l", LANG,
        "-t", str(WHISPER_CPP_THREADS),
        "-nt",              # no timestamps (optional; can speed things up)
        "-of", OUT_PREFIX,  # output prefix (we don't really care about the file)
        "-otxt",            # text output only
    ]

    print("Running command:")
    print(" ", " ".join(cmd))
    print()

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()

    proc_time = t1 - t0
    rtf = proc_time / audio_sec if audio_sec > 0 else float("inf")

    print("=== Results ===")
    print(f"Return code   : {result.returncode}")
    print(f"Processing time: {proc_time:.3f} s")
    print(f"RTF (proc/audio): {rtf:.3f} x slower than real-time")
    print()

    if result.stdout:
        print("--- whisper.cpp STDOUT (truncated) ---")
        print("\n".join(result.stdout.splitlines()[:10]))
        print("--------------------------------------")

    if result.stderr:
        print()
        print("--- whisper.cpp STDERR (truncated) ---")
        print("\n".join(result.stderr.splitlines()[:15]))
        print("--------------------------------------")


if __name__ == "__main__":
    import shutil
    main()
