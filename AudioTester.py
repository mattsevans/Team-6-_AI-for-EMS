# mic_record_test.py
# Record audio from your mic at 16 kHz mono (16-bit) and save to WAV.

import os
import sys
import wave
import time
import pyaudio
import numpy as np

RATE = 16000       # sample rate (Hz) — matches your pipeline
CHANNELS = 1
SAMPWIDTH = 2      # bytes per sample (16-bit)
CHUNK = 1024       # frames per buffer (lower = lower latency)
SECONDS = 5        # recording duration
OUTPUT = "mic_test.wav"  # output file

def list_input_devices():
    pa = pyaudio.PyAudio()
    print("\nAvailable input devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            sr = int(info.get("defaultSampleRate", 0))
            print(f"  [{i}] {info.get('name')}  (default SR={sr} Hz)")
    pa.terminate()
    print()

def record(device_index=None, seconds=SECONDS, rate=RATE, out_path=OUTPUT):
    pa = pyaudio.PyAudio()
    kwargs = dict(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=rate,
        input=True,
        frames_per_buffer=CHUNK
    )
    if device_index is not None:
        kwargs["input_device_index"] = device_index

    print(f"Opening stream (device={device_index}, rate={rate} Hz, {CHANNELS}ch)...")
    stream = pa.open(**kwargs)

    print(f"Recording {seconds} seconds... speak now.")
    frames = []
    n_chunks = int(rate / CHUNK * seconds)
    for _ in range(n_chunks):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))

    stream.stop_stream()
    stream.close()
    pa.terminate()

    pcm_bytes = b"".join(frames)
    # quick audio sanity: RMS / peak
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt((pcm**2).mean())) if pcm.size else 0.0
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    print(f"RMS={rms:.1f}, peak={peak:.0f}")

    # write WAV (16-bit mono)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)

    print(f"Saved: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    # 1) List devices so you can pick the right input
    list_input_devices()
    choice = input("Enter input device index (blank for default): ").strip()
    dev_idx = int(choice) if choice else None

    try:
        secs = input(f"Seconds to record [{SECONDS}]: ").strip()
        secs = int(secs) if secs else SECONDS
    except ValueError:
        secs = SECONDS

    try:
        record(device_index=dev_idx, seconds=secs, rate=RATE, out_path=OUTPUT)
    except OSError as e:
        print(f"\n[Error] {e}\n"
              "Tips:\n"
              " • Make sure the device index is valid.\n"
              " • Check Windows microphone privacy settings (allow desktop apps).\n"
              " • In Sound Settings, verify the correct input device and that the meter moves when you speak.\n"
              " • If you still see errors, try a different device index.\n")
        sys.exit(1)
