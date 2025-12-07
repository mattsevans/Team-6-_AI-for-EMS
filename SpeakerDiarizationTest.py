import time
import threading
from collections import deque
import keyboard   # pip install keyboard
import pyaudio
from STT_EN import diarize_audio, make_wav

# shared state
current_truth = None       # last key pressed ("0" or "1")
log = deque()              # li111111111111111111111111111111st of (predicted, truth) for each chunk

def on_key(event):
    global current_truth
    if event.event_type == "down" and event.name in ("0", "1"):
        current_truth = event.name

# hook keyboard
keyboard.hook(on_key)

# audio setup (match your stream settings)
pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paInt16,
                 channels=1,
                 rate=16000,
                 input=True,
                 frames_per_buffer=1024)
stream.start_stream()

print("Press 0 for EMS, 1 for Spkr1. Ctrl‑C to stop.")

try:
    while True:
        # 1) grab one chunk of raw audio
        raw = stream.read(1024, exception_on_overflow=False)

        # 2) diarize
        pred = diarize_audio(raw)   # returns "[EMS]" or "[SKR1]"

        # 3) map truth key to same tags
        if current_truth is None:
            # no label yet, skip
            continue
        truth = "[EMS]" if current_truth == "0" else "[SKR1]"

        # 4) record
        log.append((pred, truth))

        # (optional) print running accuracy every 50 chunks
        if len(log) % 50 == 0:
            correct = sum(1 for p, t in log if p == t)
            acc = correct / len(log)
            print(f"Chunks: {len(log):4d}, Accuracy: {acc:.1%}")

        # small sleep so we don’t spin too fast
        time.sleep(0.01)

except KeyboardInterrupt:
    # tear down
    stream.stop_stream()
    stream.close()
    pa.terminate()

    # final accuracy & approximate DER
    total = len(log)
    correct = sum(1 for p, t in log if p == t)
    acc = correct / total
    der_approx = 1 - acc
    print(f"\nTotal chunks: {total}")
    print(f"Accuracy:    {acc:.1%}")
    print(f"Approx DER:  {der_approx:.1%}")
