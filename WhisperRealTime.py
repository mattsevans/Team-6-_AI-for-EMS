import os
import wave
import tempfile
import pyaudio
import whisper

# ---------- Settings ----------
MODEL_NAME = "base"       # "tiny" or "base" for faster CPU use; "small" for better accuracy
RATE = 16000              # Whisper expects 16 kHz mono
CHANNELS = 1
CHUNK = 1024              # Frames per read
BLOCK_SEC = 5             # Seconds per transcribed chunk (lower = faster but choppier)
# ------------------------------

def save_wav(frames, path, rate=RATE, channels=CHANNELS, sampwidth=2):
    """Write PCM frames to a WAV file."""
    """
    Save a list of raw Pulse control modulation byte-strings (`frames`) as a WAV file.
    sampwidth=2 -> 16-bit PCM (matches pyaudio.paInt16 below)
    frames is a list like [b'....', b'....', ...] that we join and write.
    """
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)     # 2 bytes = 16-bit
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

def main():
    #Load Whisper once up-front 
    #fp16=False 
    print("Loading Whisper model...")
    model = whisper.load_model(MODEL_NAME)

    #Initialize PyAudio
    pa = pyaudio.PyAudio()
    try:
        #Open the microphone stream.
        #format: 16-bit signed integers, 1 channel(mono), rate: 16000 samples/sec, input=True: this is an INPUT stream,frames_per_buffer defined in settings

        print("Listening... Press Ctrl+C to stop.\n")
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        #(RATE * BLOCK_SEC / CHUNK) total frames / chunck gets read iteration
        frames_per_block = int(RATE / CHUNK * BLOCK_SEC)

        while True:
            frames = []
            for _ in range(frames_per_block):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

            # writes temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
            save_wav(frames, tmp_path)

            # Transcribe (fp16=False for CPU)
            result = model.transcribe(tmp_path, fp16=False)
            text = result.get("text", "").strip()
            if text:
                print(text)

            # Clean up the temp file
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        pa.terminate()

if __name__ == "__main__":
    main()
