import pyaudio
import numpy as np
import noisereduce as nr
import json
from vosk import Model, KaldiRecognizer

# Initialize the Vosk model and recognizer
model = Model("vosk-model-en-US-0.22")
recognizer = KaldiRecognizer(model, 16000)

# Initialize PyAudio to capture live audio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4096)
stream.start_stream()

print("Listening with noise reduction applied...")

while True:
    try:
        # Read a chunk of audio data
        data = stream.read(4096, exception_on_overflow=False)
        # Convert the byte buffer to a NumPy array of float32 samples
        audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        
        # Apply noise reduction using spectral gating.
        # Note: For best results, you might compute a noise profile from an initial silent period.
        reduced_audio = nr.reduce_noise(y=audio_np, sr=16000)
        
        # Convert the reduced audio back to int16 and then to bytes.
        reduced_audio_int16 = np.int16(reduced_audio)
        reduced_data = reduced_audio_int16.tobytes()
        
        # Feed the denoised audio chunk into Vosk.
        if recognizer.AcceptWaveform(reduced_data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                print("Recognized:", text)
        else:
            # Optionally, process partial results.
            partial = json.loads(recognizer.PartialResult())
            # Uncomment the next line to see partial results:
            # print("Partial:", partial.get("partial", ""))
    except KeyboardInterrupt:
        print("Exiting...")
        break

stream.stop_stream()
stream.close()
p.terminate()
