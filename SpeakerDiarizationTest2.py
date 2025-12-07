import pyaudio
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import KMeans

# Audio stream parameters
CHUNK = 1024          # Frames per buffer
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000          # Sample rate; resemblyzer expects 16kHz audio

# Duration (in seconds) of each audio segment to process
SEGMENT_DURATION = 3.0
NUM_CHUNKS = int(RATE / CHUNK * SEGMENT_DURATION)

# Initialize PyAudio and open the microphone stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Initialize the voice encoder (loads a pre-trained model)
encoder = VoiceEncoder()

print("Recording... Press Ctrl+C to stop.")

# Buffer to store embeddings from each segment
embeddings_buffer = []

try:
    while True:
        print("Recording a new segment...")
        segment_frames = []
        # Record for the specified segment duration
        for _ in range(NUM_CHUNKS):
            data = stream.read(CHUNK, exception_on_overflow=False)
            # Convert byte data to a numpy array of floats
            frames = np.frombuffer(data, dtype=np.float32)
            segment_frames.append(frames)
        segment_audio = np.concatenate(segment_frames)
        
        # Preprocess the segment (this will also resample if needed)
        wav = preprocess_wav(segment_audio)
        
        # Get the speaker embedding for the current segment
        embedding = encoder.embed_utterance(wav)
        embeddings_buffer.append(embedding)
        
        # For demonstration, once we collect a few segments, cluster them.
        # In this example we cluster every 5 segments.
        if len(embeddings_buffer) >= 5:
            X = np.array(embeddings_buffer)
            # Assume 2 speakers; adjust n_clusters as needed
            kmeans = KMeans(n_clusters=2, random_state=0)
            labels = kmeans.fit_predict(X)
            print("Detected speaker labels for recent segments:", labels)
            
            # Here you could map each segment to its timestamp, combine labels with transcriptions, etc.
            # For simplicity, we clear the buffer and start over.
            embeddings_buffer = []
            
except KeyboardInterrupt:
    print("Stopped recording.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
 