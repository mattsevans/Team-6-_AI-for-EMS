'''pip install -U pip setuptools wheel
pip install -U 'spacy[transformers,lookups]'
python -m spacy download en_core_web_sm

Sean Bolger 2/25/2025
SpaCy test file for nlp capstone subsystem'''

import spacy
import json
import pyaudio 
from vosk import Model, KaldiRecognizer

model = Model("vosk-model-en-us-0.22")  #defines which model of vosk is being used
recognizer = KaldiRecognizer(model, 16000)
mic = pyaudio.PyAudio() 
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
stream.start_stream()

print("🎙️ Listening... Speak into the microphone.")

while True:
    data = stream.read(4096, exception_on_overflow=False)
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        print(f"Recognized Speech: {result['text']}")

