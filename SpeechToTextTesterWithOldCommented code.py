
import json
from vosk import Model, KaldiRecognizer
from jiwer import wer
import pandas as pd
import wave
import os
from tqdm import tqdm

# Load the dataset
df = pd.read_parquet("test-00000-of-00001.parquet")
# Load Vosk model
model = Model("vosk-model-en-us-0.22")  #defines which model of vosk is being used
recognizer = KaldiRecognizer(model, 16000)

tot_error = 0



# Open the audio file
def getSTTresult(wav_path):
    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000]:
            raise ValueError("Audio file must be mono, 16-bit, with a sample rate of 8000, 16000, or 32000 Hz")

        # Initialize Vosk recognizer
        rec = KaldiRecognizer(model, wf.getframerate())

        # Read audio in chunks and process
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)

        # Get final transcription result
        result = json.loads(rec.FinalResult())
        return result.get("text")
        #print("Transcription:", result.get("text", "No transcription found"))
with tqdm(total=445) as pbar:
    for index, row in df.iterrows():
        expected_text = row["transcripts"]

        #expected_text = "Echocardiogram revealed a significant left ventricular dysfunction"
        # Save audio as WAV
        wav_path = f"temp_{index}.wav" #change of wav file over different tests
        result = getSTTresult(wav_path)
        # Transcribe with Vosk
        vosk_text = str(result).capitalize() + "."

        # Compare expected vs. actual text
        error = wer(expected_text, vosk_text)
        tot_error = (error + tot_error*index) / (index+1)

        ''' USE TO SEE SPECIFIC OUTPUTS OF STT MODEL
        print(f"🎙️ Expected: {expected_text}")
        print(f"📝 Vosk Output: {vosk_text}")
        print(f"❌ WER: {error:.2%}\n")
        print(tot_error)  '''
        pbar.update(1)

print(f"Total WER: " + tot_error)
'''
def transcribe_audio(wav_path):
    recognizer = KaldiRecognizer(model, 16000)
    with open(wav_path, "rb") as audio_file:
        data = audio_file.read()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            return result["texts"]
    return ""

def save_audio_from_bytes(audio_bytes, wav_path):
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)  # 1 for mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(16000)  # Sample rate
        wav_file.writeframes(audio_bytes)
# Process each sample
for index, row in df.iterrows():
    audio_bytes = row["audio"]["bytes"]  # Adjust this key if needed
    expected_text = row["transcripts"]

    # Save audio as WAV
    wav_path = f"temp_{index}.wav"
    save_audio_from_bytes(audio_bytes, wav_path)

    # Transcribe with Vosk
    vosk_text = transcribe_audio(wav_path)

    # Compare expected vs. actual text
    error = wer(expected_text, vosk_text)

    print(f"🎙️ Expected: {expected_text}")
    print(f"📝 Vosk Output: {vosk_text}")
    print(f"❌ WER: {error:.2%}\n")
'''