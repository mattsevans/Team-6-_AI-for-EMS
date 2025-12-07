import pandas as pd
import wave
from pathlib import Path

# Load the Parquet file
EMS_text_test = pd.read_parquet("ChineseTerms.parquet")

out_dir = Path("wav_outputs")
out_dir.mkdir(exist_ok=True)

for idx, row in EMS_text_test.iterrows():
    wav_bytes = row["audio"]["bytes"]
    out_path = out_dir / f"CN_Med_{idx}.wav"
    out_path.write_bytes(wav_bytes)
    print(f"✅ Saved {out_path}")

# (Optional) Quick header check on the first file
import wave
first = out_dir / "CN_Med_0.wav"
with wave.open(str(first), "rb") as wf:
    print("Channels:    ", wf.getnchannels())
    print("Sample rate: ", wf.getframerate())
    print("Sample width:", wf.getsampwidth(), "bytes")

"""
def save_audio_from_bytes(audio_bytes, wav_path):
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)  # 1 for mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(16000)  # Sample rate
        wav_file.writeframes(audio_bytes)

for index, row in EMS_text_test.iterrows():
    audio_bytes = row["audio"]["bytes"]  
    wav_path = f"CN_Med_{index}.wav" #change to change file name
    
    if isinstance(audio_bytes, bytes):  # Ensures it's actually raw audio data
        save_audio_from_bytes(audio_bytes, wav_path)
        print(f"✅ Saved {wav_path}")
    else:
        print(f"❌ Error: Row {index} does not contain valid audio bytes.")


import os

wav_path = "CN_Med_0.wav"  # Use a sample file from your dataset

if os.path.exists(wav_path):
    print(f"✅ File exists: {wav_path}")
    print(f"📏 File size: {os.path.getsize(wav_path)} bytes")
else:
    print("❌ WAV file was not saved correctly!")
"""