import whisper

# Load the Whisper model (choose "tiny", "base", "small", "medium", or "large")
model = whisper.load_model("base")

# Path to your .wav file
audio_path = r"C:\Users\bolge\CAPSTONE_FOLDER\AI_Wear_NLP\EN_Med_0.wav"

# Transcribe the audio
result = model.transcribe(audio_path, fp16=False)

# Print the recognized text
print("Transcription:")
print(result["text"])
