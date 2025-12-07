"""
Sean Bolger NLP for AI wearable device
SpeechToTextTester
Tests speech to text for different models. 
uses with audio datasets instead of mic like in the NLPtest
"""
import json
from vosk import Model, KaldiRecognizer
from jiwer import wer
import pandas as pd
import wave
import os
from tqdm import tqdm
import spacy
import re

#STT testing w post processing

# Load the dataset
df = pd.read_parquet("EnglishMedLargetrain.parquet")

# used to check file beggining to see it's layout
print("Columns:", df.columns.tolist())
print(df.head(3))

# Load Vosk model
model = Model("vosk-model-en-US-0.22")  #defines which model of vosk is being used
recognizer = KaldiRecognizer(model, 16000)

tot_vosk_error = 0 #assigns total wer value to start at 0
tot_spacy_error = 0 

def initializeSpaCy():
    nlp = spacy.load('en_core_web_sm')
    return(nlp)

nlp = initializeSpaCy()

# Opens the parquet file to get the expected text output and transcribes audio file
def getSTTresult(wav_path):
    with wave.open(wav_path, "rb") as wf:
        #checks channel amount, sample width, and framerate
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000]:
            raise ValueError("Audio file must be mono, 16-bit, with a sample rate of 8000, 16000, or 32000 Hz")

        # Initialize Vosk recognizer
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetMaxAlternatives(3) #assigns max of 3 alternatives


        # Read audio in chunks and process
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)

        # Get final transcription result
        result = json.loads(rec.FinalResult())
        #result_json = rec.Result()
                #print("DEBUG:", result_json)
        #result = json.loads(result_json)
        
        
        alternatives = result.get("alternatives", [])
        raw_text = result["alternatives"][0].get("text", "")
        print(f"Raw text: {raw_text}")

        if alternatives:
            # Process each alternative with spaCy.
            scores = []
            for alt in alternatives:
                text = alt.get("text", "")
                    #print(f"Alternatives {text}")
                doc = nlp(text)  # assuming you've loaded a spaCy model: nlp = spacy.load("en_core_web_sm")
                # For instance, you might prefer alternatives with more tokens or fewer unknown tokens.
                # Here, we'll use a simple heuristic: higher token count may mean a more complete sentence.
                score = len(doc)
                scores.append((score, text))
            best_candidate = max(scores, key=lambda x: x[0])[1]
        else:
            best_candidate = result.get("text", "")
        return raw_text, best_candidate
    

with tqdm(total=300) as pbar: #max 445 files
    for index, row in df.iterrows():
        expected_text = row["transcripts"]

        #expected_text = "Echocardiogram revealed a significant left ventricular dysfunction"
        wav_path = f"EN_Med_{index}.wav" #change of wav file over different tests
        
        raw_result, result = getSTTresult(wav_path)
        # Transcribe with Vosk
        vosk_text = str(result).capitalize() + "."
        vosk_text = re.sub(r"-", " ", vosk_text)
        vosk_text = re.sub(r"[^\w\s]", "", vosk_text)

        raw_result = str(raw_result).capitalize() + "."
        raw_result = re.sub(r"-", " ", raw_result)
        raw_result = re.sub(r"[^\w\s]", "", raw_result)

        expected_text = re.sub(r"-", " ", expected_text)
        expected_text = re.sub(r"[^\w\s]", "", expected_text)
        #Compares expected vs actual text
        if expected_text == "" or vosk_text == "":
            error = 1.0  # or use another marker like None or skip this file.
            print("Empty string detected in either reference or hypothesis.")
        else:
            error = wer(expected_text, vosk_text)        
        tot_spacy_error = (error + tot_spacy_error*index) / (index+1)
        
        errorReg = wer(expected_text, raw_result)
        tot_vosk_error = (errorReg + tot_vosk_error*index) / (index+1)
        
        print(f"Expected: {expected_text}") #switched from expected_text
        print(f"Vosk Output: {raw_result}") #switched to result from vosk_text
        print(f"Spacy Output: {vosk_text}") #switched to result from vosk_text
        print(f"WER: {error:.2%}\n")
        print(tot_spacy_error)

        pbar.update(1)

        if index == 300:
            break


print(f"Total Processed WER: " + str(tot_spacy_error)) # prints total program error once all tests are complete
print(f"Total Vosk WER: " + str(tot_vosk_error)) # prints total program error once all tests are complete
