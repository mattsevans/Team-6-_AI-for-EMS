"""
Sean Bolger NLP for AI wearable device
Signal to noise ratio testing.
Tests audio dataset with multiple levels of noise added to the audio
outputs the word error rate to see what levels of sound are able to be put up with
"""
import numpy as np
import pandas as pd
from vosk import Model, KaldiRecognizer
import spacy
import re
from jiwer import wer
import json
import wave


# Load the dataset
df = pd.read_parquet("EnglishMedLargetrain.parquet")
print(df.columns.tolist())
print("Columns:", df.columns.tolist())
print(df.head(3))

# Load Vosk model
model = Model("vosk-model-en-US-0.22")  #defines which model of vosk is being used
recognizer = KaldiRecognizer(model, 16000)


def initializeSpaCy(): #function initializes post processing tool
    nlp = spacy.load('en_core_web_sm')
    return(nlp)

nlp = initializeSpaCy()

#starting file path number and snr level in dbs
snr_level = -5
path_num = 0

#sets all total word error rate values for different db levels. has them with and without post processing
wer_proc_neg5 = 0
wer_reg_neg5 = 0
wer_proc_0 = 0
wer_reg_0 = 0
wer_proc_5 = 0
wer_reg_5 = 0
wer_proc_10 = 0
wer_reg_10 = 0
wer_proc_15 = 0
wer_reg_15 = 0
wer_proc_20 = 0
wer_reg_20 = 0

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
    

while path_num < 10:
    #set expected text
    expected_text = df.at[path_num,"transcripts"]
    with open("SNR_Data_Log.txt", "a") as f: #writes file name in the data log
        f.write(f"EN_Med_{path_num}\n")
    while snr_level <= 20: #tests until snr is over 20 dbs
        current_path = f"En_Med_{path_num}_{snr_level}db.wav"
        raw_result, result = getSTTresult(current_path) # Transcribe with Vosk

        #formats post processed, raw and expected text outputs
        vosk_text = str(result).capitalize() + "."
        vosk_text = re.sub(r"-", " ", vosk_text)
        vosk_text = re.sub(r"[^\w\s]", "", vosk_text)

        raw_result = str(raw_result).capitalize() + "."
        raw_result = re.sub(r"-", " ", raw_result)
        raw_result = re.sub(r"[^\w\s]", "", raw_result)

        expected_text = re.sub(r"-", " ", expected_text)
        expected_text = re.sub(r"[^\w\s]", "", expected_text)

        # Compares expected vs actual text
        if expected_text == "" or vosk_text == "":
            error = 1.0  #returns 100% error rate if either are blank
            print("Empty string detected in either reference or hypothesis.")
        else:
            error = wer(expected_text, vosk_text)        
        errorReg = wer(expected_text, raw_result)

        if snr_level == -5: #adds to WER total for any given SNR
            wer_proc_neg5 = (error + wer_proc_neg5*(path_num)) / (path_num+1)
            wer_reg_neg5 = (errorReg + wer_reg_neg5*(path_num)) / (path_num+1)
        elif snr_level == 0:
            wer_proc_0 = (error + wer_proc_0*(path_num)) / (path_num+1)
            wer_reg_0 = (errorReg + wer_reg_0*(path_num)) / (path_num+1)
        elif snr_level == 5:
            wer_proc_5 = (error + wer_proc_5*(path_num)) / (path_num+1)
            wer_reg_5 = (errorReg + wer_reg_5*(path_num)) / (path_num+1)
        elif snr_level == 10:
            wer_proc_10 = (error + wer_proc_10*(path_num)) / (path_num+1)
            wer_reg_10 = (errorReg + wer_reg_10*(path_num)) / (path_num+1)
        elif snr_level == 15:
            wer_proc_15 = (error + wer_proc_15*(path_num)) / (path_num+1)
            wer_reg_15 = (errorReg + wer_reg_15*(path_num)) / (path_num+1)
        elif snr_level == 20:
            wer_proc_20 = (error + wer_proc_20*(path_num)) / (path_num+1)
            wer_reg_20 = (errorReg + wer_reg_20*(path_num)) / (path_num+1)
        with open("SNR_Data_Log.txt", "a") as f: #updates text file with data
            f.write(f"{snr_level}\n")
            f.write(f"Proc WER: {error}\n")
            f.write(f"Reg WER: {errorReg}\n")
        #test each file with their relative expected text
        #create a text file with each WER and snr level
        #calculate avg WER per DB level
        snr_level += 5 #increments snr level by 5

    
    path_num += 1 #increases path to next audio file
    snr_level = -5 #resets SNR level

#prints total outputs for given datasets
print(f"-5 db proc wer: {wer_proc_neg5}")
print(f"-5 db reg wer: {wer_reg_neg5}")
print(f"0 db proc wer: {wer_proc_0}")
print(f"0 db reg wer: {wer_reg_0}")
print(f"5 db proc wer: {wer_proc_5}")
print(f"5 db reg wer: {wer_reg_5}")
print(f"10 db proc wer: {wer_proc_10}")
print(f"10 db reg wer: {wer_reg_10}")
print(f"15 db proc wer: {wer_proc_15}")
print(f"15 db reg wer: {wer_reg_15}")
print(f"20 db proc wer: {wer_proc_20}")
print(f"20 db reg wer: {wer_reg_20}")