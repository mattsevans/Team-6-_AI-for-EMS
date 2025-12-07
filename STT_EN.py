"""
Sean Bolger NLP for AI wearable device
STT_EN 
Various STT functions all built on assisting the combination STT to work properly
Includes Post processing features (in progress)
get_model - initializes the proper STT model
Initialize_recognizer - initializes recognizer for beginning of code
get_cst - outputs current time in a string
Diarize_audio - recognizes the current speaker
make_wav - produces a wav file. useful for diarization and TTS

after these functions is a code that uses just the english model and can be used for STT testing using onboard mic
"""
import datetime #provides access to current time
import pytz #assigns time zone
import json
import pyaudio 
from vosk import Model, KaldiRecognizer
import wave #used for wav file generation

import numpy as np
import librosa #speaker diarization imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import spacy
from textblob import TextBlob


# Load the spaCy English model
def initializeSpaCy():
    nlp = spacy.load('en_core_web_sm')
    return(nlp)

nlp = initializeSpaCy()

def post_process_text(alternatives: str) -> str:
    # Process each alternative with spaCy.
    scores = []
    for alt in alternatives: #cycles through all alternatives
        text = alt.get("text", "") #saves current alternative
        doc = nlp(text) #runs SpaCy NLP
        score = len(doc) #creates a score & assigns it to the given alternative 
        scores.append((score, text))
    best_candidate = max(scores, key=lambda x: x[0])[1] #selects best score as text output
    
    return best_candidate #returns best candidate determined by spacy

def make_wav(raw_audio, filename="temp_audio.wav"):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1) #mono audio
        wf.setsampwidth(2) #16-bit audio has 2 bytes per sample
        wf.setframerate(16000) #sample rate
        wf.writeframes(raw_audio)
    return filename

#initialize model
def get_model(language: str) -> Model:
    #Return the appropriate Vosk model based on the selected language
    if language == "english":
        return Model("vosk-model-en-US-0.22")  # Replace with the actual path for English
    elif language == "spanish":
        return Model("vosk-model-es-0.42") # Replace with the actual path for Spanish
    elif language == "chinese":
        return Model("vosk-model-cn-0.22") # Replace with the actual path for Chinese
    else:
        return Model("vosk-model-en-US-0.22")#if improper input it defaults to English

def initialize_recognizer():    
    model = get_model()
    # initialize the recognizer
    recognizer = KaldiRecognizer(model, 16000)

    #capture audio from the on computer microphone
    mic = pyaudio.PyAudio() 
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    stream.start_stream()

#formats the time in the conversation for proper output. is called for every iteration that needs to be said.
def get_cst_time() -> str:
    tz = pytz.timezone("America/Chicago") #sets time zone
    now = datetime.datetime.now(tz) #aquires current time
    return now.strftime("[%H:%M:%S]") #returns current time

raw_audio = b"" #sets variable for collecting audio for speaker diarization
global_audio_buffer = b"" #variable for last 90 seconds of audio
last_log_time = None 
#takes audio value and determines which speaker is talking for that given segment
def diarize_audio(raw_audio):     
    global global_audio_buffer, last_log_time
    #add newest segment of audio to global audio buffer
    global_audio_buffer += raw_audio
    
    #limit global audio buffer to last 90 seconds of data
    max_duration = 90 #90 second max
    sample_rate = 16000 #frequency
    sample_width = 2 #16 bit audio
    max_bytes = max_duration * sample_rate * sample_width #calculates max bytes used for 90 seconds of buffer

    new_raw_duration = len(raw_audio) / (sample_width * sample_rate) #duration of most recent audio clip added

    #crops audio data to last 90 seconds. 
    #if the raw audio needing a speaker label is more than 90 seconds then the global audio buffer is set to that time
    if len(global_audio_buffer) > max_bytes and len(raw_audio) < max_bytes: 
        global_audio_buffer = global_audio_buffer[-max_bytes:]
    elif len(raw_audio) > max_bytes:
        max_bytes = new_raw_duration * sample_rate * sample_width
        global_audio_buffer = global_audio_buffer[-max_bytes:]

    audio_path = make_wav(global_audio_buffer) #makes audio into wav file

    #load audio for processing
    audio, sr = librosa.load(audio_path, sr=None) 

    #calculates Mel-Frequency Cepstral Coefficients
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)

    #normallizes Mel-frequency cepstral coefficients
    scaler = StandardScaler() 
    mfccs_scaled = scaler.fit_transform(mfcc.T)

    hop_length = 512 #constand for framerate

    #defines blank time so that It doesn't label deadspace as a speaker
    #Trims current audio clip to be all time with actual text in it
    nonsilent = librosa.effects.split(audio, top_db=40)
    speech_frame_mask = np.zeros(mfcc.shape[1], dtype=bool)
    for s,e in nonsilent:
        frame_index = np.floor(np.arange(mfcc.shape[1])*hop_length / sr * sr).astype(int)
        speech_frame_mask |= (frame_index >= s) & (frame_index < e)

    kmeans = KMeans(n_clusters=2)  # select 2 speakers. will always look for 2 voices
    labels = np.full(mfcc.shape[1], -1, dtype=int)   #produces a 0 or 1 for if ems or spkr1 is talking or a -1 for silence
    labels[speech_frame_mask] = kmeans.fit_predict(mfccs_scaled[speech_frame_mask])

    #calculates length of new segment and produces labels for it
    frames_per_sec = sr / hop_length
    new_frames_count = int(new_raw_duration * frames_per_sec)
    new_segment_labels = labels[-new_frames_count:]

    #produces labels from each second. can be compared with actual transcription for analysis
    tz = pytz.timezone("America/Chicago") #sets time zone
    curTime = datetime.datetime.now(tz)
    now = get_cst_time()
    if last_log_time is None or (curTime - last_log_time).total_seconds() >= 1:
        # format raw labels as comma‑separated ints
        vals = ",".join(str(int(x)) for x in new_segment_labels)
        line = f"{now} CST  |  labels=[{vals}]\n"
        with open("diarization_log.txt", "a") as f: #prints labels into a text file
            f.write(f'line\n')
        last_log_time = curTime
    #determine which speaker is recognized the most
    trimmed_labels  = new_segment_labels[new_segment_labels != -1]
    if len(trimmed_labels) > 0:
        speakerAvg = sum(trimmed_labels) / len(trimmed_labels) 
    else:
        speakerAvg = -1

    if speakerAvg >= 0.5: #returns value for which speaker was recognized more
        return("[SKR1]")
    elif speakerAvg < 0:
        return("silent") #returns silent if the program didn't work properly
    else:
        return("[EMS]")



"""
#Used to test STT functions seperate of the NLP test

model = Model("vosk-model-en-us-0.22")  #defines which model of vosk is being used
    # initialize the recognizer
recognizer = KaldiRecognizer(model, 16000)

#capture audio from the on computer microphone
mic = pyaudio.PyAudio() 
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
stream.start_stream()
recognizer.SetMaxAlternatives(3)


while True:
    data = stream.read(4096, exception_on_overflow=False)
    
    #save the raw audio data from each segment to be printed
    raw_audio += data

    if recognizer.AcceptWaveform(data):
        result_json = recognizer.Result()
        result = json.loads(result_json) #get the result from recognizer
        
        alternatives = result.get("alternatives", [])

        if alternatives:
            # Process each alternative with spaCy.
            scores = []
            for alt in alternatives:
                text = alt.get("text", "")
                doc = nlp(text)  # assuming you've loaded a spaCy model: nlp = spacy.load("en_core_web_sm")
                # For instance, you might prefer alternatives with more tokens or fewer unknown tokens.
                # Here, we'll use a simple heuristic: higher token count may mean a more complete sentence.
                score = len(doc)
                scores.append((score, text))
        wav_filename = make_wav(raw_audio)
        speaker_label = diarize_audio(raw_audio)

        #adds timestamp
        timestamp = get_cst_time()        
        
        best_candidate = max(scores, key=lambda x: x[0])[1]

        #prints formatted output.
        #Format: [time] [Speaker] audio text
        #EX: [02:15] [EMS1] audio text
        print(f"{timestamp} {speaker_label} {best_candidate}")

        #clears audio that has already been diarized
        raw_audio = b""       
"""
