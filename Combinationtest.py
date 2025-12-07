import threading
import queue
import time
import json
import pyaudio
from vosk import Model, KaldiRecognizer
from STT_EN import diarize_audio, get_cst_time, make_wav, get_model
from LanguageTranslationFuncs import CN_to_EN_Text, EN_to_CN_Text, EN_to_ES_Text, ES_to_EN_Text
from TextToSpeech import initialize_TTS, Generate_TTS_SpeakerOut


#Sets up initializations and initial definition values for variables 
current_language = "english"
translation_target = None
command_queue = queue.Queue()
client = initialize_TTS()

def input_thread():
    #Thread function for terminal input text commands.
    global current_language, translation_target #selects needed global variables

    while True:
        cmd = input("Enter language command (english/spanish/chinese or translate to spanish/translate chinese/translate none): ").strip().lower()
        cmd_low = cmd.lower()
        if cmd_low in ("english", "spanish", "chinese"): #if a language is entered it changes the STT vosk model
            # Change the STT model.
            command_queue.put(("model", cmd_low))
        elif cmd_low in ("tranes", "trancn", "trannone"): #if a translation keyword is entered it adds on or removes translation feature.
            # Set the translation target based on the one-word command.
            if cmd_low == "tranes":
                command_queue.put(("translate", "spanish"))
            elif cmd_low == "trancn":
                command_queue.put(("translate", "chinese"))
            elif cmd_low == "trannone":
                command_queue.put(("translate", "none"))
        else:
            print("Invalid command. Try: english, spanish, chinese, or translate <target>.")

#main function of the project
#initilizes selected model and produces the proper type of output
def speech_to_text_loop(language: str) -> str:
    global translation_target
    #Continuously listen and print recognized text until a language switch command is received.
    model = get_model(language) #initialize model
    recognizer = KaldiRecognizer(model, 16000)
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000,
                      input=True, frames_per_buffer=4096)
    stream.start_stream()
    raw_audio = b""


    print(f"\n*** Listening in {language.upper()} mode... ***\n") #notifies when a new model has been initialized

    new_language = language  # Default to current language if no change
    
    try:
        while True:
            #Check for any terminal command to change language
            #works with input_thread to switch models when prompted
            #passes if there is no entry submitted
            try:
                cmd_type, cmd_value = command_queue.get_nowait()
                if cmd_type == "model" and cmd_value != language:
                    print(f"Switching STT model to {cmd_value.upper()} mode.\n")
                    new_language = cmd_value
                    break
                elif cmd_type == "translate":
                    if cmd_value == "none":
                        translation_target = None
                        print("Disabling translation.\n")
                    else:
                        translation_target = cmd_value
                        print(f"Enabling translation to {translation_target.upper()}.\n")
            except queue.Empty:
                pass

            #Read audio data and process it with Vosk
            data = stream.read(4096, exception_on_overflow=False)
            raw_audio += data
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    #generates .wav for diarization
                    wav_filename = make_wav(raw_audio)
                    speaker_label = diarize_audio(raw_audio)

                    #assigns current time
                    timestamp = get_cst_time()        
                
                    #prints formatted output.
                    #Format: [time] [Speaker] audio text
                    #EX: [02:15:23] [EMS1] audio text
                    full_output = f"{timestamp} {speaker_label} {text}"
                    print(full_output)

                    #Checks translation target language. if true it will add a the translated text and emit through onboard speakers
                    if language == "english" and translation_target == "spanish":
                        translated_text = EN_to_ES_Text(text)
                        Generate_TTS_SpeakerOut(translated_text, client, True, "es_MX")
                        print("Translation (Spanish):", translated_text)
                    elif language == "english" and translation_target == "chinese":
                        translated_text = EN_to_CN_Text(text)
                        Generate_TTS_SpeakerOut(translated_text, client, True, "zh_CN")
                        print("Translation (Chinese):", translated_text)

                    #Checks language. If it's not in english it automatically translates it to english and emits through onboard speakers
                    if language == "spanish":
                        es_en = ES_to_EN_Text(text)
                        Generate_TTS_SpeakerOut(es_en, client, True, "en_US")
                        print(es_en) #change back to full_output
                    if language == "chinese":
                        cn_en = CN_to_EN_Text(text)
                        Generate_TTS_SpeakerOut(cn_en, client, True, "en_US")
                        print(cn_en) #change back to full_output
                    #clears audio that has already been diarized and printed
                    raw_audio = b"" 

    finally:
        stream.stop_stream()
        stream.close()
        mic.terminate()
    
    return new_language


# Start the input thread so terminal commands can be received in parallel.
input_thread_instance = threading.Thread(target=input_thread, daemon=True)
input_thread_instance.start()

# Main loop: start with English and then switch based on terminal commands.
current_language = "english"
while True:
    # The speech_to_text_loop returns when a language switch command is issued.
    current_language = speech_to_text_loop(current_language)
