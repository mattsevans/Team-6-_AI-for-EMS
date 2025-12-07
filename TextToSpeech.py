"""
Sean Bolger NLP for AI wearable device
LanguageTranslationFunctions
Produces 4 translation functions to translate text to and from english
Test of code is commented out at the bottom. these are called in the completed combination test
"""
from google.cloud import texttospeech
from google.oauth2 import service_account
import winsound

def initialize_TTS():
    #uses key to confirm credentials and initialize the text to speech client
    key_path = r"C:\Users\bolge\CAPSTONE_FOLDER\capstone-project-450622-5fa98331b05e.json"
    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = texttospeech.TextToSpeechClient(credentials=credentials)
    
    #returns client for the TTS Main function to utilize
    return client    

def Generate_TTS_SpeakerOut(audio_out_text, client, use_speakers, language_selected, output_file="STToutput.wav"):
    """
    Saves text as audio .wav file
    if use_speakers is true it plays .wav file through speakers

    Arguments:
        audio_out_text = text to be converted
        client = the text to speech client
        use_speakers = True if you want to play the .wav file, False if you just want to save the .wav file
        language_selected = can use to switch desired language
            "en-US" - for English
            "es-US" - for Spanish spoken like people in the US
            "cmn-CN" - for Chinese       
        output_file = the name for the output .wav file to be generated
    """

    #set up the text input request
    synthesis_input = texttospeech.SynthesisInput(text=audio_out_text)

    #select paremeters. english text and a male voice
    voice_params = texttospeech.VoiceSelectionParams(
        language_code= language_selected, 
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    #configurres TTS to create a wav file
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    #uses above definitions to produce the TSS response
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config
    )

    #saves wav file
    wav_filename = "STToutput.wav"
    with open(wav_filename, "wb") as out:
        out.write(response.audio_content)

    #checks if you want the wav file to be played or not
    if use_speakers:
        #plays the wav file
        winsound.PlaySound(wav_filename, winsound.SND_FILENAME)
