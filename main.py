from TextToSpeech import initialize_TTS, Generate_TTS_SpeakerOut
from STT_EN import get_cst_time, diarize_audio, make_wav
from LanguageTranslationFuncs import CN_to_EN_Text, EN_to_CN_Text, EN_to_ES_Text, ES_to_EN_Text
from Combinationtest import get_model, input_thread, speech_to_text_loop

def main():
    #initialize Text to speech
    client = initialize_TTS()

    #inintialize all functions needed

    #begin english stt model
        #state when file is starting initialization and when it starts listening
    #include post processing models here

    #show text response

    #switch feature to spanish of chinese

    #Translate to english
        #return english audio for listener
    
    Audio_STT_Out = "Hello, this is a test of english TTS"
    text1 = EN_to_CN_Text(Audio_STT_Out)
    print(text1)
    text1 = EN_to_ES_Text(Audio_STT_Out)
    print(text1)
    Generate_TTS_SpeakerOut(Audio_STT_Out, client, True, "en_US")
    
    Audio_STT_Out = "再见. 对不起. 你好吗？"
    text1 = CN_to_EN_Text(Audio_STT_Out)
    print(text1)
    Generate_TTS_SpeakerOut(Audio_STT_Out, client, True, "cmn_CN")

    Audio_STT_Out = "Adiós. Lo siento. ¿Cómo estás?"
    text1 = ES_to_EN_Text(Audio_STT_Out)
    print(text1)
    Generate_TTS_SpeakerOut(Audio_STT_Out, client, True, "es_US")

main()