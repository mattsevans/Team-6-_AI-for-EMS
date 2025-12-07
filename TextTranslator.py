#autodetect language
#translate based on language to english. if already english return english
from translate import Translator
from langdetect import detect
from langdetect import detect_langs
text0 = "权利 与 义务 与 尊严"
text1 = "比心, In lab now! Got a bunch of random tasks done! Also I got a confirmation back for my place this summer!"
text2 = "hi"
text3 = "Hola, Buenos dias"
lang0 = detect(text0)
langs0 = detect_langs(text0)
lang1 = detect(text1)
langs1 = detect_langs(text1)
lang3 = detect(text1)
langs3 = detect_langs(text1)

translatorChi = Translator(from_lang="zh", to_lang='en')
translationChi = translatorChi.translate(text0)
translation = translatorChi.translate(text2)
translationSpn = translatorChi.translate(text3)
translationMix = translatorChi.translate(text1)
print(lang0)
print(langs0)
print(translationChi)
print(translation)
print(lang3)
print(langs3)
print(translationSpn)
print(lang1)
print(langs1)
print(translationMix)

# Import the LanguageIdentification class from SpeechBrain
from speechbrain.pretrained import LanguageIdentification

# Load the pretrained language identification model.
# In this case, we use the "lang-id-commonlanguage_ecapa" model.
# The model gets automatically downloaded and stored in the specified directory.
lang_id = LanguageIdentification.from_hparams(
    source="speechbrain/lang-id-commonlanguage_ecapa",
    savedir="pretrained_models/lang-id-commonlanguage_ecapa"
)

# Specify the path to your WAV file. Replace 'your_audio.wav' with your actual file name.
wav_file = "temp_102.wav"

# Use the classify_file method to predict the language.
# This method returns a dictionary with language prediction scores.
predictions = lang_id.classify_file(wav_file)

# Display the language predictions.
print("Language Predictions:")
print(predictions)
