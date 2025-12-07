"""
Sean Bolger NLP for AI wearable device
LanguageTranslationFunctions
Produces 4 translation functions to translate text to and from english
Test of code is commented out at the bottom. these are called in the completed combination test
"""
from translate import Translator    #initialized translation data

#defines all translation functions
translatorZH = Translator(from_lang="zh", to_lang='en')
translatorES = Translator(from_lang="es", to_lang="en")
translatorEnToZH = Translator(from_lang="en", to_lang='zh')
translatorEnToES = Translator(from_lang="en", to_lang="es")

#sets up 4 translation functions, to and from english for both spanish and chinese

#Chinese to english
def CN_to_EN_Text(CNtext):
    translation = translatorZH.translate(CNtext) 
    return(translation)

#spanish to english
def ES_to_EN_Text(EStext):
    translation = translatorES.translate(EStext) 
    return(translation)

#english to chinese
def EN_to_CN_Text(ENtext):
    translation = translatorEnToZH.translate(ENtext) 
    return(translation)

#english to spanish
def EN_to_ES_Text(ENtext):
    translation = translatorEnToES.translate(ENtext) 
    return(translation)

"""
    PROOF OF CONCEPT TESTING
text0 = "权利 与 义务 与 尊严"
text1 = CN_to_EN_Text(text0)
print(text1)
textEn = "Hi, how are you today?"
text2 = EN_to_CN_Text(textEn)
print(text2)
text3 = EN_to_SN_Text(textEn)
print(text3)
textSn = "Hola, Como estas?"
text4 = SN_to_EN_Text(textSn)
print(text4)
"""