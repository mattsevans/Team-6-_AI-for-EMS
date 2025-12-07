'''import spacy
from spacy.tokens import Doc, Span
from spacy.language import Language
from spacy_language_detection import LanguageDetector
# install using pip install googletrans
from googletrans import Translator

nlp = spacy.load("en_core_web_sm")


def custom_detection_function(spacy_object):
    # Custom detection function should take a spaCy Doc or a Span
    assert isinstance(spacy_object, Doc) or isinstance(
        spacy_object, Span), "spacy_object must be a spacy Doc or Span object but it is a {}".format(type(spacy_object))
    detection = Translator().detect(spacy_object.text)
    return {'language': detection.lang, 'score': detection.confidence}


def get_lang_detector(nlp, name):
    return LanguageDetector(language_detection_function=custom_detection_function, seed=42)  # We use the seed 42


nlp_model = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp_model.add_pipe('language_detector', last=True)

text = "This is English text. Er lebt mit seinen Eltern und seiner Schwester in Berlin. Yo me divierto todos los días en el parque. Je m'appelle Angélica Summer, j'ai 12 ans et je suis canadienne."

# Document level language detection
doc = nlp_model(text)
language = doc._.language
print(language)

# Sentence level language detection
text = "This is English text. Er lebt mit seinen Eltern und seiner Schwester in Berlin. Yo me divierto todos los días en el parque. Je m'appelle Angélica Summer, j'ai 12 ans et je suis canadienne."
doc = nlp_model(text)
for i, sent in enumerate(doc.sents):
    print(sent, sent._.language)'
    '''



import spacy

from spacy_language_detection import LanguageDetector

# This only work with spaCy 2.0, it's not working with spaCy 3.0
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
text = "This is English text. Er lebt mit seinen Eltern und seiner Schwester in Berlin. Yo me divierto todos los " \
       "días en el parque. Je m'appelle Angélica Summer, j'ai 12 ans et je suis canadienne."
doc = nlp(text)
# document level language detection. Think of it like average language of document!
print(doc._.language)
# sentence level language detection
for i, sent in enumerate(doc.sents):
    print(sent, sent._.language)

# Token level language detection from version 0.1.2
# Use this with caution because, in some cases language detection will not make sense for individual tokens
for token in doc:
    print(token, token._.language)