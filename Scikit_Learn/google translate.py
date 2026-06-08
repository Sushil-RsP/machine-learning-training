import speech_recognition as sr
from googletrans import Translator

# Initialize speech recognition and translator
recognizer = sr.Recognizer()
translator = Translator()

# Function to recognize speech and translate
def recognize_and_translate():
    with sr.Microphone() as source:
        print("Speak something...")
        audio = recognizer.listen(source)

    try:
        # Recognize speech
        text = recognizer.recognize_google(audio)

        # Translate text to Hindi
        translated_text = translator.translate(text, dest='hi').text
        print("Original:", text)
        print("Translated (Hindi):", translated_text)

    except sr.UnknownValueError:
        print("Could not understand audio")
    #except sr.RequestError as e:
     #   print("Could not request results; {0}".format(e))

# Continuous loop for real-time translation
while True:
    recognize_and_translate()