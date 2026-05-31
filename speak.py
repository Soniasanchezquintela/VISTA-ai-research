#!/usr/bin/env python3

import pyttsx3
import sys

def text_to_speech(text, gender, lang='es_ES'):
    engine = pyttsx3.init()


    engine.setProperty('rate', 125)
    engine.setProperty('volume', 0.8)

    # Find a voice matching the requested language and gender
    voices = engine.getProperty('voices')
    gender_tag = 'VoiceGenderMale' if gender == 'Male' else 'VoiceGenderFemale'
    match = None
    fallback = None
    for v in voices:
        if lang in v.languages:
            if v.gender == gender_tag:
                match = v
                break
            elif fallback is None:
                fallback = v

    chosen = match or fallback
    if chosen:
        engine.setProperty('voice', chosen.id)
        print(f"Using voice: {chosen.name} ({chosen.id})")
    else:
        print(f"No voice found for lang={lang}, using system default")

    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python habla.py <text> <gender> [lang]")
        print("Example: python habla.py 'Hola, esto es una prueba.' Female es_ES")
        print("         python habla.py 'Hello, this is a test.' Male en_US")
        sys.exit(1)
    text = sys.argv[1]
    gender = sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else 'es_ES'
    text_to_speech(text, gender, lang)
