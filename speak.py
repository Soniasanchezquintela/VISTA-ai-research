#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
VOICE_DIRS = [
    BASE_DIR / "voices",
    BASE_DIR / "voices" / "piper",
    BASE_DIR / "models" / "piper",
]


def normalize_lang(lang):
    return lang.replace("_", "-").lower()


def normalize_gender(gender):
    return gender.lower()


def language_aliases(lang):
    normalized = normalize_lang(lang)
    aliases = {normalized, normalized.replace("-", "_"), normalized.replace("-", "")}

    if normalized.startswith("en"):
        aliases.update({"en", "english"})
    elif normalized.startswith("es"):
        aliases.update({"es", "spanish", "espanol"})

    return aliases


def candidate_voice_paths(gender, lang):
    normalized_lang = normalize_lang(lang)
    normalized_gender = normalize_gender(gender)
    filenames = [
        f"{normalized_lang}-{normalized_gender}.onnx",
        f"{normalized_lang}_{normalized_gender}.onnx",
        f"{normalized_lang}.onnx",
        f"{normalized_lang.replace('-', '_')}-{normalized_gender}.onnx",
        f"{normalized_lang.replace('-', '_')}.onnx",
    ]

    for voice_dir in VOICE_DIRS:
        for filename in filenames:
            yield voice_dir / filename


def find_piper_model(gender, lang):
    explicit_model = os.environ.get("PIPER_MODEL")
    if explicit_model:
        model_path = Path(explicit_model).expanduser()
        if model_path.exists():
            return model_path

        raise RuntimeError(f"PIPER_MODEL points to a missing file: {model_path}")

    for path in candidate_voice_paths(gender, lang):
        if path.exists():
            return path

    aliases = language_aliases(lang)
    normalized_gender = normalize_gender(gender)
    discovered_models = []
    for voice_dir in VOICE_DIRS:
        if voice_dir.exists():
            discovered_models.extend(voice_dir.rglob("*.onnx"))

    scored_models = []
    for model_path in discovered_models:
        model_name = model_path.stem.lower().replace("_", "-")
        score = 0
        if any(alias.replace("_", "-") in model_name for alias in aliases):
            score += 2
        if normalized_gender in model_name:
            score += 1
        scored_models.append((score, model_path))

    scored_models.sort(key=lambda item: (-item[0], str(item[1])))
    if scored_models and scored_models[0][0] > 0:
        return scored_models[0][1]

    raise RuntimeError(
        "No Piper voice model found. Set PIPER_MODEL=/path/to/voice.onnx, "
        "or place a .onnx voice under voices/, voices/piper/, or models/piper/."
    )


def play_audio_file(path):
    if shutil.which("ffplay"):
        try:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
                check=True,
            )
            return
        except subprocess.CalledProcessError:
            pass

    if shutil.which("aplay"):
        try:
            subprocess.run(["aplay", "-q", str(path)], check=True)
            return
        except subprocess.CalledProcessError:
            pass

    raise RuntimeError("Could not find ffplay or aplay to play generated speech audio.")


def piper_text_to_speech(text, gender, lang):
    piper_bin = shutil.which("piper")
    if piper_bin is None:
        raise RuntimeError("piper executable is not installed. Run: pip install piper-tts")

    model_path = find_piper_model(gender, lang)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_path = Path(audio_file.name)

    command = [
        piper_bin,
        "--model",
        str(model_path),
        "--output_file",
        str(audio_path),
    ]

    speaker = os.environ.get("PIPER_SPEAKER")
    if speaker:
        command.extend(["--speaker", speaker])

    try:
        subprocess.run(command, input=text, text=True, check=True)
        print(f"Using offline Piper voice: {model_path}")
        play_audio_file(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)


def pyttsx3_text_to_speech(text, gender, lang):
    import pyttsx3

    engine = pyttsx3.init()

    engine.setProperty("rate", 125)
    engine.setProperty("volume", 0.8)

    voices = engine.getProperty("voices")
    gender_tag = "VoiceGenderMale" if gender == "Male" else "VoiceGenderFemale"
    normalized_lang = normalize_lang(lang)
    match = None
    fallback = None
    for voice in voices:
        voice_languages = [
            normalize_lang(language.decode() if isinstance(language, bytes) else str(language))
            for language in voice.languages
        ]
        if normalized_lang in voice_languages:
            if voice.gender == gender_tag:
                match = voice
                break
            if fallback is None:
                fallback = voice

    chosen = match or fallback
    if chosen:
        engine.setProperty("voice", chosen.id)
        print(f"Using fallback pyttsx3 voice: {chosen.name} ({chosen.id})")
    else:
        print(f"No pyttsx3 voice found for lang={lang}; using system default")

    engine.say(text)
    engine.runAndWait()


def text_to_speech(text, gender, lang="es_ES"):
    try:
        piper_text_to_speech(text, gender, lang)
    except RuntimeError as exc:
        print(f"Piper TTS unavailable: {exc}")
        print("Falling back to pyttsx3/eSpeak voice.")
        try:
            pyttsx3_text_to_speech(text, gender, lang)
        except ImportError:
            print("pyttsx3 is not installed either. Run: pip install -r requirements.txt")
            sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python speak.py <text> <gender> [lang]")
        print("Example: python speak.py 'Hola, esto es una prueba.' Female es_ES")
        print("         python speak.py 'Hello, this is a test.' Male en_US")
        print("Optional: PIPER_MODEL=/path/to/voice.onnx python speak.py 'Hello' Male en_US")
        sys.exit(1)
    text = sys.argv[1]
    gender = sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else "es_ES"
    text_to_speech(text, gender, lang)
