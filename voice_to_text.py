#!/usr/bin/env python3

import argparse
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.io.wavfile import write


SAMPLE_RATE = 16_000


def record_audio(duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record mono audio from the default microphone."""
    print(f"Recording for {duration_s:.1f} seconds...")
    audio = sd.rec(
        int(duration_s * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Recording complete.")

    # Convert shape from (samples, 1) to (samples,)
    return audio.squeeze()


def save_wav(audio: np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Save float audio as 16-bit PCM WAV."""
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    write(path, sample_rate, audio_int16)

# See this link to find out more about the different compute types:
# https://opennmt.net/CTranslate2/quantization.html
def transcribe_file(
    audio_path: Path,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str,
) -> str:
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
    )

    print(f"Detected language: {info.language}")
    print(f"Language probability: {info.language_probability:.2f}")
    if info.all_language_probs:
        print("All language probabilities:")
        for lang, prob in info.all_language_probs.items():
            print(f"  {lang}: {prob:.2f}")

    text_parts = []

    for segment in segments:
        text = segment.text.strip()
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text}")
        text_parts.append(text)

    return " ".join(text_parts).strip()


class VoiceCommandProcessor:
    def __init__(self, model_size: str = "base", language: str = "es", device: str = "auto", compute_type: str = "float32"):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type

    def process_voice_command(self, duration_s: float = 5.0) -> str:
        audio = record_audio(duration_s)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        save_wav(audio, wav_path)

        try:
            final_text = transcribe_file(
                audio_path=wav_path,
                model_size=self.model_size,
                language=self.language,
                device=self.device,
                compute_type=self.compute_type,
            )
            return final_text
        finally:
            wav_path.unlink(missing_ok=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--model", default="base", help="tiny, base, small, medium, large-v3")
    parser.add_argument("--language", default=None, help="Example: en, es. Default: auto-detect")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-type", default="int8", help="Example: int8, float16, float32")
    args = parser.parse_args()

    audio = record_audio(args.duration)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    save_wav(audio, wav_path)

    try:
        final_text = transcribe_file(
            audio_path=wav_path,
            model_size=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
        )

        print("\nFinal text:")
        print(final_text)

    finally:
        wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

