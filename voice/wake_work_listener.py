# wake_word_listener.py

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model


class WakeWordListener:
    def __init__(
        self,
        model_path: str | Path,
        wakeword_name: str = "hey_super",
        threshold: float = 0.5,
        sample_rate: int = 16_000,
        frame_ms: int = 80,
        cooldown_s: float = 1.5,
    ) -> None:
        self.model_path = str(model_path)
        self.wakeword_name = wakeword_name
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.cooldown_s = cooldown_s

        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.model = Model(wakeword_models=[self.model_path])

        self._running = False
        self._last_detection_time = 0.0

    def start(self, on_wake: Callable[[], None]) -> None:
        self._running = True

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            callback=lambda indata, frames, time_info, status: self._callback(
                indata=indata,
                status=status,
                on_wake=on_wake,
            ),
        ):
            while self._running:
                time.sleep(0.05)

    def stop(self) -> None:
        self._running = False

    def _callback(self, indata: np.ndarray, status: sd.CallbackFlags, on_wake: Callable[[], None]) -> None:
        if status:
            print(f"Audio warning: {status}")

        audio_frame = indata[:, 0].copy()

        prediction = self.model.predict(audio_frame)

        # The model key may be derived from the ONNX filename.
        # Print prediction once during integration to confirm the exact key.
        score = max(prediction.values()) if prediction else 0.0

        now = time.monotonic()
        if score >= self.threshold and now - self._last_detection_time > self.cooldown_s:
            self._last_detection_time = now
            print(f"Wake word detected, score={score:.3f}")
            on_wake()

if __name__ == "__main__":
    def on_wake():
        print("Wake word callback triggered!")

    openwakeword.utils.download_models()
    listener = WakeWordListener(model_path="models/hey_super.onnx",threshold=0.5,)
    try:
        listener.start(on_wake=on_wake)
    except KeyboardInterrupt:
        print("Stopping wake word listener...")
        listener.stop()
