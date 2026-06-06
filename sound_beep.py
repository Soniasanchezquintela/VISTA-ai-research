from __future__ import annotations

import enum
import threading
import time

import numpy as np
import sounddevice as sd


class TargetDistance(enum.Enum):
    UNKNOWN = enum.auto()   # Target out of scope: 1 Hz beep
    DETECTED = enum.auto()  # Target visible but far: 4 Hz beep
    CLOSE = enum.auto()     # Target at arm's reach: continuous tone


class SoundBeep:
    def __init__(
        self,
        frequency_hz: float = 1000.0,
        volume: float = 0.3,
        sample_rate: int = 44100,
        beep_duration_s: float = 0.10,
    ) -> None:
        self.frequency_hz = frequency_hz
        self.volume = volume
        self.sample_rate = sample_rate
        self.beep_duration_s = beep_duration_s

        self._target_distance = TargetDistance.UNKNOWN

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        """
        Launch the internal thread and start beeping.

        Calling start() resets the target distance to UNKNOWN.
        """
        with self._lock:
            self._target_distance = TargetDistance.UNKNOWN

            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="SoundBeepThread",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """
        Stop the internal thread and stop any active sound playback.
        """
        self._stop_event.set()
        sd.stop()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

        self._thread = None

    def set_target_distance(self, enum_value: TargetDistance) -> None:
        """
        Update the current target distance state.
        """
        if not isinstance(enum_value, TargetDistance):
            raise TypeError(
                f"enum_value must be a TargetDistance, got {type(enum_value).__name__}"
            )

        with self._lock:
            self._target_distance = enum_value

    def _get_target_distance(self) -> TargetDistance:
        with self._lock:
            return self._target_distance

    def _run(self) -> None:
        while not self._stop_event.is_set():
            target_distance = self._get_target_distance()

            if target_distance == TargetDistance.UNKNOWN:
                self._beep_once(beep_hz=1.0)

            elif target_distance == TargetDistance.DETECTED:
                self._beep_once(beep_hz=4.0)

            elif target_distance == TargetDistance.CLOSE:
                self._play_continuous_tone_until_state_changes()

    def _beep_once(self, beep_hz: float) -> None:
        period_s = 1.0 / beep_hz
        silence_s = max(0.0, period_s - self.beep_duration_s)

        self._play_tone_blocking(self.beep_duration_s)
        self._sleep_interruptible(silence_s)

    def _play_tone_blocking(self, duration_s: float) -> None:
        if self._stop_event.is_set():
            return

        t = np.linspace(
            0,
            duration_s,
            int(self.sample_rate * duration_s),
            endpoint=False,
        )

        signal = np.sin(2 * np.pi * self.frequency_hz * t)
        signal = self.volume * signal

        sd.play(signal, samplerate=self.sample_rate)
        sd.wait()

    def _play_continuous_tone_until_state_changes(self) -> None:
        """
        Play a real continuous tone while the state remains CLOSE.

        This uses an OutputStream callback instead of repeatedly calling
        sd.play(), avoiding the small gaps between audio chunks.
        """
        phase = 0.0
        phase_increment = 2.0 * np.pi * self.frequency_hz / self.sample_rate

        def callback(outdata, frames, time_info, status) -> None:
            nonlocal phase

            phases = phase + phase_increment * np.arange(frames)
            samples = np.sin(phases) * self.volume

            outdata[:, 0] = samples

            phase = phases[-1] + phase_increment
            phase %= 2.0 * np.pi

        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=callback,
        ):
            while not self._stop_event.is_set():
                if self._get_target_distance() != TargetDistance.CLOSE:
                    break

                time.sleep(0.02)

    def _sleep_interruptible(self, duration_s: float) -> None:
        end_time = time.monotonic() + duration_s

        while not self._stop_event.is_set():
            remaining_s = end_time - time.monotonic()
            if remaining_s <= 0:
                break

            time.sleep(min(remaining_s, 0.02))
