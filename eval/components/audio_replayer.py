import logging
import time
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

from src.va.audio.types import AudioFrame


LOGGER = logging.getLogger("eval.audio_replayer")


class AudioReplayer:
    def __init__(
        self,
        wav_path: str,
        frame_size: int = 512,
        sample_rate: int = 16000,
        realtime: bool = True,
    ):
        self.wav_path = Path(wav_path)
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.realtime = realtime

        if not self.wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {self.wav_path}")

        self._audio_int16 = self._load_wav()
        LOGGER.info(
            "Loaded WAV | path=%s | samples=%s | frame_size=%s | realtime=%s",
            self.wav_path,
            self._audio_int16.size,
            self.frame_size,
            self.realtime,
        )

    def _load_wav(self) -> np.ndarray:
        with wave.open(str(self.wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            frames = wf.getnframes()

            if channels != 1:
                raise ValueError(
                    f"Expected mono WAV (1 channel), got {channels} in {self.wav_path}"
                )
            if sr != self.sample_rate:
                raise ValueError(
                    f"Expected {self.sample_rate} Hz WAV, got {sr} Hz in {self.wav_path}"
                )
            if sampwidth != 2:
                raise ValueError(
                    f"Expected 16-bit PCM WAV (sample width 2), got {sampwidth} in {self.wav_path}"
                )

            raw = wf.readframes(frames)

        audio = np.frombuffer(raw, dtype=np.int16)
        if audio.size == 0:
            raise ValueError(f"WAV file has no audio samples: {self.wav_path}")
        return audio

    def frames(self) -> Iterator[AudioFrame]:
        frame_duration = self.frame_size / self.sample_rate
        start_mono = time.monotonic()
        sim_timestamp = start_mono
        LOGGER.debug("Replay started | frame_duration=%.6fs", frame_duration)

        total = self._audio_int16.size
        frame_index = 0
        cursor = 0

        while cursor < total:
            end = min(cursor + self.frame_size, total)
            frame_int = self._audio_int16[cursor:end]

            if frame_int.size < self.frame_size:
                pad = np.zeros(self.frame_size - frame_int.size, dtype=np.int16)
                frame_int = np.concatenate((frame_int, pad), axis=0)

            if self.realtime:
                target = start_mono + (frame_index * frame_duration)
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

            frame_float = frame_int.astype(np.float32) / 32768.0

            yield AudioFrame(
                pcm=frame_float,
                intpcm=frame_int,
                sample_rate=self.sample_rate,
                timestamp=sim_timestamp,
            )

            sim_timestamp += frame_duration
            frame_index += 1
            cursor += self.frame_size

        LOGGER.debug("Replay ended | total_frames=%s", frame_index)
