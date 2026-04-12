import time
import re
from multiprocessing.queues import Queue

import numpy as np

from src.va.config.va_config import VAConfig
from src.va.ipc.events import Event, PipelineMarkerEvent, TTSDoneEvent
from src.va.orchestrator.turn_context import TurnContext
from src.va.response.types import GeneratedToken

from .phonemizer_engine import PhonemizerEngine
from .synthesis_engine import PiperEngine
from .types import TTSAudio


class TTSPipeline:
    def __init__(
        self,
        text_queue: Queue[GeneratedToken],
        playback_queue: Queue[TTSAudio],
        event_queue: Queue[Event],
        cfg: VAConfig,
    ):
        self.text_queue = text_queue
        self.playback_queue = playback_queue
        self.event_queue = event_queue

        # Initialize Components
        print("[TTS] Loading Piper Models...")
        self.phonemizer = PhonemizerEngine(cfg.phoneme_config_path, cfg.espeak_path)
        self.engine = PiperEngine(cfg.piper_path, cfg.phoneme_config_path)

        # Buffer for sentence construction
        self.text_buffer = ""
        self.sentence_endings = re.compile(r"[.!?;:]")
        self._first_audio_marked_turn_ids: set[int] = set()

    def run(self):
        print("[TTS] Worker Ready.")
        while True:
            try:
                token = self.text_queue.get()

                # Check for EOS signal
                if token.token is None:
                    self._flush_buffer(token.ctx)
                    self.event_queue.put(TTSDoneEvent())
                    # Send EOS to playback as well.
                    self.playback_queue.put(
                        TTSAudio(
                            pcm=None,
                            sample_rate=self.engine.sample_rate,
                            ctx=token.ctx,
                        )
                    )
                    print("[TTS] Stream Finished.")
                    continue

                if token.ctx.cancelled.is_set():
                    self.text_buffer = ""  # Reset buffer
                    continue
                # Accumulate text
                self.text_buffer += token.token

                # Check if we have a full sentence
                if self._has_sentence_boundary():
                    self._process_buffer(token.ctx)

            except Exception as e:
                print(f"[TTS] Error: {e}")

    def _has_sentence_boundary(self) -> bool:
        """Check if the buffer contains a sentence-ending punctuation."""
        return bool(self.sentence_endings.search(self.text_buffer))

    def _process_buffer(self, ctx: TurnContext):
        """Splits buffer into sentences and processes them."""
        # Split but keep delimiters
        # This regex splits by punctuation but keeps it attached to the sentence
        parts = re.split(r"([.!?;:])", self.text_buffer)

        # Reconstruct sentences
        sentences = []
        current_sent = ""
        for part in parts:
            if self.sentence_endings.match(part):
                current_sent += part
                sentences.append(current_sent)
                current_sent = ""
            else:
                current_sent += part

        # If there is leftover text (incomplete sentence), keep it in buffer
        self.text_buffer = current_sent

        # Synthesize each complete sentence
        for sent in sentences:
            if sent.strip():
                self._synthesize_and_push(sent, ctx)

    def _flush_buffer(self, ctx: TurnContext):
        """Force process whatever is left in the buffer."""
        if self.text_buffer.strip():
            self._synthesize_and_push(self.text_buffer, ctx)
        self.text_buffer = ""

    def _synthesize_and_push(self, text: str, ctx: TurnContext):
        # 1. Text -> Ids
        ids = self.phonemizer.text_to_ids(text)

        # 2. Ids -> Audio Array (Float32)
        audio_float = self.engine.synthesize(ids)

        # 3. Audio Float -> Int16 Bytes
        audio_int16 = self._float_to_int16(audio_float)

        # 4. Push to Playback
        packet = TTSAudio(
            pcm=audio_int16.tobytes(),
            sample_rate=self.engine.sample_rate,
            ctx=ctx,
        )
        if ctx.turn_id not in self._first_audio_marked_turn_ids:
            self.event_queue.put(
                PipelineMarkerEvent(
                    marker="tts_first_audio_emit",
                    t_mono_ns=time.monotonic_ns(),
                    turn_id=ctx.turn_id,
                )
            )
            self._first_audio_marked_turn_ids.add(ctx.turn_id)
        self.playback_queue.put(packet)

    def _float_to_int16(self, audio: np.ndarray) -> np.ndarray:
        """Normalize and convert to int16."""
        # Normalize to prevent clipping (simple max scaling)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Scale to int16 range
        return (audio * 32767).astype(np.int16)
