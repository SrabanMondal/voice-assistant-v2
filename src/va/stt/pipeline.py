import time
from multiprocessing.queues import Queue

import numpy as np

from src.va.audio.types import AudioFrame
from src.va.config.va_config import VAConfig
from src.va.ipc.events import Event, PipelineMarkerEvent, STTFinalEvent
from src.va.stt.stt_engine import MoonshineSTT
from src.va.stt.types import TranscriptionMsg
from src.va.stt.vad_engine import SileroVAD


class SpeechPipeline:
    def __init__(
        self,
        audio_queue: Queue[AudioFrame],
        text_queue: Queue[TranscriptionMsg],
        event_q: Queue[Event],
        config: VAConfig,
    ):
        self.audio_queue = audio_queue
        self.text_queue = text_queue
        self.event_queue = event_q

        # --- 1. Load Engines ---
        print("[SpeechPipeline] Loading AI Models...")
        self.vad = SileroVAD(config.silero_path)
        self.stt = MoonshineSTT(
            config.moonshine_enc_path, config.moonshine_dec_path, config.tokenizer_path
        )

        # --- Configuration ---
        self.SAMPLE_RATE = 16000

        # The Two-Threshold Strategy
        self.PHRASE_THRESHOLD = 0.3  # Micro-pause: Emit Partial (Lazy Speculation)
        self.FINAL_THRESHOLD = 2.0  # Macro-pause: Emit Final (Commit Intent)

        self.MIN_SPEECH_DURATION = 0.2
        self.MAX_BUFFER_DURATION = 15.0  # Force flush if user talks too long, its experimental althought

        # --- State Management ---
        self.buffer = []
        self.is_triggered = False

        # Timers
        self.silence_start_time = None
        self.last_partial_time = 0.0
        self._partial_marker_emitted = False

    def run(self):
        """The infinite loop worker."""
        print("[SpeechPipeline] Worker Started.")

        while True:
            try:
                frame: AudioFrame = self.audio_queue.get()

                # VAD Check
                is_speech = self.vad.is_speech(frame.pcm)

                if is_speech:
                    self._handle_speech(frame)
                else:
                    self._handle_silence(frame)

            except Exception as e:
                print(f"[SpeechPipeline] Error: {e}")

    def _handle_speech(self, frame):
        """User is actively talking."""

        self.silence_start_time = None

        # Trigger Start
        if not self.is_triggered:
            print(" [Speech Start] ")
            self.is_triggered = True
            self.buffer = []
            self.last_partial_time = 0.0
            self._partial_marker_emitted = False

        self.buffer.append(frame.pcm)

        # Safety Check: Force flush if buffer gets massive (>15s). This is yet experimental..
        curr_len = (len(self.buffer) * len(frame.pcm)) / self.SAMPLE_RATE
        if curr_len > self.MAX_BUFFER_DURATION:
            print(" [Max Duration Reached] Force Committing.")
            self._emit_final()

    def _handle_silence(self, frame):
        """User is silent. Check thresholds."""

        # Ignore silence if we haven't started talking yet
        if not self.is_triggered:
            return

        # Add Silence to Buffer (Padding helps Moonshine accuracy maybe, anyways its harmless so why not)
        self.buffer.append(frame.pcm)

        # Start/Check Silence Timer
        now = time.time()
        if self.silence_start_time is None:
            self.silence_start_time = now
            return  # Just started being silent, wait for next frame

        silence_duration = now - self.silence_start_time

        # --- CHECK 1: FINAL THRESHOLD (Macro-Pause) ---
        if silence_duration > self.FINAL_THRESHOLD:
            print(f" [Silence > {self.FINAL_THRESHOLD}s] Finalizing...")
            self._emit_final()
            return

        # --- CHECK 2: PHRASE THRESHOLD (Micro-Pause) ---
        if silence_duration > self.PHRASE_THRESHOLD:
            # Only emit if we haven't emitted recently for THIS specific pause
            # (Prevent spamming 10 partials during a 0.5s pause)
            if (now - self.last_partial_time) > self.PHRASE_THRESHOLD:
                self._emit_partial(now)

    def _emit_partial(self, now):
        """
        Lazy Speculation: User paused briefly.
        Send update so Intent LLM for speculative intent.
        """
        text = self._transcribe_buffer()
        if len(text) > 2:
            if not self._partial_marker_emitted:
                self.event_queue.put(
                    PipelineMarkerEvent(
                        marker="stt_first_partial",
                        t_mono_ns=time.monotonic_ns(),
                    )
                )
                self._partial_marker_emitted = True
            # self.text_queue.put(TranscriptionMsg(
            #     text=text,
            #     type=TranscriptionType.PARTIAL,
            #     timestamp=now,
            #     ctx=None,
            # ))
            # For now, we don't send partial events, Since I don't speculative decoding for now.
            self.last_partial_time = now

    def _emit_final(self):
        """
        Commit: User is done.
        Send Final event and RESET state.
        """
        text = self._transcribe_buffer()

        if len(text) > 0:
            self.event_queue.put(
                PipelineMarkerEvent(
                    marker="stt_final_emit",
                    t_mono_ns=time.monotonic_ns(),
                )
            )
            # self.text_queue.put(TranscriptionMsg(
            #     text=text,
            #     type=TranscriptionType.FINAL,
            #     timestamp=time.time(),
            #     ctx=None,
            # ))
            # We can have STT to Intent, but keeping orchestrator in between is more safer
            self.event_queue.put(STTFinalEvent(text=text))

        # Reset State
        self.is_triggered = False
        self.silence_start_time = None
        self.buffer = []
        self._partial_marker_emitted = False
        print(" [Speech End] State Reset.")

    def _transcribe_buffer(self) -> str:
        if not self.buffer:
            return ""

        # Concatenate list of arrays into one array
        full_audio = np.concatenate(self.buffer)
        print(full_audio)
        try:
            return self.stt.transcribe(full_audio)
        except Exception as e:
            print(f"Inference Error: {e}")
            return ""
