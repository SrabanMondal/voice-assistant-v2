import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from multiprocessing.queues import Queue

# =========================
# Workers
# =========================
from src.va.audio.capture import AudioInput
from src.va.audio.playback import playback_thread_func
from src.va.audio.ring_buffer import RingBuffer
# ========================
# Events and Msg
# =========================
from src.va.audio.types import AudioFrame
from src.va.config.va_config import default_config
from src.va.intent.types import IntentResult
from src.va.intent.worker import run_intent_worker
from src.va.ipc.events import Event
from src.va.orchestrator.orchestrator_engine import Orchestrator
from src.va.response.types import GeneratedToken, GenerationTask
from src.va.response.worker import run_response_worker
from src.va.stt.types import TranscriptionMsg
from src.va.stt.worker import run_speech_worker
from src.va.tts.types import TTSAudio
from src.va.tts.worker import run_tts_process
from src.va.ww.worker import run_porcupine_worker

# =========================
# Master process
# =========================
access_key = os.getenv("WW_KEY")


def run():
    print("[MASTER] Starting voice assistant")
    cfg = default_config()
    # -------------------------
    # IPC Queues
    # -------------------------
    audio_q_wake: Queue[AudioFrame] = mp.Queue(maxsize=32)
    audio_q_stt: Queue[AudioFrame] = mp.Queue(maxsize=64)
    stt_text_q: Queue[TranscriptionMsg] = mp.Queue()
    intent_q: Queue[IntentResult] = mp.Queue()
    prompt_q: Queue[GenerationTask] = mp.Queue()
    tts_text_q: Queue[GeneratedToken | None] = mp.Queue()
    play_q: Queue[TTSAudio] = mp.Queue()
    event_q: Queue[Event] = mp.Queue()

    # -------------------------
    # Start workers
    # -------------------------
    wake_proc = mp.Process(
        target=run_porcupine_worker,
        args=(audio_q_wake, event_q, access_key, cfg),
        daemon=True,
    )

    stt_proc = mp.Process(
        target=run_speech_worker,
        args=(audio_q_stt, stt_text_q, event_q, cfg),
        daemon=True,
    )

    intent_proc = mp.Process(
        target=run_intent_worker,
        args=(intent_q, event_q),
        daemon=True,
    )

    response_proc = mp.Process(
        target=run_response_worker,
        args=(prompt_q, tts_text_q, event_q),
        daemon=True,
    )

    tts_proc = mp.Process(
        target=run_tts_process,
        args=(tts_text_q, play_q, event_q, cfg),
        daemon=True,
    )

    playback_thread = threading.Thread(
        target=playback_thread_func, args=(play_q,event_q), daemon=True
    )

    wake_proc.start()
    stt_proc.start()
    intent_proc.start()
    response_proc.start()
    tts_proc.start()
    playback_thread.start()

    print("[MASTER] Workers started")

    # -------------------------
    # Audio + State
    # -------------------------

    ring = RingBuffer(
        seconds=2.0,
        sample_rate=16000,
        frame_size=512,
    )

    ost = Orchestrator(parallel_intent=True)

    # -------------------------
    # Clean Shutdown
    # -------------------------
    def shutdown_handler(sig, frame):
        print("\n[MASTER] Shutting down")

        for p in (
            wake_proc,
            stt_proc,
            intent_proc,
            response_proc,
            tts_proc,
        ):
            p.terminate()

        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    # -------------------------
    # MAIN LOOP
    # -------------------------
    running = True

    while running:
        try:
            mic = AudioInput(
                sample_rate=16000,
                frame_size=512,
            )

            for frame in mic.frames():
                ring.push(frame)

                try:
                    audio_q_wake.put_nowait(frame)
                except Exception:
                    pass

                if ost.allow_stt_audio():
                    try:
                        audio_q_stt.put_nowait(frame)
                    except Exception:
                        pass

                # 4. Handle events (non-blocking)
                while not event_q.empty():
                    event = event_q.get()
                    """
                     components dict contains all queues:
                    {
                        'stt_audio_q': Queue,
                        'stt_text_q' : Queue
                        'intent_q': Queue,
                        'response_q': Queue,
                        'playback_q': Queue,  # For clearing on interrupt
                        'ring_buffer': RingBuffer
                    }
                    """
                    components = {
                        "stt_audio_q": audio_q_stt,
                        "stt_text_q": stt_text_q,
                        "intent_q": intent_q,
                        "response_q": prompt_q,
                        "playback_q": play_q,
                        "ring_buffer": ring,
                    }
                    # Can do Typed Dict, but anyways, all works.
                    ost.handle_event(event, components)

        except RuntimeError as e:
            # -------- Mic failure handling --------
            print(f"[MASTER] Mic failure: {e}")
            ring.clear()
            time.sleep(0.5)
            continue


# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    run()
