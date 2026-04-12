import logging
import time
from dataclasses import dataclass, field
from multiprocessing.queues import Queue
from queue import Empty
from threading import Event

from src.va.ipc.events import Event as VAEvent
from src.va.ipc.events import PlayBackEvent
from src.va.tts.types import TTSAudio


LOGGER = logging.getLogger("eval.playback_probe")


@dataclass
class PlaybackProbeState:
    first_audio_ts: float | None = None
    playback_end_ts: float | None = None
    done: Event = field(default_factory=Event)


def playback_probe_thread_func(
    playback_q: Queue[TTSAudio],
    event_q: Queue[VAEvent],
    state: PlaybackProbeState,
    shutdown_event: Event | None = None,
):
    """
    Probe replacement for playback thread.
    Consumes TTSAudio packets, records timing anchors, and emits PlayBackEvent on EOS.
    """
    LOGGER.info("Playback probe thread started")

    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            LOGGER.info("Playback probe shutdown requested, exiting")
            break

        LOGGER.debug("Waiting for playback queue item")
        try:
            item = playback_q.get(timeout=0.2)
        except Empty:
            continue
        LOGGER.debug("Playback queue item received: %s", type(item).__name__)

        if item is None:
            LOGGER.info("Playback probe received sentinel, exiting")
            break

        if not isinstance(item, TTSAudio):
            continue

        if item.ctx.cancelled.is_set():
            continue

        if item.pcm is None:
            if state.playback_end_ts is None:
                state.playback_end_ts = time.monotonic()
                LOGGER.info("Playback EOS observed at %.6f", state.playback_end_ts)
            event_q.put(PlayBackEvent())
            LOGGER.debug("PlayBackEvent emitted to event queue")
            state.done.set()
            continue

        if state.first_audio_ts is None:
            state.first_audio_ts = time.monotonic()
            LOGGER.info("First audio packet observed at %.6f", state.first_audio_ts)

    LOGGER.info("Playback probe thread stopped")
