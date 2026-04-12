import argparse
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from queue import Empty, Full

import numpy as np

from eval.components.audio_replayer import AudioReplayer
from eval.components.playback_probe import (PlaybackProbeState,
                                            playback_probe_thread_func)
from eval.components.resource_sampler import ResourceSampler
from src.va.audio.ring_buffer import RingBuffer
from src.va.audio.types import AudioFrame
from src.va.config.va_config import default_config
from src.va.intent.worker import run_intent_worker
from src.va.ipc.events import (Event, GenerationDoneEvent, IntentEvent,
                               PipelineMarkerEvent, PlayBackEvent,
                               STTFinalEvent, STTPartialEvent, TTSDoneEvent,
                               WakeEvent)
from src.va.orchestrator.orchestrator_engine import Orchestrator
from src.va.response.types import GeneratedToken, GenerationTask
from src.va.response.worker import run_response_worker
from src.va.stt.types import TranscriptionMsg
from src.va.stt.worker import run_speech_worker
from src.va.tts.types import TTSAudio
from src.va.tts.worker import run_tts_process
from src.va.ww.worker import run_porcupine_worker


LOGGER = logging.getLogger("eval.run_eval")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(processName)s | %(threadName)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def safe_qsize(q: mp.Queue) -> int | None:
    try:
        return q.qsize()
    except Exception:
        return None


def sleep_interruptibly(
    duration_s: float,
    shutdown_event: threading.Event,
    step_s: float = 0.1,
) -> bool:
    if duration_s <= 0:
        return shutdown_event.is_set()
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        if shutdown_event.is_set():
            return True
        remaining = deadline - time.monotonic()
        time.sleep(min(step_s, max(remaining, 0.0)))
    return shutdown_event.is_set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic evaluation harness")
    parser.add_argument("--wav", required=True, help="Path to mono 16k WAV input")
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Replay frames as fast as possible (default is realtime mode)",
    )
    parser.add_argument(
        "--inject-wake",
        action="store_true",
        default=True,
        help="Inject WakeEvent before replay starts (default: enabled)",
    )
    parser.add_argument(
        "--no-inject-wake",
        action="store_false",
        dest="inject_wake",
        help="Disable WakeEvent injection",
    )
    parser.add_argument(
        "--startup-wait-s",
        type=float,
        default=2.0,
        help="Warm-up wait after worker start (excluded from metrics)",
    )
    parser.add_argument(
        "--idle-window-s",
        type=float,
        default=5.0,
        help="Idle resource sampling window before replay",
    )
    parser.add_argument(
        "--completion-timeout-s",
        type=float,
        default=45.0,
        help="Max wait for playback completion after replay ends",
    )
    parser.add_argument(
        "--post-silence-s",
        type=float,
        default=2.5,
        help="Trailing silence injected after replay to allow STT finalization",
    )
    parser.add_argument(
        "--wait-log-interval-s",
        type=float,
        default=2.0,
        help="Heartbeat interval while waiting for playback completion",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def drain_events(
    event_q: mp.Queue,
    orchestrator: Orchestrator,
    components: dict,
    block_timeout_s: float | None = None,
    event_counts: Counter | None = None,
    timeline_ns: dict[str, int] | None = None,
    marker_delivery_ms: dict[str, float] | None = None,
) -> int:
    processed = 0
    while True:
        try:
            if block_timeout_s is not None and processed == 0:
                LOGGER.debug("Blocking for event up to %.3fs", block_timeout_s)
                event = event_q.get(timeout=block_timeout_s)
            else:
                event = event_q.get_nowait()
        except Empty:
            break

        now_ns = time.monotonic_ns()
        event_name = type(event).__name__

        if isinstance(event, PipelineMarkerEvent):
            if event_counts is not None:
                event_counts[event_name] += 1
            _record_first_ts(timeline_ns, event.marker, event.t_mono_ns)
            if marker_delivery_ms is not None and event.marker not in marker_delivery_ms:
                marker_delivery_ms[event.marker] = (now_ns - event.t_mono_ns) / 1_000_000.0
            LOGGER.debug(
                "Marker received | marker=%s | turn_id=%s | delivery_lag_ms=%.2f",
                event.marker,
                event.turn_id,
                marker_delivery_ms.get(event.marker, 0.0) if marker_delivery_ms else 0.0,
            )
            processed += 1
            continue

        if isinstance(event, STTPartialEvent):
            _record_first_ts(timeline_ns, "stt_first_partial_event", now_ns)
        elif isinstance(event, STTFinalEvent):
            _record_first_ts(timeline_ns, "stt_final_event", now_ns)
        elif isinstance(event, IntentEvent):
            _record_first_ts(timeline_ns, "intent_event", now_ns)
        elif isinstance(event, GenerationDoneEvent):
            _record_first_ts(timeline_ns, "generation_done_event", now_ns)
        elif isinstance(event, TTSDoneEvent):
            _record_first_ts(timeline_ns, "tts_done_event", now_ns)
        elif isinstance(event, PlayBackEvent):
            _record_first_ts(timeline_ns, "playback_done_event", now_ns)

        if event_counts is not None:
            event_counts[event_name] += 1
        LOGGER.debug("Dequeued event: %s", event_name)
        orchestrator.handle_event(event, components)
        processed += 1

    return processed


def format_metric(name: str, value: float | None, suffix: str = "") -> str:
    if value is None:
        return f"{name}=N/A"
    return f"{name}={value:.2f}{suffix}"


def _record_first_ts(timeline_ns: dict[str, int] | None, key: str, ts_ns: int) -> None:
    if timeline_ns is None:
        return
    timeline_ns.setdefault(key, ts_ns)


def _delta_ms(timeline_ns: dict[str, int], start_key: str, end_key: str) -> float | None:
    start_ns = timeline_ns.get(start_key)
    end_ns = timeline_ns.get(end_key)
    if start_ns is None or end_ns is None:
        return None
    return (end_ns - start_ns) / 1_000_000.0


def _format_delta(name: str, value_ms: float | None) -> str:
    if value_ms is None:
        return f"{name}: N/A"
    return f"{name}: {value_ms:.2f} ms"


def _format_stage_offset(name: str, ts_ns: int | None, anchor_ns: int) -> str:
    if ts_ns is None:
        return f"{name}: N/A"
    return f"{name}: {(ts_ns - anchor_ns) / 1_000_000.0:.2f} ms"


def _format_ratio(name: str, value: float | None) -> str:
    if value is None:
        return f"{name}: N/A"
    return f"{name}: {value:.2f}"


def _format_ratio(name: str, value: float | None) -> str:
    if value is None:
        return f"{name}: N/A"
    return f"{name}: {value:.2f}"


def run_eval(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)

    shutdown_requested = threading.Event()
    previous_signal_handlers: dict = {}

    def _request_shutdown(signum: int, _frame) -> None:
        if not shutdown_requested.is_set():
            LOGGER.warning("Received signal %s; starting graceful shutdown", signum)
        shutdown_requested.set()

    for sig in (
        signal.SIGINT,
        getattr(signal, "SIGTERM", None),
        getattr(signal, "SIGBREAK", None),
    ):
        if sig is None:
            continue
        try:
            previous_signal_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _request_shutdown)
        except ValueError:
            LOGGER.warning("Signal handler setup skipped (not running on main thread)")

    wav_path = Path(args.wav)
    if not wav_path.exists():
        LOGGER.error("WAV file not found: %s", wav_path)
        return 1

    LOGGER.info("Eval start | wav=%s | realtime=%s", wav_path, not args.fast)

    cfg = default_config()

    # -------------------------
    # IPC Queues (matching src/va/main.py)
    # -------------------------
    audio_q_wake: mp.Queue[AudioFrame] = mp.Queue(maxsize=32)
    audio_q_stt: mp.Queue[AudioFrame] = mp.Queue(maxsize=64)
    stt_text_q: mp.Queue[TranscriptionMsg] = mp.Queue()
    intent_q: mp.Queue = mp.Queue()
    prompt_q: mp.Queue[GenerationTask] = mp.Queue()
    tts_text_q: mp.Queue[GeneratedToken | None] = mp.Queue()
    play_q: mp.Queue[TTSAudio] = mp.Queue()
    event_q: mp.Queue[Event] = mp.Queue()

    # -------------------------
    # Start workers (matching src/va/main.py topology)
    # -------------------------
    access_key = os.getenv("WW_KEY")
    LOGGER.info("WW_KEY configured: %s", bool(access_key))
    processes: list[mp.Process] = []

    wake_proc = None
    if access_key:
        wake_proc = mp.Process(
            target=run_porcupine_worker,
            args=(audio_q_wake, event_q, access_key, cfg),
            daemon=True,
        )
        processes.append(wake_proc)
    else:
        LOGGER.warning("WW_KEY not set, wake-word process is skipped")

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

    processes.extend([stt_proc, intent_proc, response_proc, tts_proc])

    playback_state = PlaybackProbeState()
    playback_thread = threading.Thread(
        target=playback_probe_thread_func,
        args=(play_q, event_q, playback_state, shutdown_requested),
        daemon=True,
    )

    ring = RingBuffer(seconds=2.0, sample_rate=args.sample_rate, frame_size=args.frame_size)
    orchestrator = Orchestrator()

    components = {
        "stt_audio_q": audio_q_stt,
        "stt_text_q": stt_text_q,
        "intent_q": intent_q,
        "response_q": prompt_q,
        "playback_q": play_q,
        "ring_buffer": ring,
    }

    sampler = ResourceSampler(interval_s=0.1)

    turn_input_start: float | None = None
    event_counts: Counter = Counter()
    timeline_ns: dict[str, int] = {}
    marker_delivery_ms: dict[str, float] = {}
    replay_frame_count = 0
    dropped_stt_frames = 0

    try:
        for process in processes:
            LOGGER.info("Starting process: %s", process.name)
            process.start()
            LOGGER.info("Started process: %s pid=%s", process.name, process.pid)

        LOGGER.info("Starting playback probe thread")
        playback_thread.start()
        LOGGER.info("Starting resource sampler thread")
        sampler.start()

        if args.startup_wait_s > 0:
            LOGGER.info("Warm-up wait: %.1fs", args.startup_wait_s)
            sleep_interruptibly(args.startup_wait_s, shutdown_requested)

        idle_start = time.monotonic()
        if args.idle_window_s > 0:
            LOGGER.info("Sampling idle resources for %.1fs", args.idle_window_s)
            sleep_interruptibly(args.idle_window_s, shutdown_requested)
        idle_end = time.monotonic()
        LOGGER.info("Idle sampling window complete")

        if shutdown_requested.is_set():
            LOGGER.warning("Shutdown requested before replay start")
            return 130

        if args.inject_wake:
            LOGGER.info("Injecting initial WakeEvent")
            event_q.put(WakeEvent())
            drain_events(
                event_q,
                orchestrator,
                components,
                event_counts=event_counts,
                timeline_ns=timeline_ns,
                marker_delivery_ms=marker_delivery_ms,
            )

        replayer = AudioReplayer(
            wav_path=str(wav_path),
            frame_size=args.frame_size,
            sample_rate=args.sample_rate,
            realtime=not args.fast,
        )

        LOGGER.info("Starting replay: %s", wav_path)

        for frame in replayer.frames():
            if shutdown_requested.is_set():
                LOGGER.warning("Shutdown requested during replay; stopping frame injection")
                break

            replay_frame_count += 1
            if turn_input_start is None:
                turn_input_start = time.monotonic()
                LOGGER.info("Turn input start anchored at monotonic=%.6f", turn_input_start)

            ring.push(frame)

            try:
                audio_q_wake.put_nowait(frame)
            except Exception:
                pass

            if orchestrator.allow_stt_audio():
                try:
                    audio_q_stt.put_nowait(frame)
                except Exception:
                    dropped_stt_frames += 1
                    LOGGER.debug("Dropped STT frame due to full queue")

            if replay_frame_count % 100 == 0:
                LOGGER.debug(
                    "Replay progress | frames=%s | event_q=%s | stt_q=%s | play_q=%s",
                    replay_frame_count,
                    safe_qsize(event_q),
                    safe_qsize(audio_q_stt),
                    safe_qsize(play_q),
                )

            drain_events(
                event_q,
                orchestrator,
                components,
                event_counts=event_counts,
                timeline_ns=timeline_ns,
                marker_delivery_ms=marker_delivery_ms,
            )

        if args.post_silence_s > 0 and not shutdown_requested.is_set():
            silence_frames = int(
                np.ceil((args.post_silence_s * args.sample_rate) / args.frame_size)
            )
            LOGGER.info(
                "Injecting trailing silence | duration=%.2fs | frames=%s",
                args.post_silence_s,
                silence_frames,
            )
            frame_duration = args.frame_size / args.sample_rate
            silence_ts = time.monotonic()
            for _ in range(silence_frames):
                if shutdown_requested.is_set():
                    LOGGER.warning("Shutdown requested during trailing silence injection")
                    break
                silence_frame = AudioFrame(
                    pcm=np.zeros(args.frame_size, dtype=np.float32),
                    intpcm=np.zeros(args.frame_size, dtype=np.int16),
                    sample_rate=args.sample_rate,
                    timestamp=silence_ts,
                )
                ring.push(silence_frame)
                if orchestrator.allow_stt_audio():
                    try:
                        audio_q_stt.put_nowait(silence_frame)
                    except Exception:
                        dropped_stt_frames += 1
                        LOGGER.debug("Dropped trailing silence frame due to full queue")
                drain_events(
                    event_q,
                    orchestrator,
                    components,
                    event_counts=event_counts,
                    timeline_ns=timeline_ns,
                    marker_delivery_ms=marker_delivery_ms,
                )
                silence_ts += frame_duration
                if not args.fast:
                    if sleep_interruptibly(frame_duration, shutdown_requested):
                        LOGGER.warning("Shutdown requested while pacing trailing silence")
                        break

        if turn_input_start is None:
            LOGGER.error("Replay produced no frames")
            return 130 if shutdown_requested.is_set() else 1

        LOGGER.info(
            "Replay complete, waiting for playback completion | replay_frames=%s | dropped_stt_frames=%s",
            replay_frame_count,
            dropped_stt_frames,
        )
        wait_deadline = time.monotonic() + args.completion_timeout_s
        next_wait_log = time.monotonic() + args.wait_log_interval_s

        while (
            time.monotonic() < wait_deadline
            and not playback_state.done.is_set()
            and not shutdown_requested.is_set()
        ):
            drain_events(
                event_q,
                orchestrator,
                components,
                block_timeout_s=0.05,
                event_counts=event_counts,
                timeline_ns=timeline_ns,
                marker_delivery_ms=marker_delivery_ms,
            )
            now = time.monotonic()
            if now >= next_wait_log:
                LOGGER.info(
                    "Wait heartbeat | playback_done=%s | first_audio=%s | event_q=%s | play_q=%s | event_counts=%s",
                    playback_state.done.is_set(),
                    playback_state.first_audio_ts is not None,
                    safe_qsize(event_q),
                    safe_qsize(play_q),
                    dict(event_counts),
                )
                next_wait_log = now + args.wait_log_interval_s

        drain_events(
            event_q,
            orchestrator,
            components,
            event_counts=event_counts,
            timeline_ns=timeline_ns,
            marker_delivery_ms=marker_delivery_ms,
        )

        if shutdown_requested.is_set():
            LOGGER.warning("Shutdown requested before playback completion")

        active_start = turn_input_start
        active_end = playback_state.playback_end_ts or time.monotonic()

        time_to_first_audio_ms = None
        if playback_state.first_audio_ts is not None:
            time_to_first_audio_ms = (playback_state.first_audio_ts - turn_input_start) * 1000.0
            _record_first_ts(
                timeline_ns,
                "playback_first_audio",
                int(playback_state.first_audio_ts * 1_000_000_000),
            )

        e2e_latency_ms = None
        if playback_state.playback_end_ts is not None:
            e2e_latency_ms = (playback_state.playback_end_ts - turn_input_start) * 1000.0

        ttft_ms = _delta_ms(timeline_ns, "intent_event", "llm_first_token")

        input_audio_duration_s = (replay_frame_count * args.frame_size) / args.sample_rate

        rtf_e2e = None
        if playback_state.playback_end_ts is not None and input_audio_duration_s > 0:
            rtf_e2e = (playback_state.playback_end_ts - turn_input_start) / input_audio_duration_s

        rtf_system = None
        stt_final_event_ns = timeline_ns.get("stt_final_event")
        if (
            playback_state.playback_end_ts is not None
            and stt_final_event_ns is not None
            and input_audio_duration_s > 0
        ):
            stt_final_event_s = stt_final_event_ns / 1_000_000_000.0
            rtf_system = (playback_state.playback_end_ts - stt_final_event_s) / input_audio_duration_s

        ttft_ms = _delta_ms(timeline_ns, "intent_event", "llm_first_token")

        input_audio_duration_s = (replay_frame_count * args.frame_size) / args.sample_rate

        rtf_e2e = None
        if playback_state.playback_end_ts is not None and input_audio_duration_s > 0:
            rtf_e2e = (playback_state.playback_end_ts - turn_input_start) / input_audio_duration_s

        rtf_system = None
        stt_final_event_ns = timeline_ns.get("stt_final_event")
        if (
            playback_state.playback_end_ts is not None
            and stt_final_event_ns is not None
            and input_audio_duration_s > 0
        ):
            stt_final_event_s = stt_final_event_ns / 1_000_000_000.0
            rtf_system = (playback_state.playback_end_ts - stt_final_event_s) / input_audio_duration_s

        cpu_idle_pct = sampler.mean_cpu_between(idle_start, idle_end)
        cpu_active_pct = sampler.mean_cpu_between(active_start, active_end)
        memory_peak_mb = sampler.peak_memory_mb_between(active_start, active_end)

        print("\n=== Phase 1 Metrics ===")
        print(format_metric("e2e_latency_ms", e2e_latency_ms, " ms"))
        print(format_metric("time_to_first_audio_ms", time_to_first_audio_ms, " ms"))
        print(format_metric("cpu_idle_pct", cpu_idle_pct, " %"))
        print(format_metric("cpu_active_pct", cpu_active_pct, " %"))
        print(format_metric("memory_peak_mb", memory_peak_mb, " MB"))

        print("\n=== Additional Metrics ===")
        print(_format_delta("ttft_ms", ttft_ms))
        print(_format_ratio("rtf_system", rtf_system))
        print(_format_ratio("rtf_e2e", rtf_e2e))

        print("\n=== Additional Metrics ===")
        print(_format_delta("ttft_ms", ttft_ms))
        print(_format_ratio("rtf_system", rtf_system))
        print(_format_ratio("rtf_e2e", rtf_e2e))

        turn_start_ns = int(turn_input_start * 1_000_000_000)
        print("\n=== Streaming Timeline (Offsets From Turn Start) ===")
        for key in [
            "stt_first_partial",
            "stt_first_partial_event",
            "stt_final_emit",
            "stt_final_event",
            "intent_event",
            "llm_first_token",
            "generation_done_event",
            "tts_first_audio_emit",
            "tts_done_event",
            "playback_first_audio",
            "playback_done_event",
        ]:
            print(_format_stage_offset(key, timeline_ns.get(key), turn_start_ns))

        print("\n=== Streaming Deltas ===")
        print(_format_delta("STT_final -> IntentEvent", _delta_ms(timeline_ns, "stt_final_event", "intent_event")))
        print(_format_delta("IntentEvent -> First LLM token", _delta_ms(timeline_ns, "intent_event", "llm_first_token")))
        print(_format_delta("First LLM token -> GenerationDone", _delta_ms(timeline_ns, "llm_first_token", "generation_done_event")))
        print(_format_delta("First LLM token -> First TTS audio", _delta_ms(timeline_ns, "llm_first_token", "tts_first_audio_emit")))
        print(_format_delta("First TTS audio -> Playback first audio", _delta_ms(timeline_ns, "tts_first_audio_emit", "playback_first_audio")))

        print("\n=== event_q Marker Delivery Lag ===")
        for marker_key in ["stt_first_partial", "stt_final_emit", "llm_first_token", "tts_first_audio_emit"]:
            lag_ms = marker_delivery_ms.get(marker_key)
            if lag_ms is None:
                print(f"{marker_key}: N/A")
            else:
                print(f"{marker_key}: {lag_ms:.2f} ms")

        LOGGER.info("Final event counts: %s", dict(event_counts))

        if playback_state.playback_end_ts is None:
            if event_counts.get("STTFinalEvent", 0) > 0 and event_counts.get("IntentEvent", 0) == 0:
                LOGGER.error(
                    "Stall diagnosis: STT final arrived but no IntentEvent was observed. "
                    "This usually indicates intent queue handoff failure."
                )
            elif event_counts.get("WakeEvent", 0) > 0 and event_counts.get("STTFinalEvent", 0) == 0:
                LOGGER.error(
                    "Stall diagnosis: wake occurred but no STT final event was observed. "
                    "Input may lack trailing silence or VAD did not finalize speech."
                )

            LOGGER.warning(
                "Playback completion was not observed before timeout. Metrics are partial. "
                "Queues | event_q=%s | play_q=%s",
                safe_qsize(event_q),
                safe_qsize(play_q),
            )

        return 130 if shutdown_requested.is_set() else 0

    finally:
        LOGGER.info("Shutdown start")
        shutdown_requested.set()

        LOGGER.info("Stopping sampler")
        sampler.stop()
        sampler.join(timeout=2.0)
        if sampler.is_alive():
            LOGGER.warning("Sampler thread did not stop before timeout")

        try:
            tts_text_q.put(
                GeneratedToken(token=None, ctx=orchestrator.turn_ctx),
                timeout=0.5,
            )
            LOGGER.debug("Sent TTS queue EOS token")
        except Full:
            LOGGER.warning("TTS queue full; could not send EOS token")
        except Exception:
            LOGGER.debug("Failed to send TTS queue EOS token", exc_info=True)

        try:
            play_q.put(None, timeout=0.5)
            LOGGER.debug("Sent playback sentinel")
        except Full:
            LOGGER.warning("Playback queue full; could not send sentinel")
        except Exception:
            LOGGER.debug("Failed to send playback sentinel", exc_info=True)

        LOGGER.info("Joining playback probe thread")
        playback_thread.join(timeout=2.0)
        if playback_thread.is_alive():
            LOGGER.warning("Playback probe thread did not stop before timeout")

        for process in processes:
            if process.is_alive():
                LOGGER.info("Terminating process %s pid=%s", process.name, process.pid)
                process.terminate()

        for process in processes:
            process.join(timeout=2.0)
            if process.is_alive():
                LOGGER.warning(
                    "Process %s pid=%s still alive after terminate",
                    process.name,
                    process.pid,
                )
            else:
                LOGGER.info(
                    "Joined process %s pid=%s exitcode=%s",
                    process.name,
                    process.pid,
                    process.exitcode,
                )

        for q, qname in [
            (audio_q_wake, "audio_q_wake"),
            (audio_q_stt, "audio_q_stt"),
            (stt_text_q, "stt_text_q"),
            (intent_q, "intent_q"),
            (prompt_q, "prompt_q"),
            (tts_text_q, "tts_text_q"),
            (play_q, "play_q"),
            (event_q, "event_q"),
        ]:
            try:
                q.cancel_join_thread()
                q.close()
                LOGGER.debug("Closed queue %s", qname)
            except Exception:
                LOGGER.debug("Failed closing queue %s", qname, exc_info=True)

        for sig, prev_handler in previous_signal_handlers.items():
            try:
                signal.signal(sig, prev_handler)
            except Exception:
                LOGGER.debug("Failed restoring signal handler for %s", sig, exc_info=True)

        LOGGER.info("Shutdown end")


def main() -> int:
    args = parse_args()
    return run_eval(args)


if __name__ == "__main__":
    # Explicitly keep Windows multiprocessing compatible.
    mp.freeze_support()
    sys.exit(main())
