import time
from multiprocessing.queues import Queue

from src.va.ipc.events import Event, GenerationDoneEvent, PipelineMarkerEvent
from src.va.response.llm_engine import LLMEngine

from .types import GeneratedToken, GenerationTask


class ResponseWorker:
    def __init__(
        self,
        prompt_queue: Queue[GenerationTask],  # IN: From Master
        tts_text_queue: Queue[GeneratedToken],  # OUT: To TTS
        event_queue: Queue[Event],  # OUT: To Master (Signal)
        model_name: str,
    ):
        self.prompt_queue = prompt_queue
        self.tts_text_queue = tts_text_queue
        self.event_queue = event_queue

        # Initialize Engine
        print(f"[Response] Loading Model: {model_name}...")
        self.engine = LLMEngine(model_name)

    def run(self):
        print("[Response] Worker Ready.")

        while True:
            try:
                task: GenerationTask = self.prompt_queue.get()

                if task.ctx.cancelled.is_set():
                    continue

                print(f"[Response] Generating for Request: {task.ctx.turn_id}")

                full_response_accumulator = []
                first_token_emitted = False

                # Direct Stream Loop
                for token in self.engine.generate_stream(task.messages):
                    if not first_token_emitted:
                        self.event_queue.put(
                            PipelineMarkerEvent(
                                marker="llm_first_token",
                                t_mono_ns=time.monotonic_ns(),
                                turn_id=task.ctx.turn_id,
                            )
                        )
                        first_token_emitted = True
                    self.tts_text_queue.put(GeneratedToken(token=token, ctx=task.ctx))

                    full_response_accumulator.append(token)

                # Special Token to TTS (Stop Speaking)
                self.tts_text_queue.put(GeneratedToken(
                    token=None,
                    ctx=task.ctx))  # Using None token as EOS signal

                # B. Send Done Event to Orchestrator
                full_text = "".join(full_response_accumulator)

                self.event_queue.put(
                    GenerationDoneEvent(full_text=full_text, ctx=task.ctx)
                )

                print("[Response] Task Complete.")

            except Exception as e:
                print(f"[Response] Worker Error: {e}")
