import time
from queue import Empty
from typing import Dict, List

from src.va.intent.types import ActionType
# Event Types
from src.va.ipc.events import (Event, GenerationDoneEvent, IntentEvent,
                               PlayBackEvent, STTFinalEvent, STTPartialEvent,
                               TTSDoneEvent, WakeEvent)
from src.va.response.types import GenerationTask
from src.va.stt.types import TranscriptionMsg, TranscriptionType

from .turn_context import TurnContext


class Orchestrator:
    """
    Manages State, History, and Routing.
    """

    def __init__(self, *, parallel_intent: bool = False, disable_intent: bool = False):
        self._state = "IDLE"
        self.parallel_intent = parallel_intent and not disable_intent
        self.disable_intent = disable_intent
        # System Prompt for the Response LLM (Personality)
        self.system_prompt = (
            "You are a helpful desktop assistant. Keep answers concise. "
            "Do not output markdown or emojis unless necessary."
        )
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]  # Conversation History, with system prompt appended in starting

        # Turn context
        self.turn_ctx = TurnContext(1)
        self._pending_transcripts: Dict[int, str] = {}

    def allow_stt_audio(self) -> bool:
        return self._state == "LISTENING"

    # -----------------------
    # Event Handling
    # -----------------------
    def handle_event(self, event: Event, components: dict) -> None:
        """
        components dict contains all queues:
        {
            'stt_audio_q': Queue[AudioFrame],
            'intent_q': Queue,
            'response_q': Queue[str],
            'playback_q': Queue[TTsAudio],  # For clearing on interrupt
            'ring_buffer': RingBuffer
        }
        """
        if isinstance(event, WakeEvent):
            self._on_wake(event, components)

        elif isinstance(event, STTPartialEvent):
            # Purely for UI feedback, logic doesn't care
            print(f"\r[User (Partial)]: {event.text}", end="", flush=True)

        elif isinstance(event, STTFinalEvent):
            self._on_stt_final(event, components)

        elif isinstance(event, IntentEvent):
            self._on_intent(event, components)

        elif isinstance(event, GenerationDoneEvent):
            self._on_generation_done(event)

        elif isinstance(event, TTSDoneEvent):
            self._on_tts_done(event)
        elif isinstance(event, PlayBackEvent):
            self._on_play_done()

    # -----------------------
    # Logic Handlers
    # -----------------------

    def _on_wake(self, event: WakeEvent, comps: dict) -> None:
        print("\n[Orchestrator] WAKE DETECTED")

        # --- BARGE-IN LOGIC ---
        if self._state in ["SPEAKING", "THINKING"]:
            print("[Orchestrator] Interrupting previous action...")
            # cancel all older context
            self.turn_ctx.cancelled.set()
            # Create new context. Well this is simple, there is no persistence in sessions
            self.turn_ctx = TurnContext(turn_id=self.turn_ctx.turn_id + 1)

        self._state = "LISTENING"

        # Dump pre-roll audio
        if comps["ring_buffer"]:
            for frame in comps["ring_buffer"].dump():
                comps["stt_audio_q"].put(frame)

    def _on_stt_final(self, event: STTFinalEvent, comps: dict) -> None:
        if self._state != "LISTENING":
            return  # Ignore ghost events

        print(f"\n[User (Final)]: {event.text}")
        self._state = "THINKING"
        self._pending_transcripts[self.turn_ctx.turn_id] = event.text

        if self.disable_intent:
            self._dispatch_response(event.text, comps, self.turn_ctx)
            return

        if self.parallel_intent:
            self._dispatch_response(event.text, comps, self.turn_ctx)

        # Forward to Intent Engine
        # STT to Intent is managed by orchestrator, well thats because if later we need to handle
        # streaming STT to intent, via debouncing using buffer.
        self._forward_to_intent(event.text, comps, self.turn_ctx)

    def _on_intent(self, event: IntentEvent, comps: dict) -> None:
        if event.ctx.cancelled.is_set():
            return
        result = event.result
        print(
            f"[Orchestrator] Intent: {result.action_type.value} | Query: {result.refined_query}"
        )

        if self.parallel_intent:
            if event.ctx.turn_id != self.turn_ctx.turn_id:
                return
            if result.action_type == ActionType.TOOL_USE:
                self._interrupt_and_reroute(result, comps)
            return

        self._dispatch_from_intent_result(result, comps, event.ctx)

    def _dispatch_from_intent_result(
        self,
        result,
        comps: dict,
        ctx: TurnContext,
    ) -> None:
        query_text = (result.refined_query or "").strip()
        if not query_text:
            return

        self._maybe_run_tools(result)
        self._dispatch_response(query_text, comps, ctx)

    def _interrupt_and_reroute(self, result, comps: dict) -> None:
        print("[Orchestrator] Intent requested reroute; interrupting current turn")
        old_ctx = self.turn_ctx
        old_turn_id = old_ctx.turn_id
        old_ctx.cancelled.set()
        self._clear_queue(comps.get("playback_q"))

        self.turn_ctx = TurnContext(turn_id=old_turn_id + 1)
        reroute_query = (result.refined_query or self._pending_transcripts.pop(old_turn_id, "")).strip()
        if not reroute_query:
            self._state = "IDLE"
            return

        self._maybe_run_tools(result)
        self._dispatch_response(reroute_query, comps, self.turn_ctx)

    def _maybe_run_tools(self, result) -> None:
        # 2. Handle Tools (The Hub Logic)
        # Need to Implement, but currently not required for myself.
        if result.action_type != ActionType.TOOL_USE:
            return

        print(f"[Orchestrator] Executing Tools: {result.tool_calls}")
        # --- EXECUTE TOOLS HERE ---
        # output = tool_registry.execute(result.tool_calls)
        # For now, simulate:
        output = "Simulated Tool Output: Operation Successful."
        # Here we have to diverge to differentiate -- tool usage v/s desktop control --
        # Add Tool Output to History for tool usage, like google search
        self.history.append({"role": "system", "content": f"Tool Result: {output}"})
        # if desktop control - execute python scripts to control desktop and simulating Keyboard keys
        # No generating task needed, unlesss explicit stated, well that needs more sophisticated system
        # --- to be implemented ---

    def _dispatch_response(self, text: str, comps: dict, ctx: TurnContext) -> None:
        self.history.append({"role": "user", "content": text})

        # 3. Construct Prompt for Response LLM
        task = GenerationTask(
            messages=self.history,
            ctx=ctx,
        )

        # 4. Dispatch
        # Switch state early because TTS streams immediately, but later I need Thinking if I set qwen to thinking mode..
        self._state = "SPEAKING"
        comps["response_q"].put(task)

    def _forward_to_intent(self, text: str, comps: dict, ctx: TurnContext) -> None:
        intent_q = comps.get("intent_q")
        if intent_q is None:
            return
        intent_q.put(
            TranscriptionMsg(
                text=text,
                type=TranscriptionType.FINAL,
                timestamp=time.time(),
                ctx=ctx,
            )
        )

    def _clear_queue(self, q) -> None:
        if q is None:
            return
        while True:
            try:
                q.get_nowait()
            except Empty:
                break
            except Exception:
                break

    def _on_generation_done(self, event: GenerationDoneEvent) -> None:
        """
        Response LLM finished generating text. Append History to keep track of memory.
        """
        if (event.ctx.cancelled.is_set()):
            return
        print(f"\n[Assistant]: {event.full_text}")
        self.history.append({"role": "assistant", "content": event.full_text})
        self._pending_transcripts.pop(event.ctx.turn_id, None)
        if len(self.history) > 10:
            self.history = self.history[-10:]

    def _on_tts_done(self, event: TTSDoneEvent) -> None:
        """
        Audio finished processing
        """
        print("[Orchestrator] TTS finished processing!")
        # For now nothing to do, but maybe in Future, I will need this, who knows.

    def _on_play_done(self) -> None:
        print("[Orchestrator] Assistant going Idle!")
        self._state = "IDLE"

    # -----------------------
    # No Helpers Required For now <----------
    # -----------------------
    # def _clear_queue(self, q: multiprocessing.Queue):
    #     try:
    #         while not q.empty():
    #             q.get_nowait()
    #     except multiprocessing.Queue.empty:
    #         pass
