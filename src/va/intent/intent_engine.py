import json
import re
from multiprocessing.queues import Queue

import ollama

from src.va.intent.types import ActionType, IntentResult, ToolCall
from src.va.ipc.events import Event, IntentEvent
from src.va.stt.types import TranscriptionMsg, TranscriptionType

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are the Brain of a Desktop Assistant. 
Analyze the user's raw speech. Output JSON ONLY.

Tools Available:
- "browser_search": Search the web. Params: {"query": "str"}
- "app_open": Open a desktop app. Params: {"app_name": "str"}
- "system_control": volume/brightness. Params: {"action": "mute"|"unmute", "value": int}

Schema:
{
  "thought": "brief reasoning",
  "action_type": "chat" | "tool_use" | "ignore",
  "refined_query": "clean version of user text",
  "tool_calls": [{"tool": "name", "params": {...}}]
}

Example:
User: "Uhh play some jazz music"
JSON: {
  "thought": "User wants music. Use youtube search.",
  "action_type": "tool_use",
  "refined_query": "Play jazz music on YouTube",
  "tool_calls": [{"tool": "browser_search", "params": {"query": "jazz music youtube"}}]
}
"""


class IntentEngine:
    def __init__(
        self,
        text_queue: Queue[TranscriptionMsg],
        event_queue: Queue[Event],
        model="qwen3:0.6b",
    ):
        self.text_queue = text_queue
        self.event_queue = event_queue
        self.model = model

        # Regex to extract JSON from "Here is your JSON: {...}"
        self.json_pattern = re.compile(r"\{.*\}", re.DOTALL)

    def run(self):
        print(f"[Intent] Worker Live ({self.model})")
        while True:
            msg: TranscriptionMsg = self.text_queue.get()

            if msg.ctx.cancelled.is_set():
                continue
            if msg.type == TranscriptionType.FINAL:
                self._predict(msg)

    def _predict(self, msg: TranscriptionMsg):
        try:
            # 1. Inference
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg.text},
                ],
                think=False,
                format="json",  # This is Ollama feature
                options={"temperature": 0.2},  # Low temp for deterministic actions
            )

            raw_content = response["message"]["content"]

            # 2. Parse & Repair
            data = self._extract_json(raw_content)

            # 3. Map to Data Class
            tool_calls = []
            for tc in data.get("tool_calls", []):
                tool_calls.append(ToolCall(tool=tc["tool"], params=tc["params"]))

            result = IntentResult(
                action_type=ActionType(data.get("action_type", "chat")),
                refined_query=data.get("refined_query", msg.text),  # Fallback to raw
                thought=data.get("thought", ""),
                tool_calls=tool_calls,
            )

            # 4. Emit
            print(f"[Intent] {result.action_type.value} | {result.refined_query}")
            self.event_queue.put(IntentEvent(result=result, ctx=msg.ctx))

        except Exception as e:
            print(f"[Intent] Failed: {e}")
            # Fallback event
            self.event_queue.put(
                IntentEvent(
                    result=IntentResult(
                        action_type=ActionType.CHAT, refined_query=msg.text
                    ),
                    ctx=msg.ctx,
                )
            )

    def _extract_json(self, text: str) -> dict:
        """Robust JSON extraction for small models."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = self.json_pattern.search(text)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    print("[Intent] Json not readable")
                    pass
            return {"action_type": "chat", "refined_query": text}
