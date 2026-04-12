from dataclasses import dataclass
from typing import ClassVar

from src.va.intent.types import IntentResult
from src.va.orchestrator.turn_context import TurnContext


@dataclass(frozen=True)
class Event:
    type: ClassVar[str]


@dataclass(frozen=True)
class WakeEvent(Event):
    type: ClassVar[str] = "WAKE"


@dataclass(frozen=True)
class STTPartialEvent(Event):
    text: str
    type: ClassVar[str] = "STT_PARTIAL"


@dataclass(frozen=True)
class STTFinalEvent(Event):
    text: str
    type: ClassVar[str] = "STT_FINAL"


@dataclass(frozen=True)
class IntentEvent(Event):
    result: IntentResult
    ctx: TurnContext
    type: ClassVar[str] = "INTENT_FINAL"


@dataclass(frozen=True)
class GenerationDoneEvent(Event):
    full_text: str
    # request_id: str
    ctx: TurnContext
    type: ClassVar[str] = "GEN_DONE"


@dataclass(frozen=True)
class TTSDoneEvent(Event):
    type: ClassVar[str] = "TTS_DONE"


@dataclass(frozen=True)
class PlayBackEvent(Event):
    type: ClassVar[str] = "PLAY_DONE"


@dataclass(frozen=True)
class PipelineMarkerEvent(Event):
    marker: str
    t_mono_ns: int
    turn_id: int | None = None
    type: ClassVar[str] = "PIPELINE_MARKER"
