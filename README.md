# Voice Assistant V2

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-CPU--Only-orange)
![Architecture](https://img.shields.io/badge/Architecture-Multiprocessing%20%7C%20Event--Driven-purple)
![Inference](https://img.shields.io/badge/Inference-Local%20%7C%20Offline-brightgreen)

> **A CPU-first, event-driven, local voice assistant built with a turn-aware multiprocessing architecture for low-latency, interruption-safe interaction.**

---

## TL;DR

**Voice Assistant V2** is a local, edge-oriented voice assistant designed to run reliably on consumer laptops using CPU-only inference.
Instead of relying on cloud APIs or sequential pipelines, it uses **process isolation, streaming-first design, and turn-based cancellation** to deliver responsive, interruption-safe voice interactions.

The system is intentionally optimized to run **as a background service** without monopolizing system resources, while still supporting more aggressive speculative execution when deployed on dedicated edge hardware.

---

## Overview

Most DIY voice assistants prioritize feature demos over architectural rigor—often relying on blocking API calls, monolithic pipelines, or GPU-heavy inference.
**Voice Assistant V2** is engineered with a different goal: **predictable, low-latency local interaction on CPUs**, even under constrained system conditions.

The architecture uses quantized ONNX/GGUF models and a **Hybrid Hub-and-Spoke multiprocessing design** that cleanly separates:

- **Control flow** (conversation state, turn lifecycle, routing)
- **Data flow** (audio frames, tokens, PCM streams)
- **Compute-heavy inference** (STT, LLMs, TTS)

This allows Wake Word detection, Speech Recognition, Intent Parsing, and Speech Synthesis to execute **concurrently**, without being blocked by Python’s GIL.

### Design Goals

- **Local-first execution** (no cloud inference)
- **CPU-only inference**
- **Low time-to-first-audio**
- **Interruption-safe (barge-in) interaction**
- **Production-style control/data separation**

---

## System Architecture

The assistant consists of multiple isolated worker processes communicating via IPC queues.
At the center is a **turn-aware Orchestrator**, responsible for conversation lifecycle management—not heavy computation.

### Core Data Flow

1. **Audio Input**
   Raw `Int16` / `Float32` audio frames are multicast to Wake Word and STT workers.

2. **Pause-Aware VAD**
   A two-threshold VAD distinguishes between natural speech pauses and utterance completion.

3. **Speech-to-Text (STT)**
   Transcriptions are emitted as partial and final events.

4. **Intent Parsing**
   A local LLM converts finalized text into structured JSON (`chat` vs `tool_use`).

5. **Streaming Response Path**
   Generated tokens bypass the Orchestrator and stream directly to the TTS worker.

6. **Incremental Playback**
   Audio is synthesized and played chunk-by-chunk for low perceived latency.

---

## Key Technical Innovations

### 1. Two-Threshold Pause-Aware VAD

Instead of relying on a single silence timeout, the system distinguishes between:

- **Micro-Pause (~0.3s)**
  → Used for partial transcription and UI feedback.
- **Macro-Pause (~2.0s)**
  → Commits the utterance and triggers intent execution.

This makes the assistant feel responsive without cutting users off mid-thought.

---

### 2. Turn-Aware Orchestration (Not Just an FSM)

The Orchestrator is **not** a simple finite state machine.

It acts as a **turn-based control plane**, responsible for:

- Managing assistant state (`IDLE → LISTENING → THINKING → SPEAKING`)
- Assigning a **TurnContext** to each user interaction
- Propagating cancellation signals across workers
- Preventing stale or late IPC events from corrupting state
- Owning bounded conversation history and prompt construction

This design mirrors production voice systems, where **control flow and data flow are intentionally decoupled**.

---

### 3. Speculative Execution (Supported, Intentionally Throttled)

The system supports **speculative intent decoding** from partial STT events.
However, speculative decoding is **intentionally disabled by default**.

#### Why?

- This assistant is designed to run **as a background process on a laptop**
- Continuous speculative decoding significantly increases CPU utilization
- Sustained high CPU usage can cause thermal throttling and degrade overall system usability

Instead, speculative decoding is treated as a **deployment-time optimization**:

- **Enabled** on dedicated edge devices
- **Disabled** for background laptop usage

This reflects a deliberate engineering tradeoff, not a missing feature.

---

### 4. CPU-Optimized Local Inference

All inference components are selected and tuned for CPU efficiency:

- Quantized model weights (ONNX int4) for moonshine
- Onnx model for piper and Silero vad
- LLM token streaming via Ollama (GGUF format)
- Streaming-friendly generation
- No GPU assumptions
- Predictable memory usage

---

## Tech Stack

| Component     | Technology               | Notes                                  |
| ------------- | ------------------------ | -------------------------------------- |
| Orchestration | Python `multiprocessing` | Turn-aware, event-driven control plane |
| Wake Word     | Picovoice Porcupine      | High recall, ultra-low CPU usage       |
| VAD           | Silero VAD v4 (ONNX)     | Custom debouncing + padding            |
| STT           | Moonshine (ONNX Int4)    | Sliding buffer with context            |
| Intent Engine | Ollama (Qwen3-0.6B)      | Structured JSON extraction             |
| Response LLM  | Ollama (Qwen/Llama)      | Token-level streaming                  |
| TTS           | Piper (ONNX)             | Faster-than-realtime synthesis         |

---

## Module Overview

### `src/va/orchestrator`

The system control layer.
Coordinates events across isolated processes without handling raw audio or model inference.

Key responsibilities:

- Turn-scoped cancellation for safe concurrency
- Barge-in handling and interruption recovery
- Ghost-event suppression
- Early state transitions to support streaming
- Bounded conversation memory

---

### `src/va/stt`

Combines Silero VAD and Moonshine STT using a sliding audio buffer and pause-aware transcription logic.

---

### `src/va/intent`

Normalizes user speech into structured JSON:

```json
{
  "action_type": "tool_use",
  "refined_query": "Play jazz music on YouTube",
  "tool_calls": [
    {
      "tool": "browser_search",
      "params": { "query": "jazz music youtube" }
    }
  ]
}
```

---

### `src/va/response` & `src/va/tts`

Optimized for latency:

- Text streams to TTS as soon as sentence boundaries are detected
- Audio streams to playback immediately as PCM chunks are produced

---

## Turn-Based Conversation Model

Each user interaction is modeled as an **independent turn**.

Each turn:

- Has a unique ID
- Carries a shared cancellation token
- Can be aborted safely at any stage

### Barge-In Handling

If a wake word is detected while the assistant is speaking or thinking:

1. The active turn is cancelled
2. All downstream workers discard stale events
3. A new turn begins immediately

This enables natural interruptions without overlapping audio or race-condition artifacts.

---

## Performance Optimizations

- `spawn` multiprocessing context (safe with ONNX / PyTorch)
- Ring buffers with audio pre-roll
- Dual audio formats (`Int16` for speed, `Float32` for accuracy)
- Direct queue handoff for streaming paths
- Turn-scoped cancellation to avoid wasted compute

---

## Performance Benchmarks

The project includes a deterministic evaluation harness (`eval/run_eval.py`) that simulates a real user interaction with a pre-recorded audio file.
Latest stable run was executed under controlled conditions (no background load, near-max CPU allocation to eval):

```bash
uv run python -m eval.run_eval --wav .\eval\scenarios\audio16.wav --startup-wait-s 3 --idle-window-s 1 --completion-timeout-s 30 --log-level WARNING --disable-intent
```

### Benchmark Summary

| Metric                                         |     Result | Why it matters                                                                 |
| ---------------------------------------------- | ---------: | ------------------------------------------------------------------------------ |
| End-to-End Latency (`e2e_latency_ms`)          | 4125.00 ms | Full turn duration from input start to playback completion                     |
| Time to First Audio (`time_to_first_audio_ms`) | 3844.00 ms | User-perceived response start                                                  |
| Time to First Token (`ttft_ms`)                |        N/A | Intent was intentionally bypassed in eval-only mode (`--disable-intent`)       |
| System RTF (`rtf_system`)                      |       0.38 | Post-STT-final pipeline efficiency (Intent/LLM/TTS/Playback processing window) |
| End-to-End RTF (`rtf_e2e`)                     |       2.39 | Full UX factor including speech + silence + processing                         |
| CPU Idle (`cpu_idle_pct`)                      |      7.86% | Baseline resource footprint                                                    |
| CPU Active (`cpu_active_pct`)                  |    419.03% | Aggregate multi-core utilization during active turn                            |
| Peak Memory (`memory_peak_mb`)                 |  697.27 MB | Peak RSS measured during the active window, excluding Ollama model loading.                                                  |

### Stage Timing Highlights

| Stage Delta                             |    Result |
| --------------------------------------- | --------: |
| STT final -> first LLM token            | 234.00 ms |
| First LLM token -> first TTS audio      | 141.00 ms |
| First TTS audio -> playback first audio |   0.00 ms |

These timings confirm efficient downstream streaming once STT finalization is reached.

---

## Offline Considerations

The assistant runs fully locally **after initialization**.

- Wake Word detection uses **Picovoice Porcupine**
- A one-time internet connection may be required for access-key validation
- After validation, all audio processing and inference runs offline

Porcupine was chosen intentionally for its unmatched **CPU efficiency, robustness, and ease of custom wake-word creation**.

---

### References & Acknowledgements

This project integrates and builds upon several high-quality open-source models and tools.
The original authors deserve explicit credit for their work.

- **Moonshine (STT)**
  Offline speech recognition models used via ONNX for CPU-efficient transcription.
  Repository: [https://github.com/moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine)

- **Silero VAD**
  Voice activity detection models used with custom debouncing and pause-aware logic.
  Repository: [https://github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad)

- **Piper TTS**
  Offline text-to-speech synthesis with streaming PCM output.
  Licensed under GPL-3.0 and used as a standalone component.
  Repository: [https://github.com/OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)

- **Picovoice Porcupine**
  Wake word detection engine selected for efficiency and robustness on CPUs.
  Repository: [https://github.com/Picovoice/porcupine](https://github.com/Picovoice/porcupine)

- **Ollama**
  Local LLM runtime used for CPU-based inference and token-level streaming.
  Repository: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)

All system-level architecture, orchestration logic, multiprocessing design, streaming paths, and turn-based control were designed and implemented in this project.

---

### Citation

If you use this project in academic or research work, please consider citing the original papers below.

<details>
<summary><strong>Moonshine</strong></summary>

```bibtex
@misc{jeffries2024moonshinespeechrecognitionlive,
  title={Moonshine: Speech Recognition for Live Transcription and Voice Commands},
  author={Nat Jeffries and Evan King and Manjunath Kudlur and Guy Nicholson and James Wang and Pete Warden},
  year={2024},
  eprint={2410.15608},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2410.15608}
}
```

</details>

<details>
<summary><strong>Silero VAD</strong></summary>

```bibtex
@misc{Silero VAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  commit = {insert_some_commit_here},
  email = {hello@silero.ai}
}
```

</details>

---

## Getting Started

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/) running
- `portaudio`

### Installation

```bash
git clone https://github.com/SrabanMondal/voice-assistant-v2.git
cd voice-assistant-v2

uv sync
ollama pull qwen3:0.6b
```

> Download Moonshine and Piper ONNX models into the `models/` directory.

### Run

```bash
python main.py
```

---

## Model Weights

This repository does not redistribute model weights.

The assistant relies on pre-trained models provided by their respective authors (Moonshine, Silero, Piper, Ollama).
Due to licensing, size, and distribution considerations, users are expected to obtain the weights directly from the official repositories.

The inference pipeline, configuration, and integration logic in this project are designed to work with specific ONNX/GGUF model variants.

> Note:
> If you are experimenting with this project and need guidance on compatible model variants or configurations, feel free to reach out.

---

## Roadmap

- [ ] RAG-based long-term memory
- [ ] Vision-to-Intent (LLaVA)
- [ ] MQTT / Home Assistant integration
- [ ] Desktop & OS-level automation tools

---

## License Notes

This project is released under the MIT License.

It integrates third-party components with different licenses:

- Moonshine (MIT)
- Silero VAD (MIT)
- Piper TTS (GPL-3.0, used as a standalone component)

Users are responsible for complying with the licenses of any third-party tools they install.

---

> _Built with care, queues, and a deliberate focus on CPU-first, low-latency systems design._

---
