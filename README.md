# Dialogue Frame Locator — Quest1

A local-first pipeline that locates the **exact video frame where a line of dialogue is spoken**
in any video URL, saves that frame as an image, and reports timestamp, frame number, extracted
text, and confidence — with an explicit ambiguity flag when the match is weak.

No paid APIs, no API keys: download, transcription, matching, and frame extraction all run
locally, and every stage is cached.

---

## Key Features

- **Word-Level Timestamp Alignment** — OpenAI Whisper runs with `word_timestamps=True`; long
  segments are re-chunked to <=2.5s spans so the match window is frame-precise, not paragraph-sized.
- **Exact Frame Pinpointing** — the frame is extracted at the matched transcript segment's start
  via OpenCV seek and saved as a PNG (`data/frames/match_frame_<n>_<t>s.png`).
- **Typo- and Noise-Tolerant Matching** — `rapidfuzz` sliding-window scoring absorbs typos
  ("tempt" -> "attempt"), punctuation, casing, and ASR mishearings; a minimum-word guard rejects
  degenerate substring matches.
- **Honest Ambiguity Handling** — matches below the confidence threshold (default 0.8) are
  explicitly flagged `[AMBIGUOUS]`, never silently trusted.
- **Cache-First Everything** — video downloads (by file existence), audio (by file), transcripts
  (by audio SHA-256 + model), and the final report are all cached; reruns take seconds.
- **Resilient Acquisition** — classified download errors, parallel HLS fragment downloads,
  automatic proxy-then-direct route switching, and a DNS patch for a known-dead ok.ru edge node.
- **Streamlit Frontend** — URL + dialogue inputs, run metrics, ambiguity warning, and the saved
  frame displayed inline.

---

## Architecture

![version1.svg](docs/architecture%20diagrams/version1.svg)

## Example Outputs & Verified Benchmarks

Real pipeline runs across different sources, video lengths, and query conditions:

| Video Source | Target Dialogue (as typed) | Detected Timestamp | Frame | Transcribed Text | Confidence |
|---|---|---|---|---|---|
| [ok.ru/video/248244667877](https://ok.ru/video/248244667877) *(Sherlock Holmes)* | `my mind rebels at stagnation` | `00:05:25.300` | 7,799 | "My mind rebels at stagnation." | 1.00 `[CONFIDENT]` |
| [ok.ru/video/248244667877](https://ok.ru/video/248244667877) *(same 50-min video)* | `why should i tempt to conceal it` — **note the typo** | `00:13:00.700` | 18,718 | "Why should I attempt to conceal it?" | 0.94 `[CONFIDENT]` |
| [youtube: iLBzpjQusiQ](https://www.youtube.com/watch?v=iLBzpjQusiQ) *(13s short)* | `to the right` | `00:00:06.780` | 203 | "to the right" | 0.91 `[CONFIDENT]` |

The typo row is the demo to watch: the query said *tempt*, the actor said *attempt* — the fuzzy
matcher absorbed it and still anchored the exact frame. Cold run on the 50-minute video:
~10 minutes (download + transcribe). Warm rerun: **~3 seconds** (all caches hit).

---

## Demo

> **Demo video coming soon.** This section will embed a short walkthrough of a live run
> (cold start, cached rerun, and the typo-tolerance demo) once recorded.

---

## Documentation

All supporting documents live in [`docs/`](docs/):

| Document | Contents |
|---|---|
| [`docs/approach.md`](docs/approach.md) | Approaches thought before development: naive (OCR sliding window) → implemented (audio-first) → two future-scope designs, with pros/cons of each |
| [`docs/ERRORS_FACED.txt`](docs/ERRORS_FACED.txt) | 12 errors faced during development — symptom, root cause, fix |
| [`docs/log.md`](docs/log.md) | Detailed build log: module-by-module decisions with architecture diagrams |
| [`docs/prompt.txt`](docs/prompt.txt) | AI prompt log (approach evaluation) |
| [`docs/architecture diagrams/`](docs/architecture%20diagrams/) | SVG diagrams: initial design + one per module + final design |

---

## Prerequisites

1. **Python** 3.10+
2. **ffmpeg / ffprobe** on PATH (audio extraction + duration probing)

## Installation & Setup

### Option 1: Automated scripts

```powershell
# Windows (PowerShell / cmd)
setup.bat
```

```bash
# Linux / macOS
bash setup.sh
```

Each script creates the virtual environment, installs dependencies, and runs the unit test suite
so you know the clone works before you use it.

### Option 2: Manual

```powershell
python -m venv .venv
.venv\Scripts\activate            
pip install -r requirements.txt
python -m pytest tests/ -m "not integration"   
```

## Running the Application

### 1. CLI (one command)

```powershell
python -X utf8 -c "import logging; logging.basicConfig(level=logging.INFO); from src.pipeline import run_pipeline; run_pipeline('https://ok.ru/video/248244667877', 'my mind rebels at stagnation')"
```

### 2. Streamlit UI

```powershell
streamlit run app.py
```

Enter the video URL and target dialogue, press **Run**. The UI shows timestamp / frame /
confidence metrics, an ambiguity warning when applicable, the matched transcript text, and the
saved frame inline.

---

## Project Structure

```
Quest1/
├── app.py                        # Streamlit frontend (M6)
├── setup.bat / setup.sh          # One-command setup + test
├── requirements.txt              # Python dependencies
├── docs/                         # All supporting documents (see Documentation section)
│   ├── approach.md
│   ├── ERRORS_FACED.txt
│   ├── log.md
│   ├── prompt.txt
│   └── architecture diagrams/    # SVG pipeline diagrams (initial + per-module)
├── src/
│   ├── acquisition/              # M1: yt-dlp download, error classification, route switching
│   │   ├── downloader.py
│   │   └── exceptions.py
│   ├── transcription/            # M2a: ffmpeg audio extraction + Whisper word-chunked STT
│   │   ├── extractor.py
│   │   ├── transcriber.py
│   │   └── exceptions.py
│   ├── localization/             # M2b/M4: fuzzy window matching + ambiguity flag
│   │   ├── locator.py
│   │   ├── ambiguity.py
│   │   ├── models.py
│   │   └── exceptions.py
│   ├── frames/                   # M3: OpenCV frame sampling + PNG saving
│   │   ├── sampler.py
│   │   └── exceptions.py
│   ├── output/                   # M5: DialogueResult model + report formatting
│   │   ├── models.py
│   │   ├── result.py
│   │   └── formatter.py
│   ├── pipeline.py               # End-to-end orchestration
│   ├── config.py                 # Env-driven settings
│   └── constants.py              # Every tunable in one place
├── tests/                        # 20 unit tests + 4 integration tests
└── data/                         # All caches (raw/, audio/, frames/, processed/)
```

---

Submitted by: 
Lithikha B
