# Extract Frame from Video URL

## Core Functionalities

- Download video from URL using `yt-dlp`
  - Supports the given `ok.ru` URL
  - Cache-first: skip all network access when the video already exists locally
  - Handle extractor/download failures with a clean, classified error path
  - Parallel fragment downloads for fast HLS acquisition

- Extract audio locally using `ffmpeg` (16 kHz mono WAV, cached)

- Transcribe locally using **Whisper** (`tiny` by default) with **word-level timestamps**
  - Long segments chunked into <=2.5s spans so match windows stay frame-precise
  - Transcript cached on disk keyed by audio hash + model

- Search the transcript for the target dialogue using `rapidfuzz`
  - Sliding-window fuzzy matching over transcript chunks
  - Produce candidate window + match confidence + exact matched-segment bounds

- Extract the single representative frame at the exact moment the dialogue is spoken
  - OpenCV seek to the matched segment start
  - Save the frame as a PNG image into `data/frames/`

- Produce required output:
  - Timestamp (`HH:MM:SS.mmm`)
  - Frame number
  - Extracted dialogue text (the verified transcript segment)
  - Saved frame image path
  - Confidence score

- Handle uncertainty
  - Confidence threshold (default 0.8)
  - Mark results as ambiguous when the audio match confidence is below the threshold


## Nice-to-Have — If Time Remains

- Whisper model upgrade via config
  - `WHISPER_MODEL_SIZE=base|small` in `.env` for noisy audio, ~2x runtime

- CLI entry point (`python main.py --url <URL> --dialogue "<text>"`)
  - Never hardcode the sample video/line

- Dockerfile: containerize the application

- Caption-aware mode for future videos
  - If a video *does* have burned-in subtitles, optionally re-add visual verification
    (design preserved in `approach_audio_only.md` / git history)

- Batch mode: multiple dialogues per video in one run (transcript reused)


## Initial Architecture with tentative tech stack

![Initial Architecture](architecture%20diagrams/version1.svg)

---

## Modules

- **M0 — Environment & Repository Skeleton**
  - Public GitHub repository: `Quest1`
  - Folder structure, `requirements.txt`, `README.md`, `prompts.txt`
  - Complete

- **M1 — Acquisition**
  - Download video using `yt-dlp` (Python API)
  - Cache-first with post-download verification
  - Concurrent fragment downloads; ok.ru bad-IP deprioritization
  - Classified errors: `NetworkError`, `VideoUnavailableError`, `UnsupportedURLError`
  - Complete and validated against the actual `ok.ru` URL
![M1](architecture%20diagrams/M1.svg)

- **M2 — Audio Locate (sole localization signal)**
  - Extract audio using `ffmpeg`
  - Transcribe locally using OpenAI Whisper with word-level timestamps
  - Chunk long segments to <=2.5s spans
  - Match target dialogue using `rapidfuzz` sliding windows
  - Produce `CandidateWindow` (padded bounds, confidence, matched text, matched-segment bounds)
  - Complete
![M2](architecture%20diagrams/M2.svg)

- **M3 — Frame Extraction**
  - Pull the single frame at the matched segment start (`sample_frames(video, t, t)`)
  - Save it as a PNG into `data/frames/`
  - Complete

- **M4 — Ambiguity Handling**
  - `is_ambiguous(window)`: True when audio confidence < similarity threshold
  - Complete

- **M5 — Output Formatting**
  - `DialogueResult` dataclass: timestamp, frame number, matched text, image path,
    confidence, ambiguous flag
  - Spec-format report printed and saved to `data/processed/result.txt`
  - Complete

- **M6 — Streamlit Frontend**
  - URL + dialogue inputs, Run button
  - Timestamp/frame/confidence metrics, ambiguity warning, matched text, inline frame image
  - Complete

- **M7 — Documentation & Interview Preparation**
  - `README.md` (how to run), `docs/approach_audio_only.md` (final design rationale),
    `sample.md` (build log), `prompts.txt`
  - Rehearse design decisions: why audio-only, word-level chunking, caching, ambiguity
  - Validate with a second video/dialogue pair; nothing hardcoded to the sample input

## Why audio-only (design pivot)

The target video delivers the dialogue as **speech with no on-screen captions**. Visual OCR
verification was built, run against the real video, and removed: it could only add failure
modes (API rate limits, watermark false positives, quota exhaustion) without adding
information. The full rationale and the preserved OCR design live in
`approach_audio_only.md` and `sample.md`.
