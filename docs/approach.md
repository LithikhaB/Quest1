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



## Initial Architecture with tentative tech stack

![Initial Architecture](architecture%20diagrams/version1.svg)

---

## Modules

- **M0 — Environment & Repository Skeleton**
  - Public GitHub repository: `Quest1`
  - Folder structure, `requirements.txt`, `README.md`, `prompts.txt`
   

- **M1 — Acquisition**
  - Download video using `yt-dlp` (Python API)
  - Cache-first with post-download verification
  - Concurrent fragment downloads; ok.ru bad-IP deprioritization
  - Classified errors: `NetworkError`, `VideoUnavailableError`, `UnsupportedURLError`
    and validated against the actual `ok.ru` URL

![M1](architecture%20diagrams/M1.svg)

- **M2 — Audio Locate (sole localization signal)**
  - Extract audio using `ffmpeg`
  - Transcribe locally using OpenAI Whisper with word-level timestamps
  - Chunk long segments to <=2.5s spans
  - Match target dialogue using `rapidfuzz` sliding windows
  - Produce `CandidateWindow` (padded bounds, confidence, matched text, matched-segment bounds)
   
![M2](architecture%20diagrams/M2.svg)

- **M3 — Frame Extraction**
  - Pull the single frame at the matched segment start (`sample_frames(video, t, t)`)
  - Save it as a PNG into `data/frames/`
   
![M3](architecture%20diagrams/M3.svg)

- **M4 — Ambiguity Handling**
  - `is_ambiguous(window)`: True when audio confidence < similarity threshold
   
![M4](architecture%20diagrams/M4.svg)

- **M5 — Output Formatting**
  - `DialogueResult` dataclass: timestamp, frame number, matched text, image path,
    confidence, ambiguous flag
  - Spec-format report printed and saved to `data/processed/result.txt`
   
![M5](architecture%20diagrams/M5.svg)

- **M6 — Streamlit Frontend**
  - URL + dialogue inputs, Run button
  - Timestamp/frame/confidence metrics, ambiguity warning, matched text, inline frame image
  
![M6](architecture%20diagrams/M6.svg)
   

- **M7 — Documentation & Interview Preparation**
  - `README.md` (how to run), `docs/approach.md`, `prompts.txt`
  - Rehearse design decisions: why audio-only, word-level chunking, caching, ambiguity
