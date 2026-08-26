# Build Log
Detailed summary of how this project was built, what changed, and why.

## Initial architecture

![Initial architecture diagram](architecture%20diagrams/inital.png)

## Final architecture 

![Final architecture](architecture%20diagrams/version1.svg)

## What was built, in order

### M1 — Acquisition (kept, hardened)
- `download_video(url)` with yt-dlp Python API: metadata pass, cache check before any network,
  post-download verification, exception classification into
  `NetworkError` / `VideoUnavailableError` / `UnsupportedURLError`.
- `concurrent_fragment_downloads=8` for fast HLS downloads (ok.ru serves fragmented HLS).
- DNS patch deprioritizes a known-dead ok.ru edge IP (`95.163.61.73`).
- Route switch: configured proxy first, direct connection as automatic fallback on
  NetworkError — one bad proxy config can no longer break working downloads.

![M1 — Acquisition](architecture%20diagrams/M1.svg)

### M2 — Audio localization (kept, upgraded)
- `extract_audio` (ffmpeg -> 16kHz mono WAV, cached).
- `transcribe_audio` — OpenAI Whisper `tiny`, **`word_timestamps=True`**.
- `_chunk_words_into_segments` — regroups words so no transcript segment spans >2.5s.
  This was the single biggest precision win: Whisper merged ~10s of speech into one segment,
  making candidate windows ~13s wide; after chunking, windows are ~2s.
- `locate_candidate_window` — rapidfuzz sliding window over chunks with a minimum-word guard
  (rejects substring flukes like "it" matching a 7-word target at fake 100%).
- Transcript cache keyed on audio SHA-256 + model + `_words` format marker.

![M2 — Audio Locate](architecture%20diagrams/M2.svg)

### M3 — Frame extraction
- `_extract_representative_frame` — `sample_frames(video, t, t)` seeks to the matched segment
  start and reads that single frame.
- `save_frame_image` — `cv2.imwrite` into `data/frames/match_frame_<n>_<t>s.png`.

![M3 — Frame Extraction](architecture%20diagrams/M3.svg)

### M4 — Ambiguity
- `is_ambiguous(window)` — `window.confidence < settings.similarity_threshold` (default 0.8).

![M4 — Ambiguity](architecture%20diagrams/M4.svg)

### M5 — Output
- `DialogueResult(timestamp_seconds, frame_index, matched_text, image_path, confidence, is_ambiguous)`.
- `render_report` — spec format with `HH:MM:SS.mmm` timestamps; saved to `data/processed/result.txt`.

![M5 — Output](architecture%20diagrams/M5.svg)

### M6 — Streamlit
- `app.py` — URL + dialogue inputs, Run button, timestamp/frame/confidence metrics,
  ambiguity warning, matched transcript text, inline frame image.

![M6 — Frontend](architecture%20diagrams/M6.svg)

## Why OCR was removed

The original plan (see `docs/approach.md`) was: audio narrows the haystack, OCR verifies the
needle. Running it against the real target video exposed three things:

1. **The video has no subtitles.** "My mind rebels at stagnation" is spoken only. OCR had nothing
   to read — every frame scored ~0.38 against a channel watermark ("CHISPA MOTIVATION"), never
   crossing the 0.8 threshold.
2. **Free-tier vision APIs are operationally hostile here**: 5 requests/minute, then 20 requests/DAY
   on Gemini. A 9-frame window exhausted the daily quota mid-run; retries were useless until
   midnight Pacific.
3. **Local OCR (EasyOCR) failed on the cropped caption band** in this environment, and even if it
   worked it would only confirm what audio already knew: there is no caption text.

Defenses that were built along the way and then obsoleted: request pacing, server-honored retry
delays, daily-quota fail-fast, markdown-fence response sanitization, watermark (repeated-text)
filtering, OCR-coverage ratio with audio fallback. All removed with the OCR stage; the final
pipeline has **zero network calls after download** and cannot fail due to quotas.

The defensible framing for the interview: *"The dialogue was delivered as speech, so the extracted
text is the verified transcript segment, and the frame is time-aligned to that exact spoken
moment."* The spec's "extract the text" deliverable is satisfied by the transcript — a *stronger*
extraction than OCR-on-nothing.

## Testing

- 20 unit tests (`pytest -m "not integration"`), covering: downloader error classification,
  proxy options, idempotent caching, frame sampling/saving, transcript word-chunking, window
  matching (incl. the substring-fluke regression test), ambiguity flagging, result building,
  report formatting, and three end-to-end pipeline tests with stubbed I/O.
- 4 integration tests (real download/audio/transcription) are marked `@pytest.mark.integration`
  and deselected by default.

## Known issues

1. **ok.ru reachability**: the site is blocked/degraded on some networks.
   
2. Whisper `tiny` can mishear dialogue; set `WHISPER_MODEL_SIZE=base` in `.env` for better accuracy at 2x runtime.