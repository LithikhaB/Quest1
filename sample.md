# sample.md — Build Log & Engineering Summary

Detailed summary of how this project was built, what changed, and why.

## Final architecture (audio-driven, no OCR)

```
Video URL
  -> M1 Acquisition      src/acquisition/downloader.py   (yt-dlp, cache-first)
  -> M2 Localization     src/transcription/ + src/localization/
                          ffmpeg -> Whisper(word timestamps) -> rapidfuzz window
  -> M3 Frame extraction src/frames/sampler.py            (seek to matched moment, save PNG)
  -> M4 Ambiguity        src/localization/ambiguity.py    (confidence < threshold)
  -> M5 Output           src/output/                      (DialogueResult + report)
  -> M6 UI               app.py                           (Streamlit)
```

## What was built, in order

### M1 — Acquisition (kept, hardened)
- `download_video(url)` with yt-dlp Python API: metadata pass, cache check before any network,
  post-download verification, exception classification into
  `NetworkError` / `VideoUnavailableError` / `UnsupportedURLError`.
- `concurrent_fragment_downloads=8` for fast HLS downloads (ok.ru serves fragmented HLS).
- DNS patch deprioritizes a known-dead ok.ru edge IP (`95.163.61.73`).

### M2 — Audio localization (kept, upgraded)
- `extract_audio` (ffmpeg -> 16kHz mono WAV, cached).
- `transcribe_audio` — OpenAI Whisper `tiny`, **`word_timestamps=True`**.
- `_chunk_words_into_segments` — regroups words so no transcript segment spans >2.5s.
  This was the single biggest precision win: Whisper merged ~10s of speech into one segment,
  making candidate windows ~13s wide; after chunking, windows are ~2s.
- `locate_candidate_window` — rapidfuzz sliding window (1 segment + 1s padding),
  returns `CandidateWindow(start, end, confidence, matched_text, matched_segment_start/end)`.
- Transcript cache keyed on audio SHA-256 + model + `_words` format marker.

### M3 — Frame extraction (redefined: primary path, not verification)
- `_extract_representative_frame` — `sample_frames(video, t, t)` seeks to the matched segment
  start and reads that single frame.
- `save_frame_image` — `cv2.imwrite` into `data/frames/match_frame_<n>_<t>s.png`.

### M4 — Ambiguity
- `is_ambiguous(window)` — `window.confidence < settings.similarity_threshold` (default 0.8).

### M5 — Output
- `DialogueResult(timestamp_seconds, frame_index, matched_text, image_path, confidence, is_ambiguous)`.
- `render_report` — spec format with `HH:MM:SS.mmm` timestamps; saved to `data/processed/result.txt`.

### M6 — Streamlit
- `app.py` — URL + dialogue inputs, Run button, timestamp/frame/confidence metrics,
  ambiguity warning, matched transcript text, inline frame image.

## Why OCR was removed (the honest engineering story)

The original plan (see `APPROACH.md`) was: audio narrows the haystack, OCR verifies the needle.
Running it against the real target video exposed three things:

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

- 21 unit tests (`pytest -m "not integration"`), covering: downloader error classification and
  idempotent caching, frame sampling/saving, transcript word-chunking, window matching,
  ambiguity flagging, result building, report formatting, and three end-to-end pipeline tests
  with stubbed I/O.
- 4 integration tests (real download/audio/transcription) are marked `@pytest.mark.integration`
  and deselected by default.

## Known issues

1. **ok.ru reachability**: the site is blocked/degraded on some networks (TLS handshake killed —
   `SSL: UNEXPECTED_EOF_WHILE_READING` / `ConnectionResetError 10054`). This is ISP-level SNI
   filtering, not a code bug — no application code can bypass it. Workarounds: phone hotspot,
   Cloudflare WARP (free), ProtonVPN free, or a local proxy via the `YTDLP_PROXY` setting in
   `.env` (the downloader forwards it to yt-dlp). The cache-first downloader means this only
   matters once per video.
2. Whisper `tiny` can mishear dialogue; set `WHISPER_MODEL_SIZE=base` in `.env` for better
   accuracy at ~2x runtime.

## Submission checklist (per flowchart)

- [x] Public GitHub repo with all source code
- [x] `README.md` — how to run
- [x] `APPROACH.md` — original design (kept)
- [x] `docs/approach_audio_only.md` — final design + rationale
- [x] `prompts.txt` — AI prompts (kept, untouched)
- [x] `sample.md` — this build log
- [x] `requirements.txt` — dependencies
