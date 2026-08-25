# CLAUDE.md — Quest1 Project Context

## Project Overview
Quest1 is a computer-vision pipeline that identifies the exact video frame where a target dialogue first appears, extracts the text via OCR, and returns timestamp, frame number, extracted text, saved frame image, and confidence score.

**Entry point:** `app.py` (Streamlit stub — to be implemented)
**Target URL:** `https://ok.ru/video/248244667877`
**Cached video:** `data/raw/248244667877.mp4` (45 MB — already downloaded, used for testing)

## Module Progress

### M0 — Environment & Repo Skeleton ✅ COMPLETE
- Git repo initialized on `main` branch (2 commits: `397b8da`, `5bbfb31`)
- `.gitignore` excludes `.venv`, `__pycache__`, `.pytest_cache`, `.env`, `data/raw/*`, `data/frames/*`
- `requirements.txt` with dependencies: `google-genai`, `opencv-python`, `openai-whisper`, `pillow`, `python-dotenv`, `rapidfuzz`, `streamlit`, `yt-dlp`
- `README.md` — stub with project description
- `docs/prompts.txt` — started (contains the original approach prompt with 2 candidate approaches)
- `docs/approach.md` — design document with core functionalities and nice-to-haves
- `.env` file present with `GEMINI_API_KEY` set
- `app.py` — Streamlit entry point stub (to be implemented)

### M1 — Acquisition ✅ COMPLETE (Code) / ⚠️ PARTIAL (Test)
- **`src/acquisition/downloader.py`** — Full yt-dlp acquisition with:
  - `download_video(url, output_dir) -> DownloadResult` — main entry; skips re-download if cached
  - DNS patch function (`_apply_dns_patch`) — reorders ok.ru IPs to avoid unreachable `95.163.61.73`
  - yt-dlp options builder with retries, socket timeouts, legacy SSL
  - Metadata extraction via `extract_info(download=False)`
  - Target path resolution with extension fallback (`.mp4`, `.mkv`, `.webm`, `.flv`)
  - Cache check before download — returns cached file if it exists and is non-empty
  - Post-download verification
  - **`DownloadResult` dataclass**: `file_path: Path`, `duration_seconds: float`, `title: str`
- **`src/acquisition/exceptions.py`** — Exception hierarchy:
  - `DownloadError` (base)
  - `VideoUnavailableError` — private/removed/geo-restricted
  - `UnsupportedURLError` — invalid/unsupported URL
  - `NetworkError` — connection/socket failures
  - `_classify_and_raise_error()` — walks exception cause chain, classifies by type + keyword matching
- **`src/acquisition/__init__.py`** — exports `download_video`, `DownloadResult`, and all exceptions
- **`tests/test_downloader.py`** — 3 tests:
  - `test_download_video_real_ok_ru_success` — integration test (marks with `@pytest.mark.integration`)
  - `test_download_video_invalid_url_raises_download_error` — unit test (PASSES)
  - `test_download_video_idempotent_caching` — caching test (integration, may be slow)
- **Test status:** Unit test passes. Integration test fails due to **network connectivity** (`ConnectionResetError(10054)` — remote host forcibly closed connection on ok.ru). The video file already exists in `data/raw/248244667877.mp4` from a prior successful download. The code itself is correct; the failure is an environment network issue, not a code bug.
- **Key files:**
  - `src/constants.py` — default paths, similarity threshold (0.8), sampling interval (0.5s), socket timeout (30s), network retries (5)
  - `src/config.py` — `Settings` dataclass loaded from `.env`; `settings` singleton instance

### M2 — Audio Locate 🚧 NOT STARTED
- Planned: ffmpeg extract audio → faster-whisper transcript with timestamps → rapidfuzz match → candidate window + confidence
- Dependencies needed: `faster-whisper`, `ffmpeg` (system binary)

### M3 — Visual Verify 🚧 NOT STARTED
- Planned: Sample frames in window, crop to bottom-third, Gemini OCR each, fuzzy-match, pick first frame crossing threshold, save image
- Dependencies: `google-genai` (already in requirements), `opencv-python`

### M4 — Fallback Path 🚧 NOT STARTED
- Planned: If M2 confidence is low → coarse full-video OCR scan reusing M3's OCR/match code

### M5 — Output Formatting + Ambiguity Reporting 🚧 NOT STARTED
- Planned: Exact spec format, confidence flag, multi-candidate tie handling

### M6 — Documentation 🚧 NOT STARTED (parallel with M2–M5)
- Planned: Finalize `approach.md`, `prompts.txt`, `README.md`

### M7 — Interview Prep Pass 🚧 NOT STARTED (last)
- Planned: Re-read own code, rehearse answers, test with second video/line

## Architecture Notes
- Approach 2nd (URL → Video → Audio → Transcripts → tentative frame → OCR for correct frame → take snapshot) is the chosen strategy (audio-driven, dialogue strictly from audio)
- Confidence threshold: 0.8 (via `DEFAULT_SIMILARITY_THRESHOLD`)
- Sampling interval: 0.5s (via `DEFAULT_SAMPLING_INTERVAL_SECONDS`)
- Directories: `data/raw/` (videos), `data/frames/` (extracted frames), `data/processed/` (to be created for output)
- Project structure: `src/` package with sub-packages `acquisition/`, `utils/`
- Python 3.14 on Windows 11

## Known Issues & Decisions
1. `ok.ru` DNS/IP issues handled by `_apply_dns_patch()` which deprioritizes unreachable IP `95.163.61.73`
2. Integration test (`test_download_video_real_ok_ru_success`) fails with `ConnectionResetError(10054)` — network environment issue; video IS cached locally
3. `@pytest.mark.integration` is unregistered — should add `pyproject.toml` or `pytest.ini` config to register it
4. `requirements.txt` lists `openai-whisper` but the design calls for `faster-whisper` — needs updating for M2
5. `app.py` is a stub — full Streamlit implementation pending

## Environment
- Python 3.14.7 on Windows 11
- venv at `.venv/` (Python 3.14)
- GEMINI_API_KEY set in `.env`
- Dependencies install from `requirements.txt`

## Next Steps (M2)
1. Install `faster-whisper` + system `ffmpeg`
2. Build audio extraction from cached video
3. Implement transcript with timestamps
4. Rapidfuzz fuzzy-match for target dialogue → confidence + candidate window