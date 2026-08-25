# Quest1 — Dialogue Frame Locator

Given a video URL and a line of dialogue, this pipeline finds the exact video frame where that
dialogue is spoken, saves that frame as an image, and reports the timestamp, frame number,
extracted text (the verified transcript segment), confidence, and an ambiguity flag.

## How it works

1. **Acquire (M1)** — cache-first download via `yt-dlp` (concurrent fragment downloads, retry
   hardening, ok.ru DNS edge avoidance).
2. **Localize (M2)** — `ffmpeg` extracts audio -> local OpenAI Whisper (`tiny`) transcribes with
   **word-level timestamps** (long segments chunked to <=2.5s) -> `rapidfuzz` sliding-window match
   produces a candidate window + confidence.
3. **Extract (M3)** — the single representative frame at the matched segment start is pulled via
   OpenCV seek and saved to `data/frames/` as a PNG.
4. **Flag (M4)** — the result is marked ambiguous when the audio match confidence is below the
   configured threshold (default 0.8).
5. **Report (M5)** — a `DialogueResult` (timestamp, frame number, matched text, image path,
   confidence, ambiguous flag) is printed in the spec format and saved to
   `data/processed/result.txt`.

## Run

```powershell
python -X utf8 -c "from src.pipeline import run_pipeline; run_pipeline('https://ok.ru/video/248244667877', 'my mind rebels at stagnation')"

streamlit run app.py
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Requires `ffmpeg` and `ffprobe` on PATH. Configure thresholds/directories via `.env`
(see `src/constants.py` for defaults).

### Network note (ok.ru blocked on some ISPs)

ok.ru is blocked/degraded on some networks (TLS connections reset mid-handshake). This is an
environment issue, not a code issue. Fixes, in order of ease:

1. **Phone hotspot** — connect and rerun.
2. **Cloudflare WARP** (free, 1.1.1.1 app) or **ProtonVPN free** — system-wide VPN, then rerun.
3. **Local proxy** — if you run a proxy-style VPN (Psiphon, WARP proxy mode, etc.), set it in
   `.env` and the downloader routes through it automatically:
   ```
   YTDLP_PROXY = socks5://127.0.0.1:1080
   ```

Once downloaded, the video is cached in `data/raw/` and no network is needed again.

## Project structure

```
app.py                  Streamlit frontend
src/
  acquisition/          yt-dlp download + error classification
  transcription/        ffmpeg audio extraction, Whisper transcription (word-chunked, cached)
  localization/         rapidfuzz window matching + ambiguity flag
  frames/               OpenCV frame sampling and image saving
  output/               DialogueResult model, report formatting
  pipeline.py           End-to-end orchestration
tests/                  pytest unit suite (integration tests marked separately)
```

## Docs

- `docs/approach.md` — design document (current architecture, modules, trade-offs)
- `prompts.txt` — AI prompts used
