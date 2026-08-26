# All Explored Approaches

**Problem:** given a video URL and a line of dialogue, find the exact video frame where
that dialogue appears, extract the text, and report timestamp + frame + confidence.

---

## Approach 1 — Naive (Failed): Sliding frame windows + OCR every frame

The first instinct: dialogue "appears" on screen, so look at the pixels.

```
Video URL → download → sample frames every 0.3s around the likely region
          → send each frame's caption band to a vision OCR (Gemini)
          → fuzzy-match OCR text vs target → first frame over threshold wins
```

An audio stage (Whisper) narrows *where* to look, then every frame in that window is OCR'd and
matched against the target text frame-by-frame.

**Pros**
- Conceptually simple: "read the screen, find the sentence."
- Works regardless of whether the dialogue is spoken at all (pure burned-in captions).
- Frame-precision comes directly from pixels — no dependence on ASR timing accuracy.

**Cons (all discovered the hard way — see `ERRORS_FACED.txt`)**
- **Assumes captions exist.** Our target video has none — the line is spoken only. OCR had
  nothing to read; every frame scored ~0.38 against a channel watermark.
- **Per-frame API cost.** A 10-second window at 0.3s sampling = ~30 vision-API calls per query.
  Free tiers allow 5 requests/minute and 20/day — the quota died mid-run, twice.
- **Watermark false positives.** Persistent overlay text ("CHISPA MOTIVATION" video) matched every
  frame, poisoning both the confidence signal and the "captions exist" detection.
- **Fragile plumbing.** Markdown-fenced responses, version-specific SDK classes, poisoned caches.
- **Slow.** Rate-limit pacing made each frame cost ~15s; a full window took minutes.

**Verdict:** built completely, then removed. It only ever adds information when captions exist —
and our video is the case where they don't.

---

## Approach 2 — Implemented: Audio-first localization with word-level timestamps

### Check out [log.md](docs/log.md) for detailed Architecture-wise Implementation


The reframe: if the dialogue is **spoken**, the frame where it "first appears" is the frame at
the moment it is **said** — and a word-level transcript tells you exactly when that is.

```
Video URL → yt-dlp download (cache-first, parallel fragments)
          → ffmpeg → mono 16 kHz WAV
          → Whisper (local, word_timestamps=True)
          → words re-chunked into ≤2.5s segments
          → rapidfuzz sliding-window match vs target
          → CandidateWindow + confidence
          → OpenCV seek to matched segment start → save frame PNG
          → confidence < threshold ⇒ flag [AMBIGUOUS]
```

Two details carry the precision:

1. **Word-level chunking.** Whisper merges ~10s of speech into one segment; segment-level
   matching gave 13-second windows. Re-chunking words to ≤2.5s spans shrank match windows to
   ~2 seconds — that is what makes "exact frame" defensible.
2. **Guarded fuzzy matching.** `rapidfuzz.partial_ratio` absorbs typos ("tempt" → spoken
   "attempt", matched at 0.94), but it also gives a fake 100 to any tiny chunk contained in the
   target ("it" ⊂ "…conceal it"). A minimum-word guard (`max(3, 50% of target words)`) rejects
   those before scoring.

**Pros**
- **No API keys, no quotas, no cost.** Everything runs locally; the only network call is the
  one-time download.
- **Fast where it matters.** Cold run on a 50-min video ≈ 10 min; every rerun ≈ 3 seconds
  (video, audio, and transcript all cached).
- **Robust to query noise.** Typos, casing, punctuation, and ASR mishearings are absorbed by
  character-level fuzzy matching over short, isolated windows.
- **Honest failure mode.** Weak matches are flagged `[AMBIGUOUS]`, never silently trusted.
- **Verified.** 50-min ok.ru video: "my mind rebels at stagnation" → 05:25.3, frame 7799,
  confidence 1.00; typo'd query → 13:00.7, frame 18718, confidence 0.94.

**Cons**
- **Assumes the dialogue is spoken(as per test URL)** — a video with captions but no matching speech would need
  the OCR stage re-added (kept as a documented "caption-aware mode").
- **ASR is the single point of failure.** Heavy accents, overlapping speech, or music over the
  line can corrupt the transcript; mitigation is fuzzy matching + the ambiguity flag, plus a
  one-line model upgrade (`WHISPER_MODEL_SIZE=base`).
- **Timing accuracy is ASR-bound.** Whisper timestamps are ~50–100ms accurate — at 25 fps that
  is 1–3 frames of uncertainty. Pixels would be exact; audio is approximate.
- **Transcription is O(video length).** One-time and cached, but a cold run on long videos is
  minutes, not seconds.

---

## Approach 3 — Future scope: Embedding-based spoken term detection (no ASR at all)

If you have (or can synthesize) an **audio snippet of the target phrase**, you can skip text
entirely: embed both the query audio and sliding windows of the video's audio into vectors using
a self-supervised speech model (wav2vec2, HuBERT, or a dedicated QbE model), then rank windows
by cosine similarity or align them with DTW (dynamic time warping). This is "query-by-example
spoken term detection" from the keyword-spotting research literature — true audio-to-audio
matching.

![Approach 3](architecture%20diagrams/Approach3.svg)

**Pros**
- **Zero text conversion** — accents, homophones, spelling, and ASR noise become irrelevant by
  construction: you match *sounds*, not transcriptions.
- **Language-agnostic.** The same pipeline works for any spoken query in any language the
  encoder covers.
- **No transcript cache to maintain** — matching works directly on audio.

**Cons**
- **Needs the phrase as audio.** Our input is a *text* line; we would have to TTS it first, and
  the matcher must then survive the TTS voice differing from the actor's voice (the core
  challenge QbE research works on).
- **Compute-heavy.** Embedding a 50-minute video window-by-window is far more expensive than one
  Whisper `tiny` pass, and DTW alignment is quadratic in window length.
- **Less explainable.** "0.87 cosine similarity" is harder to defend than "the transcript says
  the words at 05:25.3" — debugging a bad match means listening to audio, not reading text.
- **Coarser boundaries.** Embedding similarity peaks near the phrase but gives less precise
  word-level onset timing than word timestamps.

---

## Approach 4 — Future scope: Keyword-spotting pre-filter + targeted ASR

A "search, then verify" cascade: instead of transcribing all 50 minutes with an accurate model,
run a **cheap detector** over the whole audio that flags regions where target keywords *might*
occur, then run **expensive, accurate ASR only on those regions**.

Example — target: *"The company reported revenue of twenty million dollars."*

Extract the informative terms: `company`, `reported`, `revenue`, `twenty million`, `dollars`.

![Approach 4](architecture%20diagrams/Approach%204.svg)

```
50-minute audio, keyword detector scores over time:

  0 ──────────████──────────────────────────████──────────
              8:20                           32:40
              candidate region               candidate region

Run accurate ASR ONLY on: 8:10–8:40   and   32:30–33:00
```

Fifty minutes of audio shrink to ~1 minute of expensive transcription.

**Pros**
- **Massive ASR savings on long videos.** The expensive model runs on ~2% of the audio; the
  cheap detector (a small streaming keyword model, or even `tiny` Whisper itself) scans
  everything in real time.
- **Best of both worlds.** Final matches still come from accurate word-level ASR, so precision
  and explainability match Approach 2.
- **Naturally parallel.** Candidate regions are independent — detect once, verify concurrently.

**Cons**
- **Recall risk.** If the cheap detector misses a keyword (accent, noise, overlap), the correct
  region is never transcribed and the line is unrecoverable — a false negative you never see.
- **Keyword choice is a heuristic.** Which words are "informative"? Common words ("it", "the")
  fire everywhere; rare words may be mispronounced or misheard by the detector itself.
- **Two-stage complexity.** More moving parts: detector thresholds, region padding, merging
  nearby candidates — each a new source of bugs (our substring-fluke incident was exactly such a
  threshold bug, in miniature).
- **Overkill for our input sizes.** The shipped pipeline already transcribes a 50-min video in
  ~90s with `tiny` and caches it forever; the cascade pays off only at hours-scale or with
  `large` models.

---

## Comparison

| | A1: OCR frames | **A2: Audio-first (shipped)** | A3: Embeddings (QbE) | A4: KWS + targeted ASR |
|---|---|---|---|---|
| Needs captions? | Yes | No | No | No |
| Needs API keys / quotas | Yes (was fatal) | **No** | No (but GPU-friendly) | No |
| Handles typos in query | Fuzzy OCR match | **Fuzzy transcript match** | N/A (audio query) | Fuzzy transcript match |
| Handles accents | N/A (no speech needed) | Via ASR quality | **Best (matches sounds)** | Via ASR quality |
| Timing precision | Frame-exact (pixels) | Word-timestamp (~1–3 frames) | Coarse | Word-timestamp |
| Cold cost, 50-min video | Minutes of API calls + $$ | ~10 min local, one-time | Highest (GPU compute) | ~2 min local |
| Explainability | Medium | **High (read the transcript)** | Low (vector similarity) | High |
| Failure mode | Silent watermark matches | Honest `[AMBIGUOUS]` flag | Silent low-similarity | Silent missed keyword |

