# Extract Frame from Video URL

## Core Functionalities

- Download video from URL using `yt-dlp`
  - Supports the given `ok.ru` URL
  - Handle extractor/download failures with a clear error path

- Extract audio and transcribe locally using **Whisper**: timestamp + transcript

- Search transcript for the target dialogue

- Sample frames within the candidate window

- Match OCR output against target dialogue: find first occurrence

- Produce required output:
  - Timestamp
  - Frame number
  - Extracted dialogue text
  - Saved frame image
  - Confidence score

- Handle uncertainty
  - Define a confidence/similarity threshold
  - Mark results as ambiguous when confidence is below the threshold

- Fallback when audio search fails
  - Perform a coarse OCR scan across the video


## Nice-to-Have — If Time Remains

- Crop OCR to the likely caption/subtitle region
  - Reduce Gemini calls
  - Improve processing speed

- Scene-cut detection
  - Skip near-duplicate frames
  - Reduce unnecessary OCR calls

- Binary-search refinement
  - Once a `no-text → text` transition is found
  - Narrow down the exact first frame with fewer Gemini calls

- Independent Gemini verification
  - Use a second Gemini pass to verify the candidate frame
  - Keep Gemini as a verifier rather than the sole decision-maker

- Dockerfile: Containerize the application

## Initial Architecture with tentative tech stack

![Initial Architecture](architecture%20diagram/version1.svg)

---

## Modules

- **M0 — Environment & Repository Skeleton**
  - Public GitHub repository: `Quest1`
  - Set up folder structure
  - Add `requirements.txt`
  - Create `argparse` CLI skeleton
  - Add initial `README.md`
  - Start `prompts.txt`

- **M1 — Acquisition**
  - Download video using `yt-dlp`
  - Add clean error handling
  - Test against the actual `ok.ru` URL first
  - Complete and validate this module before building further stages
![M1](architecture%20diagram/M1.svg)

- **M2 — Audio Locate**
  - Extract audio using `ffmpeg`
  - Transcribe locally using `faster-whisper`
  - Generate timestamped transcript segments
  - Match target dialogue using `rapidfuzz`
  - Produce candidate time window + confidence
![M2](architecture%20diagram/M2.svg)

- **M3 — Visual Verify**
  - Sample frames within the candidate window
  - Crop to the likely subtitle/caption region
  - Send selected frames to Gemini for OCR
  - Fuzzy-match OCR text against target dialogue
  - Select the first frame crossing the confidence threshold
  - Save the matched frame
![M3](architecture%20diagram/M3.svg)

- **M4 — Fallback Path**
  - Trigger when M2 produces low-confidence/no match
  - Perform a coarse full-video OCR scan
  - Reuse M3's OCR and matching logic
  - Locate an approximate region for further verification

- **M5 — Output & Ambiguity Handling**
  - Format output according to the assignment specification
  - Include timestamp, frame number, extracted text, confidence, and image
  - Report ambiguous/low-confidence results
  - Handle multiple close candidate matches

- **M6 — Documentation**
  - Finalize `APPROACH.md`
  - Finalize `prompts.txt`
  - Update `README.md`
  - Document design decisions, assumptions, thresholds, and limitations
  - Documentation is maintained in parallel with implementation, not only at the end

- **M7 — Interview Preparation & Validation**
  - Review the complete implementation
  - Rehearse the four key areas of the solution
  - Test with a second video/dialogue pair
  - Verify that the pipeline is not hardcoded to the sample input
  - Prepare to explain design decisions, trade-offs, and failure handling