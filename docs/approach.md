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