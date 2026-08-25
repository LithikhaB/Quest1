"""Data models for OCR-based frame verification results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameMatch:
    """Result of matching OCR-extracted text from a single frame against target dialogue.

    Attributes:
        frame_index: Approximate frame number within the source video.
        timestamp_seconds: Playback position of the matched frame, in seconds.
        extracted_text: Text extracted from the frame via OCR (empty if sourced from audio).
        confidence: Match score in [0.0, 1.0] — OCR-vs-target similarity, or audio match
            confidence when `source` is "audio".
        is_confident: Whether confidence met or exceeded the configured similarity threshold.
        source: Origin of this match: "ocr" (visual verification) or "audio"
            (fallback to the transcript-derived timestamp when no captions exist).
    """

    frame_index: int
    timestamp_seconds: float
    extracted_text: str
    confidence: float
    is_confident: bool
    source: str = "ocr"


@dataclass(frozen=True)
class VerificationResult:
    """Aggregate outcome of OCR verification over a candidate window.

    Attributes:
        best_match: Highest-scoring FrameMatch seen across all verified frames.
        any_caption_text_detected: Whether any frame yielded readable caption text.
            False indicates the window likely has no on-screen captions at all,
            which lets callers fall back to audio-derived timing.
    """

    best_match: FrameMatch
    any_caption_text_detected: bool
