"""Fuzzy dialogue localization against a timestamped transcript."""

import logging

from rapidfuzz import fuzz

from src.constants import DEFAULT_MAX_WINDOW_SEGMENTS, DEFAULT_WINDOW_PADDING_SECONDS
from src.localization.exceptions import EmptyTranscriptError
from src.localization.models import CandidateWindow
from src.transcription.transcriber import Transcript

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase and strip text for consistent fuzzy comparison.

    Args:
        text: Raw input text.

    Returns:
        str: Normalized text.
    """
    return text.strip().lower()


def locate_candidate_window(
    transcript: Transcript,
    target_dialogue: str,
    padding_seconds: float = DEFAULT_WINDOW_PADDING_SECONDS,
    max_window_segments: int = DEFAULT_MAX_WINDOW_SEGMENTS,
) -> CandidateWindow:
    """Find the transcript window most likely to contain the target dialogue.

    Slides a window of up to `max_window_segments` consecutive transcript
    segments, scores each concatenated window against the target text using
    fuzzy string matching, and returns the highest-scoring window expanded
    by `padding_seconds` on each side.

    Args:
        transcript: Timestamped transcript to search.
        target_dialogue: The dialogue text to locate.
        padding_seconds: Seconds of padding added to each side of the best window.
        max_window_segments: Maximum number of consecutive segments to merge per window.

    Returns:
        CandidateWindow: Best-matching time window with a confidence score.

    Raises:
        EmptyTranscriptError: If the transcript has no segments.
    """
    if not transcript.segments:
        raise EmptyTranscriptError("Transcript contains no segments to search.")

    normalized_target: str = _normalize(target_dialogue)
    segments = transcript.segments

    best_score: float = -1.0
    best_start: float = segments[0].start_seconds
    best_end: float = segments[0].end_seconds
    best_text: str = segments[0].text

    for window_size in range(1, max_window_segments + 1):
        for start_idx in range(len(segments) - window_size + 1):
            window = segments[start_idx : start_idx + window_size]
            window_text: str = " ".join(segment.text for segment in window)
            score: float = fuzz.partial_ratio(normalized_target, _normalize(window_text))

            if score > best_score:
                best_score = score
                best_start = window[0].start_seconds
                best_end = window[-1].end_seconds
                best_text = window_text

    logger.info(
        "Best dialogue match: score=%.1f window=[%.2f, %.2f]",
        best_score, best_start, best_end,
    )

    return CandidateWindow(
        start_seconds=max(0.0, best_start - padding_seconds),
        end_seconds=best_end + padding_seconds,
        confidence=best_score / 100.0,
        matched_text=best_text,
    )
