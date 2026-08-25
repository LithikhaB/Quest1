"""Unit tests for dialogue localization fuzzy matching logic."""

import pytest

from src.localization.exceptions import EmptyTranscriptError
from src.localization.locator import locate_candidate_window
from src.transcription.transcriber import Transcript, TranscriptSegment


def _build_transcript() -> Transcript:
    """Build a small synthetic transcript fixture for locator tests."""
    segments = [
        TranscriptSegment(start_seconds=0.0, end_seconds=2.0, text="Good morning everyone"),
        TranscriptSegment(start_seconds=2.0, end_seconds=5.0, text="my mind rebels at stagnation"),
        TranscriptSegment(start_seconds=5.0, end_seconds=7.0, text="let us begin the meeting"),
    ]
    return Transcript(segments=segments, language="en")


def test_locate_candidate_window_finds_matching_segment() -> None:
    """An exact-text match should produce a high-confidence window around the right segment."""
    transcript = _build_transcript()
    result = locate_candidate_window(transcript, "my mind rebels at stagnation")

    assert result.confidence > 0.9
    assert result.start_seconds <= 2.0
    assert result.end_seconds >= 5.0
    assert result.matched_segment_start_seconds == pytest.approx(2.0)
    assert result.matched_segment_end_seconds == pytest.approx(5.0)


def test_locate_candidate_window_handles_partial_noise() -> None:
    """A partial, reworded version of the line should still score reasonably high."""
    transcript = _build_transcript()
    result = locate_candidate_window(transcript, "mind rebels stagnation")

    assert result.confidence > 0.7


def test_locate_candidate_window_raises_on_empty_transcript() -> None:
    """An empty transcript should raise EmptyTranscriptError rather than fail silently."""
    empty_transcript = Transcript(segments=[], language="en")

    with pytest.raises(EmptyTranscriptError):
        locate_candidate_window(empty_transcript, "anything")
