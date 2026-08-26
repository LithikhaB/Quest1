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


def test_locate_candidate_window_rejects_substring_fluke_windows() -> None:
    """A tiny segment contained in the target (e.g. 'it') must never win with score 100."""
    segments = [
        TranscriptSegment(start_seconds=0.0, end_seconds=2.0, text="it"),
        TranscriptSegment(
            start_seconds=10.0, end_seconds=14.0,
            text="why should i attempt to conceal it",
        ),
    ]
    transcript = Transcript(segments=segments, language="en")

    result = locate_candidate_window(transcript, "why should i tempt to conceal it")

    assert result.matched_segment_start_seconds == pytest.approx(10.0)
    # The correct window is selected; confidence is high but no longer exactly 1.0
    # because token_sort_ratio suppresses substring flukes rather than relying on
    # the min-word guard alone.
    assert result.confidence > 0.75


def test_locate_candidate_window_raises_on_empty_transcript() -> None:
    """An empty transcript should raise EmptyTranscriptError rather than fail silently."""
    empty_transcript = Transcript(segments=[], language="en")

    with pytest.raises(EmptyTranscriptError):
        locate_candidate_window(empty_transcript, "anything")


def test_locate_candidate_window_handles_homophone_mishearing() -> None:
    """A homophone mishearing (e.g. "reveals" instead of "rebels") should still score above threshold.

    The hybrid_score (token_sort_ratio + metaphone) ensures that a window with a
    different-but-sounding word still achieves high confidence, where plain
    partial_ratio might penalise the character-edit distance.
    """
    segments = [
        TranscriptSegment(start_seconds=0.0, end_seconds=2.0, text="my mind reveals at stagnation"),
        TranscriptSegment(start_seconds=2.0, end_seconds=5.0, text="let us begin the meeting"),
    ]
    transcript = Transcript(segments=segments, language="en")

    result = locate_candidate_window(transcript, "my mind rebels at stagnation")

    # The homophone-aware scorer finds the window; confidence is high even though
    # the text differs (rebels vs reveals).
    assert result.confidence > 0.5
    assert result.start_seconds <= 2.0
    # With padding of 1.0s, the window end is segment end + 1.0
    assert result.end_seconds >= 3.0
