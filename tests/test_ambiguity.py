"""Unit tests for ambiguity flagging."""

from src.localization.ambiguity import is_ambiguous
from src.localization.models import CandidateWindow


def _window(confidence: float) -> CandidateWindow:
    return CandidateWindow(
        start_seconds=0.0, end_seconds=4.0, confidence=confidence,
        matched_text="x", matched_segment_start_seconds=1.0, matched_segment_end_seconds=3.0,
    )


def test_high_confidence_is_not_ambiguous() -> None:
    assert is_ambiguous(_window(0.92)) is False


def test_low_confidence_is_ambiguous() -> None:
    assert is_ambiguous(_window(0.42)) is True
