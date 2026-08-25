"""Data models for dialogue localization results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateWindow:
    """A candidate time window where the target dialogue is spoken.

    Attributes:
        start_seconds: Padded window start, in seconds.
        end_seconds: Padded window end, in seconds.
        confidence: Fuzzy match confidence score in the range [0.0, 1.0].
        matched_text: The transcript text that produced the match.
        matched_segment_start_seconds: Un-padded start of the matched transcript span.
        matched_segment_end_seconds: Un-padded end of the matched transcript span.
    """

    start_seconds: float
    end_seconds: float
    confidence: float
    matched_text: str
    matched_segment_start_seconds: float
    matched_segment_end_seconds: float
