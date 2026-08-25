"""Data models for dialogue localization results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateWindow:
    """A candidate time window where the target dialogue likely appears.

    Attributes:
        start_seconds: Start of the candidate window, in seconds.
        end_seconds: End of the candidate window, in seconds.
        confidence: Fuzzy match confidence score in the range [0.0, 1.0].
        matched_text: The transcript text that produced the match.
    """

    start_seconds: float
    end_seconds: float
    confidence: float
    matched_text: str
