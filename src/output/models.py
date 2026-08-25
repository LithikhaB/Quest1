"""Data model for the final pipeline answer."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DialogueResult:
    """The final answer: where the dialogue appears and the extracted text.

    Attributes:
        timestamp_seconds: Playback position of the representative frame, in seconds.
        frame_index: Approximate frame number within the source video.
        matched_text: The transcript text matched against the target dialogue.
        image_path: Path to the saved frame image.
        confidence: Audio match confidence in [0.0, 1.0].
        is_ambiguous: True when confidence is below the configured threshold.
    """

    timestamp_seconds: float
    frame_index: int
    matched_text: str
    image_path: Path
    confidence: float
    is_ambiguous: bool
