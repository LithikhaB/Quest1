"""Builds the final DialogueResult from localization and frame-extraction outputs."""

from pathlib import Path

from src.frames.sampler import Frame
from src.localization.ambiguity import is_ambiguous
from src.localization.models import CandidateWindow
from src.output.models import DialogueResult


def build_dialogue_result(
    candidate_window: CandidateWindow, frame: Frame, image_path: Path
) -> DialogueResult:
    """Assemble a DialogueResult from the matched window, extracted frame, and saved image."""
    return DialogueResult(
        timestamp_seconds=frame.timestamp_seconds,
        frame_index=frame.index,
        matched_text=candidate_window.matched_text,
        image_path=image_path,
        confidence=candidate_window.confidence,
        is_ambiguous=is_ambiguous(candidate_window),
    )
