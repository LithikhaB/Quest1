"""Ambiguity flagging based purely on transcript-match confidence."""

from src.config import settings
from src.localization.models import CandidateWindow


def is_ambiguous(candidate_window: CandidateWindow) -> bool:
    """Return True when the audio match confidence is below the similarity threshold."""
    return candidate_window.confidence < settings.similarity_threshold
