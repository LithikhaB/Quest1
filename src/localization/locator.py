"""Fuzzy dialogue localization against a timestamped transcript."""

import logging
import re

from rapidfuzz import fuzz

try:
    import jellyfish

    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False

from src.constants import (
    DEFAULT_MAX_WINDOW_SEGMENTS,
    DEFAULT_MIN_WINDOW_WORDS,
    DEFAULT_MIN_WINDOW_WORD_RATIO,
    DEFAULT_WINDOW_PADDING_SECONDS,
)
from src.localization.exceptions import EmptyTranscriptError
from src.localization.models import CandidateWindow
from src.transcription.transcriber import Transcript

logger = logging.getLogger(__name__)

_WORD_PATTERN: re.Pattern[str] = re.compile(r"[a-z0-9']+")


def _normalize(text: str) -> str:
    """Lowercase and strip text for consistent fuzzy comparison."""
    return text.strip().lower()


def _word_count(text: str) -> int:
    """Count alphanumeric words in text, ignoring punctuation."""
    return len(_WORD_PATTERN.findall(text.lower()))


def _min_window_words(target_dialogue: str) -> int:
    """Compute the minimum words a candidate window must contain to be taken seriously."""
    target_words: int = _word_count(target_dialogue)
    return max(DEFAULT_MIN_WINDOW_WORDS, int(target_words * DEFAULT_MIN_WINDOW_WORD_RATIO))


def _phonetic_score(candidate_text: str, target_text: str) -> float:
    """Compute a phonetic similarity score using metaphone encoding.

    Takes the ratio of metaphone codes across the word level, so words that sound
    alike (e.g. "rebels"/"reveals", "their"/"there"/"they're") score highly even
    when their character-edit distance is large.  Falls back to 0.0 when
    ``jellyfish`` is not installed.  Returns a score in the range 0--100.
    """
    if not _HAS_JELLYFISH:
        return 0.0

    cand_phones: str = " ".join(jellyfish.metaphone(w) for w in candidate_text.split())
    target_phones: str = " ".join(jellyfish.metaphone(w) for w in target_text.split())
    if not cand_phones or not target_phones:
        return 0.0
    return fuzz.ratio(cand_phones, target_phones)  # 0--100


def hybrid_score(candidate_text: str, target_text: str) -> float:
    """Return the better of the text‑based and phonetic similarity scores.

    * ``text_score`` — ``token_sort_ratio`` handles word reordering and suppresses
      substring flukes (a lone "it" against a long target).  Returns 0--100.
    * ``phonetic_score`` — ``metaphone``-based ratio catches ASR homophone confusion
      (``"rebels"`` mis‑heard as ``"reveals"``).  Returns 0--100.
    * Taking the ``max`` means we strictly widen what counts as a match — a real typo
      still scores well on the text side, a real mishearing scores well on the phonetic
      side, and we don't need to know in advance which kind of error we're dealing with.
    """
    text_score: float = fuzz.token_sort_ratio(
        candidate_text.lower(), target_text.lower()
    )  # 0--100
    phonetic_score: float = _phonetic_score(candidate_text, target_text)  # 0--100
    return max(text_score, phonetic_score)  # 0--100


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
    min_words: int = _min_window_words(target_dialogue)
    segments = transcript.segments

    best_score: float = -1.0
    best_start: float = segments[0].start_seconds
    best_end: float = segments[0].end_seconds
    best_text: str = segments[0].text

    for window_size in range(1, max_window_segments + 1):
        for start_idx in range(len(segments) - window_size + 1):
            window = segments[start_idx : start_idx + window_size]
            window_text: str = " ".join(segment.text for segment in window)

            # A window far shorter than the target can only match via substring
            # flukes (e.g. "it" inside a long phrase) — reject it outright.
            if _word_count(window_text) < min_words:
                continue

            score: float = hybrid_score(normalized_target, _normalize(window_text))

            if score > best_score:
                best_score = score
                best_start = window[0].start_seconds
                best_end = window[-1].end_seconds
                best_text = window_text

    if best_score < 0:
        logger.warning(
            "No transcript window met the minimum length (%d words) for target %r; "
            "returning first segment with zero confidence.",
            min_words, target_dialogue[:50],
        )
        best_score = 0.0

    logger.info(
        "Best dialogue match: score=%.1f window=[%.2f, %.2f]",
        best_score, best_start, best_end,
    )

    return CandidateWindow(
        start_seconds=max(0.0, best_start - padding_seconds),
        end_seconds=best_end + padding_seconds,
        confidence=best_score / 100.0,
        matched_text=best_text,
        matched_segment_start_seconds=best_start,
        matched_segment_end_seconds=best_end,
    )
