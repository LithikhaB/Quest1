"""Frame-by-frame OCR verification of dialogue against sampled video frames."""

import logging
from collections import Counter
import numpy as np
from rapidfuzz import fuzz

from src.config import settings
from src.constants import DEFAULT_CAPTION_CROP_RATIO, DEFAULT_MIN_CAPTION_TEXT_LENGTH
from src.frames.sampler import Frame
from src.ocr.exceptions import NoFramesToVerifyError, OcrRequestFailedError
from src.ocr.gemini_client import extract_text_from_image
from src.ocr.models import FrameMatch, VerificationResult

logger = logging.getLogger(__name__)

_WATERMARK_MIN_FRAMES: int = 3
_WATERMARK_MAJORITY_FRACTION: float = 0.5


def _crop_to_caption_band(image: np.ndarray, crop_ratio: float = DEFAULT_CAPTION_CROP_RATIO) -> np.ndarray:
    """Crop an image to its bottom band, where on-screen captions typically appear.

    Cropping before sending to OCR reduces irrelevant visual content in the
    request and cuts down on false positives from non-caption on-screen text.

    Args:
        image: Frame image as a BGR numpy array.
        crop_ratio: Fraction of the frame height to keep, measured from the bottom.

    Returns:
        numpy.ndarray: Cropped image containing only the bottom band.
    """
    height: int = image.shape[0]
    crop_start: int = int(height * (1 - crop_ratio))
    return image[crop_start:height, :, :]


def _normalize(text: str) -> str:
    """Lowercase and strip text for consistent fuzzy comparison.

    Args:
        text: Raw input text.

    Returns:
        str: Normalized text.
    """
    return text.strip().lower()


def _detect_watermark_text(frame_texts: list[tuple[Frame, str]]) -> str | None:
    """Identify persistent overlay text (e.g. channel watermarks) across frames.

    Text that repeats identically across many frames is almost certainly a
    watermark or logo, not the dialogue caption being searched for. Leaving it
    in would pollute similarity scores (every frame gets the same weak score)
    and falsely signal that captions exist.

    Args:
        frame_texts: (frame, OCR text) pairs collected from the sampled window.

    Returns:
        str | None: The watermark text if detected, otherwise None.
    """
    non_empty: list[str] = [
        text.strip() for _, text in frame_texts
        if len(text.strip()) >= DEFAULT_MIN_CAPTION_TEXT_LENGTH
    ]
    if not non_empty:
        return None

    top_text, top_count = Counter(non_empty).most_common(1)[0]
    if top_count >= _WATERMARK_MIN_FRAMES and top_count >= _WATERMARK_MAJORITY_FRACTION * len(non_empty):
        return top_text
    return None


def verify_frames(
    frames: list[Frame],
    target_dialogue: str,
    similarity_threshold: float | None = None,
) -> VerificationResult:
    """Run OCR on sampled frames in order and find the first confident dialogue match.

    Frames are checked in chronological order. As soon as a frame's OCR text
    crosses the similarity threshold against the target dialogue, that frame
    is returned immediately without OCR-ing the remaining frames. If no frame
    crosses the threshold, the highest-scoring frame seen is returned with
    `is_confident=False` so the caller can flag the result as ambiguous rather
    than silently trusting a weak match.

    The result also reports whether any frame produced readable caption text.
    When it did not, the video window likely has no on-screen captions at all,
    and the caller should fall back to audio-derived timing instead of treating
    the low-confidence visual match as meaningful.

    Args:
        frames: Chronologically ordered frames to verify, typically from `sample_frames`.
        target_dialogue: The dialogue text to match against extracted OCR text.
        similarity_threshold: Minimum confidence to accept a match immediately.
            Defaults to settings.ocr_similarity_threshold.

    Returns:
        VerificationResult: The best FrameMatch plus a flag indicating whether
        any caption text was detected across all verified frames.

    Raises:
        NoFramesToVerifyError: If `frames` is empty.
    """
    if not frames:
        raise NoFramesToVerifyError("No frames were provided for OCR verification.")

    threshold: float = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.ocr_similarity_threshold
    )
    normalized_target: str = _normalize(target_dialogue)

    best_match: FrameMatch | None = None
    any_caption_text_detected: bool = False

    for frame in frames:
        cropped_image = _crop_to_caption_band(frame.image)
        try:
            extracted_text = extract_text_from_image(cropped_image)
        except OcrRequestFailedError as err:
            logger.warning(
                "OCR unavailable for frame %d (%.2fs); treating as no text. Cause: %s",
                frame.index, frame.timestamp_seconds, err,
            )
            extracted_text = ""
        if len(extracted_text.strip()) >= DEFAULT_MIN_CAPTION_TEXT_LENGTH:
            any_caption_text_detected = True
        score: float = fuzz.partial_ratio(normalized_target, _normalize(extracted_text)) / 100.0

        logger.info(
            "Frame %d @ %.2fs OCR='%s' score=%.2f",
            frame.index, frame.timestamp_seconds, extracted_text, score,
        )

        candidate = FrameMatch(
            frame_index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            extracted_text=extracted_text,
            confidence=score,
            is_confident=score >= threshold,
        )

        if best_match is None or candidate.confidence > best_match.confidence:
            best_match = candidate

        if candidate.is_confident:
            logger.info(
                "Confident match found at frame %d (%.2fs); stopping early.",
                frame.index, frame.timestamp_seconds,
            )
            return VerificationResult(best_match=candidate, any_caption_text_detected=True)

    logger.warning(
        "No frame crossed the similarity threshold (%.2f); returning best available match with confidence %.2f.",
        threshold, best_match.confidence,
    )
    return VerificationResult(
        best_match=best_match, any_caption_text_detected=any_caption_text_detected
    )
