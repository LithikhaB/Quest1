"""Unit tests for OCR frame verification logic, using a stubbed OCR backend.

These tests never call the real Gemini API — extract_text_from_image is
monkeypatched so verifier control flow (early exit, best-match fallback,
empty-input handling, caption-presence detection) can be tested without
network access or an API key.
"""

import numpy as np
import pytest

import src.ocr.verifier as verifier_module
from src.frames.sampler import Frame
from src.ocr.exceptions import NoFramesToVerifyError
from src.ocr.models import VerificationResult


def _make_frame(index: int, timestamp: float) -> Frame:
    """Build a minimal synthetic frame with a blank image for verifier tests.

    Args:
        index: Frame index to assign.
        timestamp: Timestamp in seconds to assign.

    Returns:
        Frame: A frame wrapping a small blank BGR image.
    """
    blank_image = np.zeros((100, 200, 3), dtype=np.uint8)
    return Frame(index=index, timestamp_seconds=timestamp, image=blank_image)


def test_verify_frames_returns_first_confident_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first frame whose OCR text crosses the threshold should be returned immediately."""
    ocr_outputs = iter(["nothing here", "my mind rebels at stagnation", "should not be reached"])
    monkeypatch.setattr(
        verifier_module, "extract_text_from_image", lambda image: next(ocr_outputs)
    )

    frames = [_make_frame(0, 0.0), _make_frame(1, 0.5), _make_frame(2, 1.0)]
    result = verifier_module.verify_frames(
        frames, "my mind rebels at stagnation", similarity_threshold=0.9
    )

    assert isinstance(result, VerificationResult)
    assert result.best_match.frame_index == 1
    assert result.best_match.is_confident is True
    assert result.any_caption_text_detected is True


def test_verify_frames_falls_back_to_best_when_none_confident(monkeypatch: pytest.MonkeyPatch) -> None:
    """If no frame crosses the threshold, the best-scoring frame is returned as unconfident."""
    ocr_outputs = iter(["completely unrelated", "somewhat related to stagnation"])
    monkeypatch.setattr(
        verifier_module, "extract_text_from_image", lambda image: next(ocr_outputs)
    )

    frames = [_make_frame(0, 0.0), _make_frame(1, 0.5)]
    result = verifier_module.verify_frames(
        frames, "my mind rebels at stagnation", similarity_threshold=0.95
    )

    assert result.best_match.is_confident is False
    assert result.best_match.frame_index == 1
    assert result.any_caption_text_detected is True


def test_verify_frames_reports_no_captions_when_all_ocr_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every frame OCRs to empty/whitespace text, any_caption_text_detected is False.

    Note: extract_text_from_image guarantees sanitized output (markdown fences and
    empty JSON payloads are stripped inside the client), so the verifier only ever
    sees plain caption text or empty strings.
    """
    ocr_outputs = iter(["", "   ", ""])
    monkeypatch.setattr(
        verifier_module, "extract_text_from_image", lambda image: next(ocr_outputs)
    )

    frames = [_make_frame(0, 0.0), _make_frame(1, 0.3), _make_frame(2, 0.6)]
    result = verifier_module.verify_frames(frames, "some dialogue")

    assert result.any_caption_text_detected is False
    assert result.best_match.is_confident is False


def test_verify_frames_raises_on_empty_frame_list() -> None:
    """Verifying an empty frame list should raise NoFramesToVerifyError, not crash."""
    with pytest.raises(NoFramesToVerifyError):
        verifier_module.verify_frames([], "anything")
