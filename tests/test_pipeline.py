"""Unit tests for end-to-end pipeline orchestration with stubbed stages.

These tests monkeypatch acquisition/transcription/OCR boundaries so the
pipeline's fallback and persistence logic can be exercised without network
access, a Gemini API key, or a real video file.
"""

import numpy as np
import pytest

import src.pipeline as pipeline_module
from src.acquisition.downloader import DownloadResult
from src.config import settings
from src.constants import DEFAULT_WINDOW_PADDING_SECONDS
from src.frames.sampler import Frame
from src.localization.models import CandidateWindow
from src.ocr.models import FrameMatch, VerificationResult


def test_locate_exact_frame_falls_back_to_audio_when_no_captions(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """With no caption text detected, the result should anchor at the audio timestamp."""
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"fake")
    fake_download = DownloadResult(
        file_path=video_file, duration_seconds=400.0, title="test video"
    )
    candidate_window = CandidateWindow(
        start_seconds=320.0,
        end_seconds=328.0,
        confidence=0.92,
        matched_text="my mind rebels at stagnation",
    )
    monkeypatch.setattr(
        pipeline_module,
        "_acquire_and_locate_window",
        lambda url, dialogue: (fake_download, candidate_window),
    )

    blank = np.zeros((10, 10, 3), dtype=np.uint8)

    def fake_sample_frames(path, start_seconds, end_seconds, interval_seconds=None):
        if start_seconds == end_seconds:
            return [Frame(index=8050, timestamp_seconds=start_seconds, image=blank)]
        return [
            Frame(index=i, timestamp_seconds=start_seconds + 0.3 * i, image=blank)
            for i in range(3)
        ]

    monkeypatch.setattr(pipeline_module, "sample_frames", fake_sample_frames)
    monkeypatch.setattr(
        pipeline_module,
        "verify_frames",
        lambda frames, dialogue: VerificationResult(
            best_match=FrameMatch(
                frame_index=0,
                timestamp_seconds=320.0,
                extracted_text="",
                confidence=0.1,
                is_confident=False,
            ),
            any_caption_text_detected=False,
        ),
    )

    saved_paths: list = []
    monkeypatch.setattr(
        pipeline_module,
        "save_frame_image",
        lambda image, file_path: saved_paths.append(file_path) or file_path,
    )

    result = pipeline_module.locate_exact_frame("https://example.com/v", "some line")

    assert result.source == "audio"
    expected_anchor = min(
        candidate_window.start_seconds + DEFAULT_WINDOW_PADDING_SECONDS,
        candidate_window.end_seconds,
    )
    assert result.timestamp_seconds == pytest.approx(expected_anchor)
    assert result.confidence == pytest.approx(0.92)
    assert result.is_confident is (0.92 >= settings.similarity_threshold)
    assert len(saved_paths) == 1


def test_locate_exact_frame_returns_confident_ocr_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A confident OCR match should be returned as-is (source 'ocr') without fallback."""
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"fake")
    fake_download = DownloadResult(file_path=video_file, duration_seconds=400.0, title="t")
    candidate_window = CandidateWindow(
        start_seconds=100.0, end_seconds=108.0, confidence=0.95, matched_text="x"
    )
    monkeypatch.setattr(
        pipeline_module,
        "_acquire_and_locate_window",
        lambda url, dialogue: (fake_download, candidate_window),
    )

    blank = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(
        pipeline_module,
        "sample_frames",
        lambda path, s, e, interval_seconds=None: [
            Frame(index=int(e * 25), timestamp_seconds=e, image=blank)
        ],
    )
    confident_match = FrameMatch(
        frame_index=2700,
        timestamp_seconds=108.0,
        extracted_text="my mind rebels at stagnation",
        confidence=0.97,
        is_confident=True,
    )
    monkeypatch.setattr(
        pipeline_module,
        "verify_frames",
        lambda frames, dialogue: VerificationResult(
            best_match=confident_match, any_caption_text_detected=True
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "save_frame_image",
        lambda image, file_path: file_path,
    )

    result = pipeline_module.locate_exact_frame("https://example.com/v", "some line")

    assert result.source == "ocr"
    assert result.is_confident is True
    assert result.frame_index == 2700
