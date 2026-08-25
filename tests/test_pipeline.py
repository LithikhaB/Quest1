"""Unit tests for the audio-only end-to-end pipeline with stubbed stages."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

import src.pipeline as pipeline_module
from src.acquisition.downloader import DownloadResult
from src.frames.sampler import Frame
from src.localization.models import CandidateWindow
from src.output.models import DialogueResult


def _window(confidence: float) -> CandidateWindow:
    return CandidateWindow(
        start_seconds=1.0, end_seconds=5.0, confidence=confidence,
        matched_text="my mind rebels at stagnation",
        matched_segment_start_seconds=2.0, matched_segment_end_seconds=4.0,
    )


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, confidence: float) -> list:
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"fake")
    fake_download = DownloadResult(file_path=video_file, duration_seconds=10.0, title="test")
    monkeypatch.setattr(
        pipeline_module,
        "_acquire_and_locate_window",
        lambda url, dialogue: (fake_download, _window(confidence)),
    )

    sampled: list = []
    blank = np.zeros((10, 10, 3), dtype=np.uint8)

    def fake_sample(path, start_seconds, end_seconds, interval_seconds=None):
        frame = Frame(index=50, timestamp_seconds=start_seconds, image=blank)
        sampled.append(frame)
        return [frame]

    monkeypatch.setattr(pipeline_module, "sample_frames", fake_sample)
    monkeypatch.setattr(
        pipeline_module, "save_frame_image", lambda image, file_path: file_path
    )
    monkeypatch.setattr(
        pipeline_module,
        "settings",
        SimpleNamespace(processed_dir=tmp_path, frames_dir=tmp_path),
    )
    return sampled


def test_locate_exact_frame_returns_dialogue_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The pipeline should anchor the frame at the matched segment start."""
    _stub_pipeline(monkeypatch, tmp_path, confidence=0.92)

    result = pipeline_module.locate_exact_frame("https://example.com/v", "some line")

    assert isinstance(result, DialogueResult)
    assert result.timestamp_seconds == pytest.approx(2.0)
    assert result.frame_index == 50
    assert result.matched_text == "my mind rebels at stagnation"
    assert result.confidence == pytest.approx(0.92)
    assert result.is_ambiguous is False


def test_locate_exact_frame_flags_ambiguity_on_low_confidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Low audio confidence must mark the result ambiguous."""
    _stub_pipeline(monkeypatch, tmp_path, confidence=0.42)

    result = pipeline_module.locate_exact_frame("https://example.com/v", "some line")

    assert result.is_ambiguous is True


def test_run_pipeline_prints_report(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
    """run_pipeline should print a report containing the timestamp and frame."""
    _stub_pipeline(monkeypatch, tmp_path, confidence=0.92)

    report = pipeline_module.run_pipeline("https://example.com/v", "some line")

    assert "00:00:02.000" in report
    assert "Frame     : 50" in report
