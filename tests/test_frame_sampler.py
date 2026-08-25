"""Integration tests for frame sampling against a real video file."""

from pathlib import Path

import pytest

from src.frames.sampler import sample_frames


@pytest.mark.integration
def test_sample_frames_from_real_video() -> None:
    """Sample a short window of frames from the M1 sample video and check basic shape/count."""
    video_path = Path("data/raw/248244667877.mp4")
    if not video_path.exists():
        pytest.skip("Sample video not found; run the M1 downloader first.")

    frames = sample_frames(video_path, start_seconds=10.0, end_seconds=12.0, interval_seconds=0.5)

    assert len(frames) >= 3
    for frame in frames:
        assert frame.image.ndim == 3
        assert frame.image.shape[2] == 3
        assert frame.timestamp_seconds >= 10.0
