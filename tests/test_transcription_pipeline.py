"""Integration tests for audio extraction and transcription against a real video file."""

from pathlib import Path

import pytest

from src.transcription.extractor import extract_audio
from src.transcription.transcriber import transcribe_audio


@pytest.mark.integration
def test_extract_and_transcribe_real_video() -> None:
    """Extract audio and transcribe the M1 sample video end-to-end.

    Uses the "tiny" Whisper model to keep runtime reasonable. Requires ffmpeg
    on PATH, an internet connection for the first Whisper model download, and
    the sample video already downloaded to data/raw/248244667877.mp4 by M1.
    """
    video_path = Path("data/raw/248244667877.mp4")
    if not video_path.exists():
        pytest.skip("Sample video not found; run the M1 downloader first.")

    audio_path = extract_audio(video_path)
    assert audio_path.exists()
    assert audio_path.stat().st_size > 0

    transcript = transcribe_audio(audio_path, model_size="tiny")
    assert len(transcript.segments) > 0
    assert all(segment.end_seconds > segment.start_seconds for segment in transcript.segments)
