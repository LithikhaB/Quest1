"""Audio extraction module using ffmpeg to convert video files into mono WAV audio."""

import logging
import shutil
import subprocess
from pathlib import Path

from src.config import settings
from src.constants import DEFAULT_AUDIO_SAMPLE_RATE
from src.transcription.exceptions import AudioExtractionError

logger = logging.getLogger(__name__)


def _resolve_ffmpeg_binary() -> str:
    """Locate the ffmpeg executable on the system PATH.

    Returns:
        str: Absolute path to the ffmpeg binary.

    Raises:
        AudioExtractionError: If ffmpeg is not found on the system PATH.
    """
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise AudioExtractionError("ffmpeg executable not found on system PATH.")
    return binary


def extract_audio(video_path: Path, output_dir: Path | None = None) -> Path:
    """Extract mono 16kHz WAV audio from a video file using ffmpeg.

    If the target audio file already exists, extraction is skipped and the
    cached path is returned.

    Args:
        video_path: Path to the source video file.
        output_dir: Optional directory to store the extracted audio. Defaults to settings.audio_dir.

    Returns:
        Path: Path to the extracted WAV audio file.

    Raises:
        AudioExtractionError: If the source video is missing or ffmpeg fails.
    """
    if not video_path.exists():
        raise AudioExtractionError(f"Source video file not found: {video_path}")

    destination_dir: Path = output_dir if output_dir is not None else settings.audio_dir
    destination_dir.mkdir(parents=True, exist_ok=True)

    audio_path: Path = destination_dir / f"{video_path.stem}.wav"
    if audio_path.exists() and audio_path.stat().st_size > 0:
        logger.info("Audio already extracted at %s; skipping ffmpeg.", audio_path)
        return audio_path

    ffmpeg_binary: str = _resolve_ffmpeg_binary()
    command: list[str] = [
        ffmpeg_binary,
        "-y",
        "-i", str(video_path),
        "-ac", "1",
        "-ar", str(DEFAULT_AUDIO_SAMPLE_RATE),
        "-vn",
        str(audio_path),
    ]

    logger.info("Extracting audio from %s to %s", video_path, audio_path)
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0 or not audio_path.exists():
        raise AudioExtractionError(
            f"ffmpeg failed to extract audio from {video_path}: {result.stderr.strip()}"
        )

    logger.info("Audio extraction complete: %s", audio_path)
    return audio_path
