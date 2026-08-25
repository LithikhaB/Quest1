"""Configuration management and environment settings loader."""

from dataclasses import dataclass
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.constants import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_FRAMES_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_VIDEO_DIR,
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TRANSCRIPT_CACHE_DIR,
    DEFAULT_WHISPER_MODEL_SIZE,
    DEFAULT_YTDLP_PROXY,
)

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Application configuration and runtime settings.

    Attributes:
        raw_video_dir: Directory where downloaded videos are stored.
        frames_dir: Directory where extracted frame images are saved.
        audio_dir: Directory where extracted audio files are stored.
        similarity_threshold: Minimum confidence for an unambiguous match.
        sampling_interval_seconds: Interval in seconds between sampled frames.
        whisper_model_size: Whisper model size identifier used for transcription.
        transcript_cache_dir: Directory where transcription cache is stored.
        processed_dir: Directory where final reports and outputs are stored.
        ytdlp_proxy: Optional proxy URL (http/socks5) routed through yt-dlp for
            network-restricted environments.
    """

    raw_video_dir: Path = DEFAULT_RAW_VIDEO_DIR
    frames_dir: Path = DEFAULT_FRAMES_DIR
    audio_dir: Path = DEFAULT_AUDIO_DIR
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    sampling_interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS
    whisper_model_size: str = DEFAULT_WHISPER_MODEL_SIZE
    transcript_cache_dir: Path = DEFAULT_TRANSCRIPT_CACHE_DIR
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    ytdlp_proxy: str | None = DEFAULT_YTDLP_PROXY


def load_settings(env_file_path: Path | None = None) -> Settings:
    """Load configuration from environment variables and the .env file."""
    if env_file_path is not None:
        load_dotenv(dotenv_path=env_file_path)
    else:
        load_dotenv()

    raw_video_dir: Path = Path(os.getenv("RAW_VIDEO_DIR", str(DEFAULT_RAW_VIDEO_DIR)))
    frames_dir: Path = Path(os.getenv("FRAMES_DIR", str(DEFAULT_FRAMES_DIR)))
    audio_dir: Path = Path(os.getenv("AUDIO_DIR", str(DEFAULT_AUDIO_DIR)))
    transcript_cache_dir: Path = Path(
        os.getenv("TRANSCRIPT_CACHE_DIR", str(DEFAULT_TRANSCRIPT_CACHE_DIR))
    )
    processed_dir: Path = Path(os.getenv("PROCESSED_DIR", str(DEFAULT_PROCESSED_DIR)))
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", DEFAULT_WHISPER_MODEL_SIZE)
    ytdlp_proxy: str | None = os.getenv("YTDLP_PROXY") or None

    for directory in (raw_video_dir, frames_dir, audio_dir, transcript_cache_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return Settings(
        raw_video_dir=raw_video_dir,
        frames_dir=frames_dir,
        audio_dir=audio_dir,
        similarity_threshold=float(
            os.getenv("SIMILARITY_THRESHOLD", str(DEFAULT_SIMILARITY_THRESHOLD))
        ),
        sampling_interval_seconds=float(
            os.getenv("SAMPLING_INTERVAL_SECONDS", str(DEFAULT_SAMPLING_INTERVAL_SECONDS))
        ),
        whisper_model_size=whisper_model_size,
        transcript_cache_dir=transcript_cache_dir,
        processed_dir=processed_dir,
        ytdlp_proxy=ytdlp_proxy,
    )


settings: Settings = load_settings()
