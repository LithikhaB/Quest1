"""Configuration management and environment settings loader."""

from dataclasses import dataclass
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.constants import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_FRAMES_DIR,
    DEFAULT_OCR_CACHE_DIR,
    DEFAULT_RAW_VIDEO_DIR,
    DEFAULT_OCR_SIMILARITY_THRESHOLD,
    DEFAULT_SAMPLING_INTERVAL_SECONDS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TRANSCRIPT_CACHE_DIR,
    DEFAULT_WHISPER_MODEL_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Application configuration and runtime settings.

    Attributes:
        gemini_api_key: API key for Google Gemini model access.
        raw_video_dir: Path to directory where downloaded videos are stored.
        frames_dir: Path to directory where extracted frames are saved.
        audio_dir: Path to directory where extracted audio files are stored.
        similarity_threshold: Minimum similarity score for dialogue matching.
        sampling_interval_seconds: Interval in seconds between sampled frames.
        whisper_model_size: Whisper model size identifier used for transcription.
        transcript_cache_dir: Path to directory where transcription cache is stored.
        ocr_cache_dir: Path to directory where OCR cache is stored.
    """

    gemini_api_key: str | None = None
    raw_video_dir: Path = DEFAULT_RAW_VIDEO_DIR
    frames_dir: Path = DEFAULT_FRAMES_DIR
    audio_dir: Path = DEFAULT_AUDIO_DIR
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ocr_similarity_threshold: float = DEFAULT_OCR_SIMILARITY_THRESHOLD
    sampling_interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS
    whisper_model_size: str = DEFAULT_WHISPER_MODEL_SIZE
    transcript_cache_dir: Path = DEFAULT_TRANSCRIPT_CACHE_DIR
    ocr_cache_dir: Path = DEFAULT_OCR_CACHE_DIR


def load_settings(env_file_path: Path | None = None) -> Settings:
    """Load configuration from environment variables and .env file.

    Args:
        env_file_path: Optional custom path to .env file.

    Returns:
        Settings: Configured settings instance with resolved values.

    Raises:
        None.
    """
    if env_file_path is not None:
        load_dotenv(dotenv_path=env_file_path)
    else:
        load_dotenv()

    api_key: str | None = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set in environment or .env file.")
        api_key = None

    raw_video_dir_str: str = os.getenv("RAW_VIDEO_DIR", str(DEFAULT_RAW_VIDEO_DIR))
    frames_dir_str: str = os.getenv("FRAMES_DIR", str(DEFAULT_FRAMES_DIR))
    audio_dir_str: str = os.getenv("AUDIO_DIR", str(DEFAULT_AUDIO_DIR))
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", DEFAULT_WHISPER_MODEL_SIZE)

    raw_video_dir: Path = Path(raw_video_dir_str)
    frames_dir: Path = Path(frames_dir_str)
    audio_dir: Path = Path(audio_dir_str)
    transcript_cache_dir_str: str = os.getenv(
        "TRANSCRIPT_CACHE_DIR", str(DEFAULT_TRANSCRIPT_CACHE_DIR)
    )
    ocr_cache_dir_str: str = os.getenv(
        "OCR_CACHE_DIR", str(DEFAULT_OCR_CACHE_DIR)
    )
    transcript_cache_dir: Path = Path(transcript_cache_dir_str)
    ocr_cache_dir: Path = Path(ocr_cache_dir_str)

    raw_video_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_cache_dir.mkdir(parents=True, exist_ok=True)
    ocr_cache_dir.mkdir(parents=True, exist_ok=True)

    similarity_threshold_val: float = float(
        os.getenv("SIMILARITY_THRESHOLD", str(DEFAULT_SIMILARITY_THRESHOLD))
    )
    ocr_similarity_threshold_val: float = float(
        os.getenv("OCR_SIMILARITY_THRESHOLD", str(DEFAULT_OCR_SIMILARITY_THRESHOLD))
    )
    sampling_interval_val: float = float(
        os.getenv("SAMPLING_INTERVAL_SECONDS", str(DEFAULT_SAMPLING_INTERVAL_SECONDS))
    )

    return Settings(
        gemini_api_key=api_key,
        raw_video_dir=raw_video_dir,
        frames_dir=frames_dir,
        audio_dir=audio_dir,
        similarity_threshold=similarity_threshold_val,
        ocr_similarity_threshold=ocr_similarity_threshold_val,
        sampling_interval_seconds=sampling_interval_val,
        whisper_model_size=whisper_model_size,
        transcript_cache_dir=transcript_cache_dir,
        ocr_cache_dir=ocr_cache_dir,
    )


settings: Settings = load_settings()
