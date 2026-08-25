"""Speech-to-text transcription module using OpenAI Whisper with segment-level timestamps."""

import hashlib
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import whisper

from src.config import settings
from src.constants import DEFAULT_MAX_TRANSCRIPT_CHUNK_SPAN_SECONDS
from src.transcription.exceptions import ModelLoadError, TranscriptionFailedError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptSegment:
    """A single timestamped segment of transcribed speech.

    Attributes:
        start_seconds: Segment start time in seconds.
        end_seconds: Segment end time in seconds.
        text: Transcribed text for this segment.
    """

    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True)
class Transcript:
    """Full transcription result for an audio file.

    Attributes:
        segments: Ordered list of timestamped transcript segments.
        language: Detected or specified spoken language code.
    """

    segments: list[TranscriptSegment]
    language: str


@lru_cache(maxsize=1)
def _load_model(model_size: str) -> Any:
    """Load and cache a Whisper model instance for the process lifetime.

    Args:
        model_size: Whisper model size identifier (e.g. "tiny", "base", "small").

    Returns:
        whisper.Whisper: Loaded Whisper model instance.

    Raises:
        ModelLoadError: If the model fails to load.
    """
    try:
        logger.info("Loading Whisper model: %s", model_size)
        return whisper.load_model(model_size)
    except Exception as err:
        raise ModelLoadError(f"Failed to load Whisper model '{model_size}': {err}") from err


def _file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file's contents.

    Args:
        file_path: Path to the file to hash.

    Returns:
        str: Lowercase hex SHA-256 digest of the file contents.
    """
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chunk_words_into_segments(
    words: list[dict[str, Any]],
    max_span_seconds: float = DEFAULT_MAX_TRANSCRIPT_CHUNK_SPAN_SECONDS,
) -> list[TranscriptSegment]:
    """Group word-level timestamps into short transcript segments.

    Whisper sometimes merges many seconds of speech into one segment, which is
    too coarse for frame-precise localization. Word-level timestamps are
    regrouped so no segment spans more than `max_span_seconds`, keeping the
    transcript fine-grained without changing its schema.

    Args:
        words: Word entries with "word", "start", and "end" keys.
        max_span_seconds: Maximum time span allowed per chunk.

    Returns:
        list[TranscriptSegment]: Chunked segments in chronological order.
    """
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for word in words:
        word_start: float = float(word.get("start", 0.0))
        word_end: float = float(word.get("end", word_start))
        if current and word_end - float(current[0]["start"]) > max_span_seconds:
            chunks.append(current)
            current = []
        current.append({"start": word_start, "end": word_end, "word": str(word.get("word", ""))})
    if current:
        chunks.append(current)

    chunked: list[TranscriptSegment] = []
    for chunk in chunks:
        text: str = " ".join(str(w["word"]).strip() for w in chunk).strip()
        if not text:
            continue
        chunked.append(
            TranscriptSegment(
                start_seconds=float(chunk[0]["start"]),
                end_seconds=float(chunk[-1]["end"]),
                text=text,
            )
        )
    return chunked


def _cache_path_for(audio_path: Path, model_size: str) -> Path:
    """Resolve the JSON cache file path for a given audio file and model.

    The cache filename is derived from the audio file's SHA-256 hash plus the
    model size and a format marker, so identical audio + model combinations
    always map to the same cache file and older cache formats are ignored.

    Args:
        audio_path: Path to the source audio file.
        model_size: Whisper model size identifier.

    Returns:
        Path: Absolute path to the JSON cache file.
    """
    audio_hash: str = _file_sha256(audio_path)
    cache_dir: Path = settings.transcript_cache_dir
    return cache_dir / f"{audio_hash}_{model_size}_words.json"


def _load_cached_transcript(cache_path: Path) -> Transcript | None:
    """Load a cached Transcript from a JSON file if it exists.

    Args:
        cache_path: Path to the JSON cache file.

    Returns:
        Transcript | None: Deserialized Transcript, or None if the cache is
        missing, unreadable, or malformed.
    """
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        segments: list[TranscriptSegment] = [
            TranscriptSegment(
                start_seconds=float(seg["start_seconds"]),
                end_seconds=float(seg["end_seconds"]),
                text=str(seg["text"]),
            )
            for seg in data.get("segments", [])
        ]
        logger.info(
            "Loaded cached transcript for %s (%d segments, model=%s)",
            cache_path.stem[:16],
            len(segments),
            data.get("model_size", "unknown"),
        )
        return Transcript(segments=segments, language=str(data.get("language", "unknown")))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as err:
        logger.warning(
            "Transcript cache at %s is corrupt (%s); ignoring and re-transcribing.",
            cache_path, err,
        )
        return None


def _save_cached_transcript(cache_path: Path, transcript: Transcript, model_size: str) -> None:
    """Persist a Transcript to a JSON cache file.

    Args:
        cache_path: Path to the JSON cache file.
        transcript: Transcript to serialize.
        model_size: Whisper model size used to produce the transcript.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "model_size": model_size,
        "language": transcript.language,
        "segments": [
            {
                "start_seconds": seg.start_seconds,
                "end_seconds": seg.end_seconds,
                "text": seg.text,
            }
            for seg in transcript.segments
        ],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(
        "Cached transcript at %s (%d segments, model=%s)",
        cache_path,
        len(transcript.segments),
        model_size,
    )


def transcribe_audio(
    audio_path: Path,
    model_size: str | None = None,
) -> Transcript:
    """Transcribe an audio file into timestamped text segments.

    Results are cached to disk keyed on the audio file's SHA-256 hash plus the
    model size. On a cache hit, the cached Transcript is loaded and Whisper is
    not re-run. On a miss, the audio is transcribed and the result is written
    to the cache so subsequent runs skip the expensive model call entirely.

    Args:
        audio_path: Path to the WAV audio file to transcribe.
        model_size: Optional Whisper model size override. Defaults to settings.whisper_model_size.

    Returns:
        Transcript: Timestamped transcription result.

    Raises:
        TranscriptionFailedError: If the audio file is missing or transcription fails.
        ModelLoadError: If the underlying Whisper model cannot be loaded.
    """
    if not audio_path.exists():
        raise TranscriptionFailedError(f"Audio file not found: {audio_path}")

    resolved_model_size: str = model_size if model_size is not None else settings.whisper_model_size
    cache_path: Path = _cache_path_for(audio_path, resolved_model_size)

    cached: Transcript | None = _load_cached_transcript(cache_path)
    if cached is not None:
        return cached

    model = _load_model(resolved_model_size)

    logger.info("Transcribing audio: %s", audio_path)
    try:
        result: dict[str, Any] = model.transcribe(
            str(audio_path), verbose=False, word_timestamps=True
        )
    except Exception as err:
        raise TranscriptionFailedError(f"Whisper transcription failed for {audio_path}: {err}") from err

    segments: list[TranscriptSegment] = []
    for segment in result.get("segments", []):
        words: list[dict[str, Any]] = list(segment.get("words") or [])
        if words:
            segments.extend(_chunk_words_into_segments(words))
        else:
            segments.append(
                TranscriptSegment(
                    start_seconds=float(segment["start"]),
                    end_seconds=float(segment["end"]),
                    text=str(segment["text"]).strip(),
                )
            )

    transcript = Transcript(
        segments=segments,
        language=str(result.get("language", "unknown")),
    )

    _save_cached_transcript(cache_path, transcript, resolved_model_size)

    return transcript