"""End-to-end pipeline orchestration connecting acquisition, transcription, localization, and OCR verification."""

import logging
from pathlib import Path

from src.acquisition.downloader import DownloadResult, download_video
from src.config import settings
from src.constants import DEFAULT_WINDOW_PADDING_SECONDS
from src.frames.exceptions import FrameSamplingError
from src.frames.sampler import sample_frames, save_frame_image
from src.localization.locator import locate_candidate_window
from src.localization.models import CandidateWindow
from src.ocr.models import FrameMatch
from src.ocr.verifier import verify_frames
from src.transcription.extractor import extract_audio
from src.transcription.transcriber import transcribe_audio

logger = logging.getLogger(__name__)


def _acquire_and_locate_window(
    video_url: str, target_dialogue: str
) -> tuple[DownloadResult, CandidateWindow]:
    """Acquire a video and locate its best-matching dialogue window via audio transcription.

    Args:
        video_url: Source URL of the video to search.
        target_dialogue: The dialogue text to locate within the video.

    Returns:
        tuple[DownloadResult, CandidateWindow]: The acquired video's metadata and the
        localized candidate time window.

    Raises:
        DownloadError: If the video cannot be acquired.
        AudioExtractionError: If audio cannot be extracted from the video.
        ModelLoadError: If the Whisper model cannot be loaded.
        TranscriptionFailedError: If transcription fails.
        EmptyTranscriptError: If transcription produces no segments.
    """
    download_result = download_video(video_url)
    logger.info("Acquired video: %s", download_result.title)

    audio_path = extract_audio(download_result.file_path)
    transcript = transcribe_audio(audio_path)
    candidate_window = locate_candidate_window(transcript, target_dialogue)

    return download_result, candidate_window


def locate_dialogue_window(video_url: str, target_dialogue: str) -> CandidateWindow:
    """Acquire a video, transcribe its audio, and locate the best-matching dialogue window.

    Args:
        video_url: Source URL of the video to search.
        target_dialogue: The dialogue text to locate within the video.

    Returns:
        CandidateWindow: Best-matching time window with a confidence score.

    Raises:
        DownloadError: If the video cannot be acquired.
        AudioExtractionError: If audio cannot be extracted from the video.
        ModelLoadError: If the Whisper model cannot be loaded.
        TranscriptionFailedError: If transcription fails.
        EmptyTranscriptError: If transcription produces no segments.
    """
    _, candidate_window = _acquire_and_locate_window(video_url, target_dialogue)
    return candidate_window


def _build_audio_fallback_match(
    video_path: Path, candidate_window: CandidateWindow
) -> FrameMatch:
    """Anchor the result at the audio-derived timestamp when no captions exist.

    When OCR verification finds no readable caption text anywhere in the
    candidate window, the video most likely has no on-screen subtitles. The
    dialogue still "first appears" when it is spoken, so the transcript-matched
    window start (un-padded) is reported as the answer, carrying the audio
    match confidence and an empty `extracted_text`.

    Args:
        video_path: Path to the local video file.
        candidate_window: Audio-localized window containing padding on each side.

    Returns:
        FrameMatch: Match anchored where the target dialogue is spoken.

    Raises:
        FrameSamplingError: If the anchor frame cannot be read from the video.
    """
    anchor_seconds: float = min(
        candidate_window.start_seconds + DEFAULT_WINDOW_PADDING_SECONDS,
        candidate_window.end_seconds,
    )
    anchor_frames = sample_frames(video_path, anchor_seconds, anchor_seconds)
    anchor_frame = anchor_frames[0]

    logger.warning(
        "No caption text detected in the candidate window; falling back to the "
        "audio-derived timestamp %.2fs (audio confidence %.2f).",
        anchor_frame.timestamp_seconds, candidate_window.confidence,
    )

    return FrameMatch(
        frame_index=anchor_frame.index,
        timestamp_seconds=anchor_frame.timestamp_seconds,
        extracted_text="",
        confidence=candidate_window.confidence,
        is_confident=candidate_window.confidence >= settings.similarity_threshold,
        source="audio",
    )


def _persist_result_frame(video_path: Path, match: FrameMatch) -> FrameMatch:
    """Save the result frame image to disk for the final match.

    Args:
        video_path: Path to the local video file.
        match: The final match whose frame should be persisted.

    Returns:
        FrameMatch: The unchanged match; persistence failures are logged, not raised,
        so a missing image never masks a successful localization.
    """
    try:
        exact_frames = sample_frames(video_path, match.timestamp_seconds, match.timestamp_seconds)
        file_name: str = (
            f"match_frame_{match.frame_index}_{match.timestamp_seconds:.2f}s.png"
        )
        save_frame_image(exact_frames[0].image, settings.frames_dir / file_name)
    except FrameSamplingError as err:
        logger.warning("Could not save result frame image: %s", err)
    return match


def locate_exact_frame(video_url: str, target_dialogue: str) -> FrameMatch:
    """Run the full pipeline: acquire, transcribe, locate a candidate window, then verify via OCR.

    Args:
        video_url: Source URL of the video to search.
        target_dialogue: The dialogue text to locate within the video.

    Returns:
        FrameMatch: The first confident frame match, or the best available match if no
        frame crossed the configured similarity threshold.

    Raises:
        DownloadError: If the video cannot be acquired.
        AudioExtractionError: If audio cannot be extracted from the video.
        ModelLoadError: If the Whisper model cannot be loaded.
        TranscriptionFailedError: If transcription fails.
        EmptyTranscriptError: If transcription produces no segments.
        FrameSamplingError: If frames cannot be sampled from the video.
        MissingApiKeyError: If no Gemini API key is configured.
        NoFramesToVerifyError: If frame sampling yields no frames.
    """
    download_result, candidate_window = _acquire_and_locate_window(video_url, target_dialogue)

    frames = sample_frames(
        download_result.file_path,
        candidate_window.start_seconds,
        candidate_window.end_seconds,
    )
    verification = verify_frames(frames, target_dialogue)
    best_match: FrameMatch = verification.best_match

    if best_match.is_confident:
        return _persist_result_frame(download_result.file_path, best_match)

    if not verification.any_caption_text_detected:
        fallback_match = _build_audio_fallback_match(
            download_result.file_path, candidate_window
        )
        return _persist_result_frame(download_result.file_path, fallback_match)

    logger.warning(
        "Caption text was detected but no frame crossed the similarity threshold; "
        "returning ambiguous best match (confidence %.2f).",
        best_match.confidence,
    )
    return _persist_result_frame(download_result.file_path, best_match)
