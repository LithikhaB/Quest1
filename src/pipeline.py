"""End-to-end pipeline: acquire, transcribe, locate, extract the exact frame, and report."""

import logging
from pathlib import Path

from src.acquisition.downloader import DownloadResult, download_video
from src.config import settings
from src.frames.sampler import Frame, sample_frames, save_frame_image
from src.localization.locator import locate_candidate_window
from src.localization.models import CandidateWindow
from src.output.formatter import render_report
from src.output.models import DialogueResult
from src.output.result import build_dialogue_result
from src.transcription.extractor import extract_audio
from src.transcription.transcriber import transcribe_audio

logger = logging.getLogger(__name__)


def _acquire_and_locate_window(
    video_url: str, target_dialogue: str
) -> tuple[DownloadResult, CandidateWindow]:
    """Download the video, transcribe its audio, and locate the target dialogue."""
    download_result = download_video(video_url)
    logger.info("Acquired video: %s", download_result.title)

    audio_path = extract_audio(download_result.file_path)
    transcript = transcribe_audio(audio_path)
    candidate_window = locate_candidate_window(transcript, target_dialogue)

    return download_result, candidate_window


def _extract_representative_frame(video_path: Path, candidate_window: CandidateWindow) -> Frame:
    """Extract the single video frame at the exact moment the dialogue is spoken."""
    moment_seconds: float = max(
        0.0, min(candidate_window.matched_segment_start_seconds, candidate_window.matched_segment_end_seconds)
    )
    frames = sample_frames(video_path, moment_seconds, moment_seconds)
    return frames[0]


def _persist_frame_image(frame: Frame) -> Path:
    """Save the representative frame image into the frames directory."""
    file_name: str = f"match_frame_{frame.index}_{frame.timestamp_seconds:.2f}s.png"
    return save_frame_image(frame.image, settings.frames_dir / file_name)


def locate_dialogue_window(video_url: str, target_dialogue: str) -> CandidateWindow:
    """Acquire, transcribe, and locate the best-matching dialogue window."""
    _, candidate_window = _acquire_and_locate_window(video_url, target_dialogue)
    return candidate_window


def locate_exact_frame(video_url: str, target_dialogue: str) -> DialogueResult:
    """Run the full audio-driven pipeline and return the final result."""
    download_result, candidate_window = _acquire_and_locate_window(video_url, target_dialogue)
    frame = _extract_representative_frame(download_result.file_path, candidate_window)
    image_path = _persist_frame_image(frame)

    result = build_dialogue_result(candidate_window, frame, image_path)

    report = render_report(result)
    logger.info("\n%s", report)
    report_path = settings.processed_dir / "result.txt"
    try:
        report_path.write_text(report, encoding="utf-8")
    except OSError as err:
        logger.warning("Could not save result report: %s", err)

    return result


def run_pipeline(video_url: str, target_dialogue: str) -> str:
    """Run the full pipeline and return the formatted result report."""
    result = locate_exact_frame(video_url, target_dialogue)
    report = render_report(result)
    print(report)
    return report
