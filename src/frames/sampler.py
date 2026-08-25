"""Frame sampling module for extracting frames from a video within a time window."""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.constants import DEFAULT_SAMPLING_INTERVAL_SECONDS
from src.frames.exceptions import FrameSamplingError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class Frame:
    """A single video frame sampled at a known timestamp.

    Equality comparison is disabled because the underlying image is a numpy
    array, whose element-wise comparison is not meaningful as an identity check.

    Attributes:
        index: Approximate frame number within the source video.
        timestamp_seconds: Playback position of this frame, in seconds.
        image: Frame image data as a BGR numpy array (OpenCV convention).
    """

    index: int
    timestamp_seconds: float
    image: np.ndarray


def sample_frames(
    video_path: Path,
    start_seconds: float,
    end_seconds: float,
    interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS,
) -> list[Frame]:
    """Sample frames from a video at a fixed interval within a time window.

    Frame positions are seeked by timestamp in milliseconds, which OpenCV does
    not guarantee to be frame-exact for every codec or container. The reported
    frame index is therefore an approximation derived from timestamp and frame
    rate, not a verified exact frame count.

    Args:
        video_path: Path to the source video file.
        start_seconds: Start of the sampling window, in seconds.
        end_seconds: End of the sampling window, in seconds.
        interval_seconds: Time gap between consecutive sampled frames.

    Returns:
        list[Frame]: Frames sampled in chronological order.

    Raises:
        FrameSamplingError: If the video cannot be opened or no frames are read.
    """
    if not video_path.exists():
        raise FrameSamplingError(f"Source video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FrameSamplingError(f"Failed to open video file: {video_path}")

    fps: float = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frames: list[Frame] = []

    try:
        timestamp: float = start_seconds
        while timestamp <= end_seconds:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, image = capture.read()
            if success:
                approx_index: int = int(round(timestamp * fps)) if fps > 0 else len(frames)
                frames.append(Frame(index=approx_index, timestamp_seconds=timestamp, image=image))
            else:
                logger.warning("Failed to read frame at %.2fs; skipping.", timestamp)
            timestamp += interval_seconds
    finally:
        capture.release()

    if not frames:
        raise FrameSamplingError(
            f"No frames could be read from {video_path} between {start_seconds}s and {end_seconds}s."
        )

    logger.info("Sampled %d frames from %.2fs to %.2fs", len(frames), start_seconds, end_seconds)
    return frames


def save_frame_image(image: np.ndarray, file_path: Path) -> Path:
    """Write a single frame image to disk as a PNG file.

    Args:
        image: Frame image as a BGR numpy array (OpenCV convention).
        file_path: Destination path for the PNG file.

    Returns:
        Path: The path the image was written to.

    Raises:
        FrameSamplingError: If the image cannot be encoded or written.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(file_path), image):
        raise FrameSamplingError(f"Failed to write frame image to {file_path}")
    logger.info("Saved frame image to %s", file_path)
    return file_path
