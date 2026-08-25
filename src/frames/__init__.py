"""Video frame sampling package."""

from src.frames.exceptions import FrameSamplingError
from src.frames.sampler import Frame, sample_frames

__all__ = ["sample_frames", "Frame", "FrameSamplingError"]
