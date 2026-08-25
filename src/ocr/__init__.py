"""OCR-based frame verification package."""

from src.ocr.exceptions import (
    MissingApiKeyError,
    NoFramesToVerifyError,
    OcrError,
    OcrRequestFailedError,
)
from src.ocr.gemini_client import extract_text_from_image
from src.ocr.models import FrameMatch
from src.ocr.verifier import verify_frames

__all__ = [
    "verify_frames",
    "extract_text_from_image",
    "FrameMatch",
    "OcrError",
    "MissingApiKeyError",
    "OcrRequestFailedError",
    "NoFramesToVerifyError",
]
