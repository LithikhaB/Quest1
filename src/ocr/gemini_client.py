"""Gemini vision OCR client for extracting on-screen text from video frames."""

import hashlib
import logging
import re
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import errors, types
import time as _time
from src.config import settings
from src.constants import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_PACING_BUFFER_SECONDS,
    DEFAULT_GEMINI_REQUESTS_PER_MINUTE,
    DEFAULT_OCR_BASE_RETRY_DELAY_SECONDS,
    DEFAULT_OCR_MAX_ATTEMPTS,
)
from src.ocr.exceptions import MissingApiKeyError, OcrRequestFailedError

logger = logging.getLogger(__name__)

_OCR_PROMPT: str = (
    "Extract only the on-screen caption or subtitle text visible in this image. "
    "Respond with the exact text as plain text only — no markdown, no code fences, no JSON. "
    "If there is no readable on-screen text, respond with an empty string."
)

_MARKDOWN_FENCE_PATTERN: re.Pattern[str] = re.compile(
    r"^```[\w-]*\s*\n?(.*?)\n?\s*```$", re.DOTALL
)
_EMPTY_PAYLOAD_TOKENS: frozenset[str] = frozenset({"{}", "[]", "null", '""', "''"})


def _sanitize_ocr_text(raw_text: str) -> str:
    """Normalize a raw Gemini OCR response into plain caption text.

    Vision models sometimes wrap their answer in markdown code fences (e.g.
    ```` ```json {} ``` ````) even when asked for plain text. This strips such
    fencing and maps empty JSON payloads to an empty string so downstream
    fuzzy matching never sees formatting artifacts.

    Args:
        raw_text: Raw response text from the model.

    Returns:
        str: Sanitized caption text; empty string when no caption was reported.
    """
    text: str = raw_text.strip()
    fence_match = _MARKDOWN_FENCE_PATTERN.match(text)
    if fence_match is not None:
        text = fence_match.group(1).strip()
    if text in _EMPTY_PAYLOAD_TOKENS:
        return ""
    return text

_RATE_LIMIT_STATUS_CODE: int = 429
_RETRY_DELAY_PATTERN: re.Pattern[str] = re.compile(r"[Rr]etry\D{0,30}?(\d+(?:\.\d+)?)\s*s")
_MIN_REQUEST_INTERVAL_SECONDS: float = (
    60.0 / DEFAULT_GEMINI_REQUESTS_PER_MINUTE + DEFAULT_GEMINI_PACING_BUFFER_SECONDS
)
_last_request_monotonic: float | None = None
_daily_quota_exhausted: bool = False


def _build_client() -> genai.Client:
    """Construct a Gemini API client using the configured API key.

    Returns:
        genai.Client: Configured Gemini client instance.

    Raises:
        MissingApiKeyError: If no Gemini API key is configured.
    """
    if not settings.gemini_api_key:
        raise MissingApiKeyError("GEMINI_API_KEY is not configured; cannot perform OCR.")
    return genai.Client(api_key=settings.gemini_api_key)


def _image_cache_path(model: str, image: np.ndarray) -> Path:
    """Resolve the on-disk cache file path for a given image and OCR model.

    The cache key is the SHA-256 hash of the encoded PNG image bytes plus the
    model name, so identical cropped images against the same model map to the
    same cache file.

    Args:
        model: Gemini model identifier used for OCR.
        image: Frame image as a BGR numpy array.

    Returns:
        Path: Absolute path to the cache file.
    """
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OcrRequestFailedError("Failed to encode frame image as PNG for caching.")

    image_bytes: bytes = encoded.tobytes()
    image_hash: str = hashlib.sha256(image_bytes).hexdigest()
    cache_dir: Path = settings.ocr_cache_dir
    return cache_dir / f"{image_hash}_{model}.txt"


def _load_cached_ocr(cache_path: Path) -> str | None:
    """Load a cached OCR result from disk if it exists.

    Args:
        cache_path: Path to the cached OCR text file.

    Returns:
        str | None: Cached OCR text, or None if no cache file exists.
    """
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as err:
        logger.warning("Failed to read OCR cache at %s: %s", cache_path, err)
        return None


def _save_cached_ocr(cache_path: Path, text: str) -> None:
    """Persist OCR text to a cache file.

    Args:
        cache_path: Path where the OCR text will be written.
        text: OCR result to persist.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)


def _pace_requests() -> None:
    """Sleep as needed so consecutive Gemini requests stay under the per-minute quota.

    Free-tier Gemini models allow only a handful of requests per minute, and a
    frame-by-frame OCR loop can easily exceed that. Enforcing a minimum interval
    between outgoing requests prevents 429 RESOURCE_EXHAUSTED errors from
    occurring in the first place.
    """
    global _last_request_monotonic
    if _last_request_monotonic is not None:
        elapsed: float = _time.monotonic() - _last_request_monotonic
        remaining: float = _MIN_REQUEST_INTERVAL_SECONDS - elapsed
        if remaining > 0:
            logger.info(
                "Pacing Gemini requests: waiting %.1fs to stay under %d requests/minute.",
                remaining, DEFAULT_GEMINI_REQUESTS_PER_MINUTE,
            )
            _time.sleep(remaining)
    _last_request_monotonic = _time.monotonic()


def _extract_retry_delay_seconds(err: Exception) -> float | None:
    """Parse the server-suggested retry delay (in seconds) from a rate-limit error.

    Gemini 429 responses embed a hint such as ``Please retry in 36s`` or a
    ``RetryInfo.retryDelay`` field. Honoring it avoids burning retries inside
    the same exhausted quota window.

    Args:
        err: The raised exception from the Gemini API call.

    Returns:
        float | None: Suggested delay in seconds, or None if not parseable.
    """
    match = _RETRY_DELAY_PATTERN.search(str(err))
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_text_from_image(image: np.ndarray, model: str = DEFAULT_GEMINI_MODEL) -> str:
    """Extract on-screen text from a single image using Gemini vision OCR.

    Results are cached to disk keyed on the SHA-256 of the encoded PNG image
    bytes plus the model name. On a cache hit, the cached text is returned
    without contacting the Gemini API. On a miss, the request is made and the
    result is written to the cache so subsequent runs of the same image are
    instant.

    Retries a limited number of times on request failure (e.g. transient rate
    limiting) before giving up, since a single dropped call should not abort
    an entire OCR verification run.

    Args:
        image: Frame image as a BGR numpy array (OpenCV convention).
        model: Gemini model identifier to use for OCR.

    Returns:
        str: Extracted text, stripped of surrounding whitespace. Empty string if none found.

    Raises:
        MissingApiKeyError: If no Gemini API key is configured.
        OcrRequestFailedError: If the OCR request still fails after all retry attempts.
    """
    cache_path: Path = _image_cache_path(model, image)
    cached_text: str | None = _load_cached_ocr(cache_path)
    if cached_text is not None:
        logger.info("OCR cache hit for %s (model=%s)", cache_path.stem[:16], model)
        return _sanitize_ocr_text(cached_text)

    client = _build_client()

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OcrRequestFailedError("Failed to encode frame image as PNG for OCR.")

    max_attempts = DEFAULT_OCR_MAX_ATTEMPTS
    base_retry_delay_seconds = DEFAULT_OCR_BASE_RETRY_DELAY_SECONDS
    response = None
    global _daily_quota_exhausted

    if _daily_quota_exhausted:
        raise OcrRequestFailedError(
            "Gemini daily request quota is exhausted for this API key; "
            "OCR is unavailable until the daily quota resets."
        )

    for attempt in range(1, max_attempts + 1):
        _pace_requests()
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/png"),
                    _OCR_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            break
        except errors.APIError as err:
            is_rate_limited: bool = getattr(err, "code", None) == _RATE_LIMIT_STATUS_CODE
            is_daily_quota_exhausted: bool = "perday" in str(err).lower()
            if is_daily_quota_exhausted:
                _daily_quota_exhausted = True
                logger.error(
                    "Gemini daily request quota exhausted; skipping all remaining OCR requests."
                )
            if attempt == max_attempts or is_daily_quota_exhausted:
                raise OcrRequestFailedError(
                    f"Gemini OCR request failed after {attempt} attempts: {err}"
                ) from err
            if is_rate_limited:
                server_delay = _extract_retry_delay_seconds(err)
                retry_delay: float = (
                    server_delay + DEFAULT_GEMINI_PACING_BUFFER_SECONDS
                    if server_delay is not None
                    else base_retry_delay_seconds * (2 ** (attempt - 1))
                )
            else:
                retry_delay = base_retry_delay_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Gemini OCR request failed (attempt %d/%d): %s; retrying in %.1fs.",
                attempt, max_attempts, err, retry_delay,
            )
            _time.sleep(retry_delay)
        except Exception as err:
            if attempt == max_attempts:
                raise OcrRequestFailedError(
                    f"Gemini OCR request failed after {max_attempts} attempts: {err}"
                ) from err
            retry_delay = base_retry_delay_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Gemini OCR request failed (attempt %d/%d): %s; retrying in %.1fs.",
                attempt, max_attempts, err, retry_delay,
            )
            _time.sleep(retry_delay)

    text: str | None = getattr(response, "text", None)
    extracted: str = _sanitize_ocr_text(text) if text is not None else ""

    _save_cached_ocr(cache_path, extracted)
    return extracted