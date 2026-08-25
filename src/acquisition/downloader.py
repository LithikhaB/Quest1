"""Video acquisition and caching module using the yt-dlp Python API."""

import logging
import re
import shutil
import socket
import subprocess
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp
from yt_dlp.utils import (
    DownloadError as YtDlpDownloadError,
    ExtractorError,
    GeoRestrictedError,
    UnavailableVideoError,
    UnsupportedError,
)

from src.acquisition.exceptions import (
    DownloadError,
    NetworkError,
    UnsupportedURLError,
    VideoUnavailableError,
)
from src.config import settings
from src.constants import (
    DEFAULT_CONCURRENT_FRAGMENT_DOWNLOADS,
    DEFAULT_NETWORK_RETRIES,
    DEFAULT_SOCKET_TIMEOUT,
    DEFAULT_VIDEO_OUT_TEMPLATE,
    OK_RU_DEPRIORITIZED_IP,
)

logger = logging.getLogger(__name__)

_dns_patch_applied: bool = False


@dataclass
class DownloadResult:
    """Represents the outcome of a video acquisition.

    Attributes:
        file_path: Absolute or relative Path to the saved video file.
        duration_seconds: Total duration of the video in seconds.
        title: Title of the acquired video.
    """

    file_path: Path
    duration_seconds: float
    title: str


def _apply_dns_patch() -> None:
    """Deprioritize a known-unreachable ok.ru IP during DNS resolution.

    ok.ru resolves to multiple IPs. One of them (OK_RU_DEPRIORITIZED_IP) was
    observed during testing to intermittently refuse or reset connections. This
    patches socket.getaddrinfo process-wide so that IP is tried last rather than
    first, instead of failing the whole request when a working IP was available.
    Idempotent: only applied once per process, even if called repeatedly.
    """
    global _dns_patch_applied
    if _dns_patch_applied:
        return

    original_getaddrinfo = socket.getaddrinfo

    def _safe_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        results = original_getaddrinfo(host, port, *args, **kwargs)
        if host and "ok.ru" in str(host):
            return [r for r in results if r[4][0] != OK_RU_DEPRIORITIZED_IP] + [
                r for r in results if r[4][0] == OK_RU_DEPRIORITIZED_IP
            ]
        return results

    socket.getaddrinfo = _safe_getaddrinfo
    _dns_patch_applied = True


def _build_ydl_options(output_dir: Path) -> dict[str, Any]:
    """Construct configuration dictionary for YoutubeDL instances.

    Args:
        output_dir: Directory where downloaded video files will be saved.

    Returns:
        dict[str, Any]: Options dictionary for yt-dlp YoutubeDL instance.
    """
    output_template: str = str(output_dir / DEFAULT_VIDEO_OUT_TEMPLATE)
    options: dict[str, Any] = {
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": DEFAULT_NETWORK_RETRIES,
        "fragment_retries": DEFAULT_NETWORK_RETRIES,
        "concurrent_fragment_downloads": DEFAULT_CONCURRENT_FRAGMENT_DOWNLOADS,
        "socket_timeout": DEFAULT_SOCKET_TIMEOUT,
        "legacyserverconnect": True,
    }
    if settings.ytdlp_proxy:
        options["proxy"] = settings.ytdlp_proxy
        logger.info("Routing yt-dlp traffic through configured proxy.")
    return options


def _collect_exception_chain(exception: Exception) -> list[Exception]:
    """Walk the exception cause chain and return all exceptions.

    Args:
        exception: The starting exception to walk.

    Returns:
        list[Exception]: All exceptions in the chain including the root cause.
    """
    chain: list[Exception] = []
    current: BaseException | None = exception
    while current is not None:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _classify_and_raise_error(url: str, exception: Exception) -> None:
    """Classify underlying exception and raise the appropriate DownloadError subclass.

    Args:
        url: URL of the video that failed to download.
        exception: The raw exception raised during processing.

    Raises:
        UnsupportedURLError: If the URL is unsupported or invalid.
        VideoUnavailableError: If the video is private, removed, or geo-restricted.
        NetworkError: If a connection or network failure occurred.
        DownloadError: For all other unclassified acquisition failures.
    """
    chain = _collect_exception_chain(exception)
    all_messages: str = " ".join(str(e) for e in chain).lower()

    network_exception_types = (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
        ConnectionResetError,
        OSError,
        socket.error,
    )
    has_network_exception = any(isinstance(e, network_exception_types) for e in chain)
    has_network_keywords = any(
        kw in all_messages
        for kw in ("connection aborted", "connection reset", "connection refused",
                   "timed out", "timeout", "transport error", "network")
    )

    if has_network_exception or has_network_keywords:
        error_msg = f"Network connection failure while acquiring video from {url}: {exception}"
        logger.error("Download failed for %s: %s", url, error_msg)
        raise NetworkError(error_msg) from exception

    has_unavailable_keywords = any(
        kw in all_messages
        for kw in ("private", "unavailable", "removed", "not found", "sign in", "geo", "geo restricted")
    )
    has_unavailable_exception = any(isinstance(e, (UnavailableVideoError, GeoRestrictedError)) for e in chain)

    if has_unavailable_exception or has_unavailable_keywords:
        error_msg = f"The requested video is unavailable or restricted: {url}"
        logger.error("Download failed for %s: %s", url, error_msg)
        raise VideoUnavailableError(error_msg) from exception

    has_unsupported = any(isinstance(e, UnsupportedError) for e in chain)
    has_unsupported_keywords = any(
        kw in all_messages
        for kw in ("is not a valid url", "unsupported url", "no suitable extractor", "unable to extract")
    )

    if has_unsupported or has_unsupported_keywords:
        error_msg = f"The provided URL is unsupported or invalid: {url}"
        logger.error("Download failed for %s: %s", url, error_msg)
        raise UnsupportedURLError(error_msg) from exception

    error_msg = f"Failed to acquire video from {url}: {exception}"
    logger.error("Download failed for %s: %s", url, error_msg)
    raise DownloadError(error_msg) from exception


def _extract_metadata(ydl: yt_dlp.YoutubeDL, url: str) -> dict[str, Any]:
    """Extract video metadata without initiating a full media download.

    Args:
        ydl: Configured YoutubeDL instance.
        url: Web URL of the target video.

    Returns:
        dict[str, Any]: Extracted metadata dictionary from yt-dlp.

    Raises:
        DownloadError: If metadata extraction fails.
    """
    try:
        extracted = ydl.extract_info(url, download=False)
        if extracted is None:
            raise DownloadError(f"No video information could be found for {url}")
        return extracted
    except Exception as err:
        _classify_and_raise_error(url, err)
        return {}


def _find_cached_file(output_dir: Path, url: str) -> Path | None:
    """Search the output directory for an existing video file matching the URL's video ID.

    This performs no network access — it is a pure filesystem check, used to
    let a fully cached video skip metadata re-fetching entirely.

    Args:
        output_dir: Directory to search for cached video files.
        url: Source URL from which the video ID is extracted.

    Returns:
        Path to the cached file if found and non-empty, otherwise None.
    """
    video_id_match = re.search(r"/video/(\d+)", url)
    if not video_id_match:
        return None

    video_id: str = video_id_match.group(1)
    for ext in ["mp4", "mkv", "webm", "flv", "ts"]:
        candidate = output_dir / f"{video_id}.{ext}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _probe_duration_seconds(video_path: Path) -> float:
    """Read a video file's duration using ffprobe, with no network access.

    Used for the fully-cached path, where metadata is not re-fetched from the
    source URL. Failure to determine duration is non-fatal: it is logged and
    a duration of 0.0 is returned, since duration is not required for the
    rest of the pipeline to function.

    Args:
        video_path: Path to the local video file.

    Returns:
        float: Duration in seconds, or 0.0 if it could not be determined.
    """
    ffprobe_binary = shutil.which("ffprobe")
    if ffprobe_binary is None:
        logger.warning("ffprobe not found on PATH; duration for %s will default to 0.0.", video_path)
        return 0.0

    command = [
        ffprobe_binary, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0 or not result.stdout.strip():
        logger.warning("ffprobe failed to read duration for %s; defaulting to 0.0.", video_path)
        return 0.0

    try:
        return float(result.stdout.strip())
    except ValueError:
        logger.warning("ffprobe returned a non-numeric duration for %s; defaulting to 0.0.", video_path)
        return 0.0


def _resolve_target_path(ydl: yt_dlp.YoutubeDL, info: dict[str, Any], output_dir: Path) -> Path:
    """Determine the local filesystem target path for a video.

    Args:
        ydl: Configured YoutubeDL instance.
        info: Extracted metadata dictionary.
        output_dir: Destination directory.

    Returns:
        Path: Resolved filesystem path where the video file will reside.
    """
    prepared_filename: str = ydl.prepare_filename(info)
    target_path = Path(prepared_filename)
    if target_path.exists():
        return target_path

    video_id = str(info.get("id", "video"))
    for ext in ["mp4", "mkv", "webm", "flv"]:
        candidate = output_dir / f"{video_id}.{ext}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    return target_path


def download_video(url: str, output_dir: Path | None = None) -> DownloadResult:
    """Acquire a video from a URL and save it to the specified output directory.

    If the video file already exists on disk in the output directory, the
    cached file is used with no network access at all — not even to re-fetch
    metadata. Duration is read locally via ffprobe in that case, so a fully
    cached run can succeed even if the source site is temporarily unreachable.

    Args:
        url: Direct web URL of the target video.
        output_dir: Optional directory to store the downloaded video. Defaults to raw_video_dir.

    Returns:
        DownloadResult: Container holding the local file path, duration, and title.

    Raises:
        UnsupportedURLError: If the URL is unsupported or invalid.
        VideoUnavailableError: If the video is private, removed, or geo-restricted.
        NetworkError: If a connection or network failure occurred.
        DownloadError: For all other acquisition failures.
    """
    _apply_dns_patch()

    destination_dir: Path = output_dir if output_dir is not None else settings.raw_video_dir
    destination_dir.mkdir(parents=True, exist_ok=True)

    cached_path = _find_cached_file(destination_dir, url)
    if cached_path is not None:
        logger.info("Video already cached at %s; skipping network access entirely.", cached_path)
        return DownloadResult(
            file_path=cached_path,
            duration_seconds=_probe_duration_seconds(cached_path),
            title=cached_path.stem,
        )

    logger.info("Starting acquisition for video URL: %s", url)

    ydl_options: dict[str, Any] = _build_ydl_options(destination_dir)

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        info = _extract_metadata(ydl, url)
        target_path = _resolve_target_path(ydl, info, destination_dir)
        duration = float(info.get("duration") or 0.0)
        title = str(info.get("title") or target_path.stem)

        if target_path.exists() and target_path.stat().st_size > 0:
            logger.info("Video already cached at %s; skipping download.", target_path)
            return DownloadResult(
                file_path=target_path,
                duration_seconds=duration,
                title=title,
            )

        try:
            ydl.process_ie_result(info, download=True)
        except Exception as download_err:
            _classify_and_raise_error(url, download_err)

        verified_path = _resolve_target_path(ydl, info, destination_dir)
        if not (verified_path.exists() and verified_path.stat().st_size > 0):
            error_msg = f"Download finished but expected file not found at {verified_path}"
            logger.error("Download verification failed: %s", error_msg)
            raise DownloadError(error_msg)

        logger.info(
            "Successfully acquired video '%s' (duration: %.2fs) at %s",
            title,
            duration,
            verified_path,
        )
        return DownloadResult(
            file_path=verified_path,
            duration_seconds=duration,
            title=title,
        )
