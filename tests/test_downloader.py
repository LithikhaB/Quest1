"""Unit and integration tests for the video downloader module."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import src.acquisition.downloader as downloader_module
from src.acquisition.downloader import DownloadResult, _build_ydl_options, download_video
from src.acquisition.exceptions import DownloadError


def test_build_ydl_options_includes_proxy_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """The proxy setting must be forwarded to yt-dlp options when present."""
    monkeypatch.setattr(
        downloader_module, "settings", SimpleNamespace(ytdlp_proxy="socks5://127.0.0.1:1080")
    )
    options = _build_ydl_options(Path("data/raw"))
    assert options["proxy"] == "socks5://127.0.0.1:1080"


def test_build_ydl_options_omits_proxy_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """No proxy key should be present when no proxy is configured."""
    monkeypatch.setattr(downloader_module, "settings", SimpleNamespace(ytdlp_proxy=None))
    options = _build_ydl_options(Path("data/raw"))
    assert "proxy" not in options


@pytest.mark.integration
def test_download_video_real_ok_ru_success(tmp_path: Path) -> None:
    """Integration test verifying video download from ok.ru.

    Hits the network to download a real video, confirming file creation
    and valid metadata extraction.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If output validation fails.
    """
    test_url: str = "https://ok.ru/video/248244667877"
    result: DownloadResult = download_video(url=test_url, output_dir=tmp_path)

    assert isinstance(result, DownloadResult)
    assert result.file_path.exists()
    assert result.file_path.stat().st_size > 0
    assert result.duration_seconds > 0
    assert len(result.title) > 0


def test_download_video_invalid_url_raises_download_error(tmp_path: Path) -> None:
    """Test that an invalid or unresolvable URL raises a DownloadError.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If DownloadError is not raised.
    """
    invalid_url: str = "https://not-a-real-site.invalid/video/123"

    with pytest.raises(DownloadError):
        download_video(url=invalid_url, output_dir=tmp_path)


@pytest.mark.integration
def test_download_video_idempotent_caching(tmp_path: Path) -> None:
    """Test that downloading an already acquired video returns cached result.

    Marked as integration: the first call in a fresh tmp_path directory has
    nothing to find locally, so it always requires a real network download
    before the second call can exercise the cache-hit path.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If cached result verification fails.
    """
    test_url: str = "https://ok.ru/video/248244667877"
    first_result: DownloadResult = download_video(url=test_url, output_dir=tmp_path)
    initial_mtime: float = first_result.file_path.stat().st_mtime

    second_result: DownloadResult = download_video(url=test_url, output_dir=tmp_path)
    second_mtime: float = second_result.file_path.stat().st_mtime

    assert first_result.file_path == second_result.file_path
    assert initial_mtime == second_mtime
    assert second_result.duration_seconds == first_result.duration_seconds
