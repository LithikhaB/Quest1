"""Video acquisition and download management package."""

from src.acquisition.downloader import DownloadResult, download_video
from src.acquisition.exceptions import (
    DownloadError,
    NetworkError,
    UnsupportedURLError,
    VideoUnavailableError,
)

__all__ = [
    "download_video",
    "DownloadResult",
    "DownloadError",
    "VideoUnavailableError",
    "UnsupportedURLError",
    "NetworkError",
]
