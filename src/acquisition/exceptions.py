"""Exception hierarchy for video acquisition failures."""


class DownloadError(Exception):
    """Base exception for all video download and acquisition errors."""

    def __init__(self, message: str) -> None:
        """Initialize the download error with a human-readable message.

        Args:
            message: Explanation of the download error cause.
        """
        super().__init__(message)
        self.message: str = message


class VideoUnavailableError(DownloadError):
    """Raised when the requested video is private, removed, or geo-restricted."""


class UnsupportedURLError(DownloadError):
    """Raised when the provided URL is not supported by any known extractor."""


class NetworkError(DownloadError):
    """Raised when network connectivity or socket transport fails during acquisition."""
