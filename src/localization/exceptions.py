"""Exception hierarchy for dialogue localization failures."""


class LocalizationError(Exception):
    """Base exception for all dialogue-to-transcript localization errors."""

    def __init__(self, message: str) -> None:
        """Initialize the localization error with a human-readable message.

        Args:
            message: Explanation of the error cause.
        """
        super().__init__(message)
        self.message: str = message


class EmptyTranscriptError(LocalizationError):
    """Raised when the transcript contains no segments to search."""
