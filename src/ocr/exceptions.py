"""Exception hierarchy for OCR extraction and frame verification failures."""


class OcrError(Exception):
    """Base exception for all OCR and frame verification errors."""

    def __init__(self, message: str) -> None:
        """Initialize the OCR error with a human-readable message.

        Args:
            message: Explanation of the error cause.
        """
        super().__init__(message)
        self.message: str = message


class MissingApiKeyError(OcrError):
    """Raised when no Gemini API key is configured."""


class OcrRequestFailedError(OcrError):
    """Raised when a Gemini OCR request fails or returns an unusable response."""


class NoFramesToVerifyError(OcrError):
    """Raised when frame verification is attempted with an empty frame list."""
