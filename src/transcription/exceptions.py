"""Exception hierarchy for audio extraction and transcription failures."""


class TranscriptionError(Exception):
    """Base exception for all audio extraction and transcription errors."""

    def __init__(self, message: str) -> None:
        """Initialize the transcription error with a human-readable message.

        Args:
            message: Explanation of the error cause.
        """
        super().__init__(message)
        self.message: str = message


class AudioExtractionError(TranscriptionError):
    """Raised when audio cannot be extracted from a source video file."""


class ModelLoadError(TranscriptionError):
    """Raised when the Whisper transcription model fails to load."""


class TranscriptionFailedError(TranscriptionError):
    """Raised when speech-to-text transcription fails for a valid audio file."""
