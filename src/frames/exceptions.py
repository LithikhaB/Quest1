"""Exception hierarchy for video frame sampling failures."""


class FrameSamplingError(Exception):
    """Raised when frames cannot be sampled from a video file."""

    def __init__(self, message: str) -> None:
        """Initialize the frame sampling error with a human-readable message.

        Args:
            message: Explanation of the error cause.
        """
        super().__init__(message)
        self.message: str = message
