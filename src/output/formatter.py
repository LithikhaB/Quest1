"""Output formatting for the final DialogueResult."""

from src.output.models import DialogueResult


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    total_millis = int(round(max(0.0, seconds) * 1000))
    hours, rem = divmod(total_millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def render_report(result: DialogueResult) -> str:
    """Render the final result in the spec's output format."""
    flag = "AMBIGUOUS" if result.is_ambiguous else "CONFIDENT"
    return "\n".join(
        [
            "=== Dialogue Frame Locator Result ===",
            f"Timestamp : {format_timestamp(result.timestamp_seconds)}",
            f"Frame     : {result.frame_index}",
            f'Text      : "{result.matched_text}"',
            f"Confidence: {result.confidence:.2f} [{flag}]",
            f"Image     : {result.image_path}",
        ]
    )
