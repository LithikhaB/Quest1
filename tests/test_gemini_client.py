"""Unit tests for Gemini OCR client text sanitization (no network or API key needed)."""

from src.ocr.gemini_client import _sanitize_ocr_text


def test_sanitize_strips_markdown_json_fence_to_empty() -> None:
    """A markdown-fenced empty JSON payload should sanitize to an empty string."""
    assert _sanitize_ocr_text("```json\n{}\n```") == ""


def test_sanitize_unwraps_fenced_text() -> None:
    """Text wrapped in code fences should have the fencing removed but text kept."""
    assert _sanitize_ocr_text("```text\nhello world\n```") == "hello world"


def test_sanitize_maps_empty_payloads_to_empty_string() -> None:
    """Bare empty JSON/null payloads mean 'no caption' and should become ''."""
    for token in ("{}", "[]", "null", '""'):
        assert _sanitize_ocr_text(token) == ""


def test_sanitize_keeps_plain_text_untouched() -> None:
    """Ordinary caption text must pass through unchanged."""
    assert _sanitize_ocr_text("  my mind rebels at stagnation  ") == (
        "my mind rebels at stagnation"
    )


def test_sanitize_handles_already_empty_input() -> None:
    """Empty or whitespace-only input should remain empty."""
    assert _sanitize_ocr_text("") == ""
    assert _sanitize_ocr_text("   \n  ") == ""
