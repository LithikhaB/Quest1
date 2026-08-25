"""Unit tests for Whisper word-level chunking of transcript segments."""

from src.transcription.transcriber import _chunk_words_into_segments


def test_chunking_splits_long_word_spans() -> None:
    """Words spanning more than the max chunk span should be split."""
    words = [
        {"word": "one", "start": 0.0, "end": 0.5},
        {"word": "two", "start": 0.5, "end": 1.0},
        {"word": "three", "start": 1.0, "end": 1.6},
        {"word": "four", "start": 2.9, "end": 3.4},
    ]
    segments = _chunk_words_into_segments(words, max_span_seconds=2.0)

    assert len(segments) == 2
    assert segments[0].text == "one two three"
    assert segments[0].start_seconds == 0.0
    assert segments[0].end_seconds == 1.6
    assert segments[1].text == "four"
    assert segments[1].start_seconds == 2.9


def test_chunking_keeps_short_spans_together() -> None:
    """Words within the max span should remain a single segment."""
    words = [
        {"word": "to", "start": 10.2, "end": 10.4},
        {"word": "the", "start": 10.4, "end": 10.6},
        {"word": "right", "start": 10.6, "end": 11.0},
    ]
    segments = _chunk_words_into_segments(words, max_span_seconds=2.5)

    assert len(segments) == 1
    assert segments[0].text == "to the right"
    assert segments[0].start_seconds == 10.2
    assert segments[0].end_seconds == 11.0


def test_chunking_handles_empty_word_list() -> None:
    """An empty word list should produce no segments."""
    assert _chunk_words_into_segments([]) == []
