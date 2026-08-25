"""Unit tests for result assembly and output formatting."""

from pathlib import Path

import numpy as np

from src.frames.sampler import Frame
from src.localization.models import CandidateWindow
from src.output.formatter import format_timestamp, render_report
from src.output.result import build_dialogue_result


def _window(confidence: float) -> CandidateWindow:
    return CandidateWindow(
        start_seconds=1.0, end_seconds=5.0, confidence=confidence,
        matched_text="my mind rebels at stagnation",
        matched_segment_start_seconds=2.0, matched_segment_end_seconds=4.0,
    )


def _frame() -> Frame:
    return Frame(index=100, timestamp_seconds=2.0, image=np.zeros((10, 10, 3), dtype=np.uint8))


def test_format_timestamp_components() -> None:
    assert format_timestamp(325.0) == "00:05:25.000"
    assert format_timestamp(3723.456) == "01:02:03.456"
    assert format_timestamp(0.0) == "00:00:00.000"


def test_build_dialogue_result_confident() -> None:
    result = build_dialogue_result(_window(0.92), _frame(), Path("img.png"))

    assert result.timestamp_seconds == 2.0
    assert result.frame_index == 100
    assert result.matched_text == "my mind rebels at stagnation"
    assert result.image_path == Path("img.png")
    assert result.confidence == 0.92
    assert result.is_ambiguous is False


def test_build_dialogue_result_ambiguous() -> None:
    result = build_dialogue_result(_window(0.42), _frame(), Path("img.png"))
    assert result.is_ambiguous is True


def test_render_report_contains_all_fields() -> None:
    result = build_dialogue_result(_window(0.92), _frame(), Path("img.png"))
    report = render_report(result)

    assert "00:00:02.000" in report
    assert "Frame     : 100" in report
    assert '"my mind rebels at stagnation"' in report
    assert "0.92 [CONFIDENT]" in report
    assert "img.png" in report


def test_render_report_flags_ambiguity() -> None:
    result = build_dialogue_result(_window(0.42), _frame(), Path("img.png"))
    assert "AMBIGUOUS" in render_report(result)
