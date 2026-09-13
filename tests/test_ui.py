"""Smoke tests for the Streamlit demo (the `ui` dependency group)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


def _app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    assert not at.exception
    return at


def test_app_loads_without_exceptions():
    _app()


def test_skew_tab_runs():
    at = _app()
    button = next(b for b in at.button if b.label == "Run skew detection")
    button.click().run(timeout=15)
    assert not at.exception


def test_release_gate_tab_runs():
    at = _app()
    button = next(b for b in at.button if b.label == "Evaluate release gate")
    button.click().run(timeout=15)
    assert not at.exception
    # default sliders (gini=0.42 above min 0.30, brier=0.18 below max 0.25, reviewed_by
    # left blank) should refuse specifically on the missing review, not silently pass.
    assert any("REFUSED" in e.value for e in at.error)
    assert any("reviewed" in m.value.lower() for m in at.markdown)


def test_feature_store_write_button_runs():
    at = _app()
    button = next(b for b in at.button if b.label == "Write")
    button.click().run(timeout=15)
    assert not at.exception
