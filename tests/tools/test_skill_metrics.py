"""Tests for tools/skill_metrics.py — SkillMetrics sidecar tracking."""

import json
import pytest
from pathlib import Path
from tools.skill_metrics import SkillMetrics


@pytest.fixture()
def metrics(tmp_path):
    return SkillMetrics(path=tmp_path / ".metrics.json")


class TestSkillMetrics:
    def test_record_view(self, metrics):
        metrics.record_view("my-skill")
        assert metrics._data["my-skill"]["views"] == 1
        metrics.record_view("my-skill")
        assert metrics._data["my-skill"]["views"] == 2

    def test_ema_update(self, metrics):
        metrics.record_view("s1")
        metrics._data["s1"]["views"] = 5
        for rate in [1.0, 1.0, 0.0, 1.0, 1.0]:
            metrics.record_session_outcome(["s1"], rate)
        score = metrics._data["s1"]["score"]
        assert 0.5 < score < 0.9

    def test_label_untested(self, metrics):
        metrics.record_view("new-skill")
        assert metrics.get_label("new-skill") == "untested"

    def test_label_reliable(self, metrics):
        metrics._data["good"] = {"views": 10, "score": 0.85}
        assert metrics.get_label("good") == "reliable"

    def test_label_unreliable(self, metrics):
        metrics._data["bad"] = {"views": 8, "score": 0.25}
        assert metrics.get_label("bad") == "unreliable"

    def test_label_mixed(self, metrics):
        metrics._data["mid"] = {"views": 6, "score": 0.55}
        assert metrics.get_label("mid") == "mixed"

    def test_get_flagged(self, metrics):
        metrics._data = {
            "good": {"views": 10, "score": 0.80},
            "bad": {"views": 6, "score": 0.20},
            "new": {"views": 2, "score": 0.10},
        }
        flagged = metrics.get_flagged()
        assert flagged == ["bad"]

    def test_save_load_roundtrip(self, metrics):
        metrics.record_view("test")
        metrics.save()
        m2 = SkillMetrics(path=metrics._path)
        m2.load()
        assert m2._data == metrics._data

    def test_corrupt_file_recovery(self, metrics):
        metrics._path.write_text("not json")
        metrics.load()
        assert metrics._data == {}

    def test_missing_file_ok(self, metrics):
        metrics.load()
        assert metrics._data == {}

    def test_get_metrics_returns_label(self, metrics):
        metrics._data["s1"] = {"views": 10, "score": 0.80, "last_used": "2026-03-12"}
        m = metrics.get_metrics(["s1"])
        assert m["s1"]["label"] == "reliable"
        assert m["s1"]["views"] == 10

    def test_record_session_skips_unknown_skills(self, metrics):
        metrics.record_session_outcome(["nonexistent"], 1.0)
        assert "nonexistent" not in metrics._data
