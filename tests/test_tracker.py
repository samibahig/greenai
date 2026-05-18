"""Tests for the GreenAI Tracker."""

import time
import pytest

from greenai import Tracker, Report
from greenai.backends.simple import SimpleBackend


class TestReport:
    def test_energy_wh_conversion(self):
        r = Report(duration_seconds=10, energy_kwh=0.001, co2_kg=0.0002, backend="simple")
        assert r.energy_wh == pytest.approx(1.0)

    def test_co2_g_conversion(self):
        r = Report(duration_seconds=10, energy_kwh=0.001, co2_kg=0.0002, backend="simple")
        assert r.co2_g == pytest.approx(0.2)

    def test_summary_contains_backend(self):
        r = Report(duration_seconds=5, energy_kwh=0.0001, co2_kg=0.00002, backend="simple")
        assert "simple" in r.summary()

    def test_to_dict_keys(self):
        r = Report(duration_seconds=5, energy_kwh=0.0001, co2_kg=0.00002, backend="simple")
        d = r.to_dict()
        assert "duration_seconds" in d
        assert "energy_kwh" in d
        assert "co2_kg" in d
        assert "backend" in d


class TestSimpleBackend:
    def test_energy_is_positive_after_stop(self):
        b = SimpleBackend()
        b.start()
        time.sleep(0.05)
        b.stop()
        assert b.energy_kwh > 0

    def test_stop_before_start_raises(self):
        b = SimpleBackend()
        with pytest.raises(RuntimeError):
            b.stop()

    def test_name(self):
        assert SimpleBackend().name == "simple"

    def test_metadata_has_wattage(self):
        b = SimpleBackend(average_wattage=200)
        b.start()
        b.stop()
        assert b.metadata["assumed_wattage_w"] == 200


class TestTracker:
    def test_basic_start_stop(self):
        t = Tracker()
        t.start()
        time.sleep(0.05)
        report = t.stop()
        assert isinstance(report, Report)
        assert report.duration_seconds > 0
        assert report.energy_kwh > 0
        assert report.co2_kg > 0

    def test_report_property(self):
        t = Tracker()
        assert t.report is None
        t.start()
        t.stop()
        assert t.report is not None

    def test_double_start_raises(self):
        t = Tracker()
        t.start()
        with pytest.raises(RuntimeError):
            t.start()
        t.stop()

    def test_stop_without_start_raises(self):
        t = Tracker()
        with pytest.raises(RuntimeError):
            t.stop()

    def test_is_running(self):
        t = Tracker()
        assert not t.is_running
        t.start()
        assert t.is_running
        t.stop()
        assert not t.is_running

    def test_context_manager(self):
        with Tracker() as t:
            time.sleep(0.02)
        assert t.report is not None
        assert t.report.duration_seconds > 0

    def test_decorator(self):
        @Tracker.track(print_report=False)
        def dummy():
            time.sleep(0.02)

        dummy()

    def test_measure_context_manager(self):
        with Tracker.measure() as t:
            time.sleep(0.02)
        assert t.report is not None

    def test_custom_backend(self):
        backend = SimpleBackend(average_wattage=300)
        t = Tracker(backend=backend)
        t.start()
        time.sleep(0.02)
        report = t.stop()
        assert report.backend == "simple"

    def test_timestamps_set(self):
        t = Tracker()
        t.start()
        report = t.stop()
        assert report.started_at is not None
        assert report.stopped_at is not None
        assert report.stopped_at >= report.started_at


class TestRecommender:
    def test_budget_exceeded_gives_high_priority(self):
        from greenai import Recommender
        r = Report(duration_seconds=10, energy_kwh=0.001, co2_kg=0.1, backend="simple")
        rec = Recommender(carbon_budget_kg=0.05)
        suggestions = rec.analyze(r)
        priorities = [s.priority for s in suggestions]
        assert "high" in priorities

    def test_long_run_triggers_fp16_recommendation(self):
        from greenai import Recommender
        r = Report(duration_seconds=7200, energy_kwh=0.3, co2_kg=0.07, backend="simple")
        rec = Recommender()
        suggestions = rec.analyze(r)
        titles = [s.title for s in suggestions]
        assert any("FP16" in t or "mixed-precision" in t for t in titles)

    def test_large_model_triggers_quantization(self):
        from greenai import Recommender
        r = Report(duration_seconds=100, energy_kwh=0.05, co2_kg=0.01, backend="simple")
        rec = Recommender(model_size_params=500_000_000)
        suggestions = rec.analyze(r)
        titles = [s.title for s in suggestions]
        assert any("quantization" in t.lower() for t in titles)

    def test_always_has_scheduling_suggestion(self):
        from greenai import Recommender
        r = Report(duration_seconds=1, energy_kwh=0.00001, co2_kg=0.000001, backend="simple")
        rec = Recommender()
        suggestions = rec.analyze(r)
        assert len(suggestions) >= 1
