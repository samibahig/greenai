"""Tests for the simple GreenAI class."""

import time
import pytest

from greenai import GreenAI
from greenai.backends.simple import SimpleBackend


class TestGreenAIBasic:
    def test_start_stop_report(self):
        g = GreenAI()
        g.start()
        time.sleep(0.02)
        g.stop()
        data = g.report()
        assert data["duration_seconds"] > 0
        assert data["energy_kwh"] > 0
        assert data["co2_kg"] > 0
        assert data["backend"] == "simple"

    def test_chaining(self):
        g = GreenAI()
        result = g.start()
        assert result is g
        result = g.stop()
        assert result is g

    def test_is_running(self):
        g = GreenAI()
        assert not g.is_running
        g.start()
        assert g.is_running
        g.stop()
        assert not g.is_running

    def test_double_start_raises(self):
        g = GreenAI()
        g.start()
        with pytest.raises(RuntimeError):
            g.start()
        g.stop()

    def test_stop_before_start_raises(self):
        g = GreenAI()
        with pytest.raises(RuntimeError):
            g.stop()

    def test_report_before_stop_raises(self):
        g = GreenAI()
        with pytest.raises(RuntimeError):
            g.report()

    def test_properties(self):
        g = GreenAI()
        g.start()
        time.sleep(0.02)
        g.stop()
        assert g.duration > 0
        assert g.energy_wh > 0
        assert g.co2_g > 0


class TestGreenAIContextManager:
    def test_basic_context_manager(self):
        with GreenAI() as g:
            time.sleep(0.02)
        data = g.report()
        assert data["duration_seconds"] > 0

    def test_measure_context_manager(self):
        with GreenAI.measure() as g:
            time.sleep(0.02)
        data = g.report()
        assert data["duration_seconds"] > 0


class TestGreenAIDecorator:
    def test_decorator_no_args(self):
        @GreenAI.track
        def dummy():
            time.sleep(0.02)
        dummy()  # should not raise

    def test_decorator_with_args(self):
        @GreenAI.track(print_report=False)
        def dummy():
            time.sleep(0.02)
        dummy()  # should not raise

    def test_decorator_preserves_return_value(self):
        @GreenAI.track(print_report=False)
        def compute():
            return 42
        assert compute() == 42


class TestGreenAIOutput:
    def test_report_returns_dict(self):
        g = GreenAI()
        g.start()
        g.stop()
        result = g.report()
        assert isinstance(result, dict)
        assert set(result.keys()) >= {"duration_seconds", "energy_kwh", "co2_kg", "backend"}

    def test_repr_not_started(self):
        g = GreenAI()
        assert "not started" in repr(g)

    def test_repr_after_stop(self):
        g = GreenAI()
        g.start()
        g.stop()
        r = repr(g)
        assert "GreenAI(" in r
        assert "duration" in r

    def test_custom_backend(self):
        g = GreenAI(backend=SimpleBackend(average_wattage=500))
        g.start()
        time.sleep(0.02)
        g.stop()
        data = g.report()
        assert data["backend"] == "simple"
