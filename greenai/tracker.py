"""
Tracker — the main entry point for GreenAI.

Usage:
    from greenai import Tracker

    tracker = Tracker()           # uses SimpleBackend by default
    tracker.start()
    train_model(...)
    report = tracker.stop()
    print(report)                 # human-readable summary
    print(report.to_dict())       # machine-readable dict
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable, Generator, Optional, Type

from greenai.backends.base import BaseBackend
from greenai.backends.simple import SimpleBackend
from greenai.report import Report

# Carbon intensity in g CO₂ / kWh — used to convert energy → CO₂
# Source: IEA 2023 global average (override via Tracker(co2_intensity=...))
_DEFAULT_CO2_INTENSITY_G_PER_KWH = 233.0


class Tracker:
    """
    Tracks the carbon footprint of any computation.

    Args:
        backend: A BaseBackend instance.  Defaults to SimpleBackend.
        co2_intensity_g_per_kwh: Grid carbon intensity (g CO₂/kWh).
            Ignored when the backend reports its own emissions directly
            (e.g. CodeCarbonBackend).

    Examples:
        # Basic usage
        tracker = Tracker()
        tracker.start()
        result = heavy_computation()
        report = tracker.stop()

        # Context manager
        with Tracker() as tracker:
            result = heavy_computation()
        print(tracker.report)

        # Decorator
        @Tracker.track
        def heavy_computation():
            ...

        # Custom backend
        from greenai.backends import CodeCarbonBackend
        tracker = Tracker(backend=CodeCarbonBackend(country_iso_code="FRA"))
    """

    def __init__(
        self,
        backend: Optional[BaseBackend] = None,
        co2_intensity_g_per_kwh: float = _DEFAULT_CO2_INTENSITY_G_PER_KWH,
    ) -> None:
        self._backend: BaseBackend = backend or SimpleBackend()
        self._co2_intensity = co2_intensity_g_per_kwh
        self._started_at: Optional[datetime] = None
        self._report: Optional[Report] = None
        self._running = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def start(self) -> "Tracker":
        """Begin tracking. Returns self for chaining."""
        if self._running:
            raise RuntimeError(
                "Tracker is already running. Call stop() before start()."
            )
        self._started_at = datetime.now(timezone.utc)
        self._backend.start()
        self._running = True
        return self

    def stop(self) -> Report:
        """
        Stop tracking and return a Report.

        Returns:
            A Report with duration, energy, CO₂, and backend metadata.

        Raises:
            RuntimeError: If stop() is called before start().
        """
        if not self._running:
            raise RuntimeError("Tracker is not running. Call start() first.")

        stopped_at = datetime.now(timezone.utc)
        self._backend.stop()
        self._running = False

        energy_kwh = self._backend.energy_kwh
        duration = (stopped_at - self._started_at).total_seconds()  # type: ignore[operator]

        # Use backend-provided CO₂ if available, otherwise compute from intensity
        backend_meta = self._backend.metadata
        if "emissions_kg" in backend_meta and backend_meta["emissions_kg"] is not None:
            co2_kg = float(backend_meta["emissions_kg"])
        else:
            co2_kg = energy_kwh * self._co2_intensity / 1000.0

        self._report = Report(
            duration_seconds=duration,
            energy_kwh=energy_kwh,
            co2_kg=co2_kg,
            backend=self._backend.name,
            metadata=backend_meta,
            started_at=self._started_at,
            stopped_at=stopped_at,
        )
        return self._report

    @property
    def report(self) -> Optional[Report]:
        """The last completed report, or None if tracking hasn't finished."""
        return self._report

    @property
    def is_running(self) -> bool:
        """True while tracking is active."""
        return self._running

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Tracker":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        if self._running:
            self.stop()

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    @staticmethod
    def track(
        fn: Optional[Callable] = None,
        *,
        backend: Optional[BaseBackend] = None,
        co2_intensity_g_per_kwh: float = _DEFAULT_CO2_INTENSITY_G_PER_KWH,
        print_report: bool = True,
    ):
        """
        Decorator that automatically tracks a function.

        Usage:
            @Tracker.track
            def train():
                ...

            @Tracker.track(backend=CodeCarbonBackend(), print_report=False)
            def train():
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                t = Tracker(backend=backend, co2_intensity_g_per_kwh=co2_intensity_g_per_kwh)
                t.start()
                try:
                    result = func(*args, **kwargs)
                finally:
                    report = t.stop()
                    if print_report:
                        print(report)
                return result
            return wrapper

        if fn is not None:
            return decorator(fn)
        return decorator

    # ------------------------------------------------------------------
    # Context manager (standalone function form)
    # ------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def measure(
        backend: Optional[BaseBackend] = None,
        co2_intensity_g_per_kwh: float = _DEFAULT_CO2_INTENSITY_G_PER_KWH,
    ) -> Generator["Tracker", None, None]:
        """
        Standalone context manager for one-liner usage.

        Usage:
            with Tracker.measure() as t:
                train_model()
            print(t.report)
        """
        t = Tracker(backend=backend, co2_intensity_g_per_kwh=co2_intensity_g_per_kwh)
        t.start()
        try:
            yield t
        finally:
            t.stop()
