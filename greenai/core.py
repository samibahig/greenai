"""
GreenAI — the main class. Simple as it gets.

    g = GreenAI()
    g.start()
    train()
    g.stop()
    g.report()
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable, Dict, Generator, Optional

from greenai.backends.base import BaseBackend
from greenai.backends.simple import SimpleBackend

_DEFAULT_CO2_INTENSITY_G_PER_KWH = 233.0


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _fmt_energy(kwh: float) -> str:
    wh = kwh * 1000
    if wh < 1:
        return f"{wh * 1000:.3f} mWh"
    if wh < 1000:
        return f"{wh:.3f} Wh"
    return f"{kwh:.4f} kWh"


def _fmt_co2(kg: float) -> str:
    g = kg * 1000
    if g < 1:
        return f"{g * 1000:.3f} mg"
    if g < 1000:
        return f"{g:.3f} g"
    return f"{kg:.4f} kg"


class GreenAI:
    """
    Track the carbon cost of any ML run in three lines.

    Basic:
        g = GreenAI()
        g.start()
        train()
        g.stop()
        g.report()          # prints a clean summary

    Context manager:
        with GreenAI() as g:
            train()
        g.report()

    Decorator:
        @GreenAI.track
        def train():
            ...

    Custom backend:
        from greenai.backends import CodeCarbonBackend
        g = GreenAI(backend=CodeCarbonBackend(country_iso_code="FRA"))
    """

    def __init__(
        self,
        backend: Optional[BaseBackend] = None,
        co2_intensity_g_per_kwh: float = _DEFAULT_CO2_INTENSITY_G_PER_KWH,
    ) -> None:
        self._backend: BaseBackend = backend or SimpleBackend()
        self._co2_intensity = co2_intensity_g_per_kwh
        self._started_at: Optional[datetime] = None
        self._stopped_at: Optional[datetime] = None
        self._running = False
        self._result: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def start(self) -> "GreenAI":
        if self._running:
            raise RuntimeError("Already tracking. Call stop() first.")
        self._started_at = datetime.now(timezone.utc)
        self._backend.start()
        self._running = True
        return self

    def stop(self) -> "GreenAI":
        if not self._running:
            raise RuntimeError("Not tracking. Call start() first.")
        self._stopped_at = datetime.now(timezone.utc)
        self._backend.stop()
        self._running = False

        duration = (self._stopped_at - self._started_at).total_seconds()  # type: ignore[operator]
        energy_kwh = self._backend.energy_kwh
        meta = self._backend.metadata

        co2_kg = (
            float(meta["emissions_kg"])
            if meta.get("emissions_kg") is not None
            else energy_kwh * self._co2_intensity / 1000.0
        )

        self._result = {
            "duration_seconds": duration,
            "energy_kwh": energy_kwh,
            "co2_kg": co2_kg,
            "backend": self._backend.name,
            "started_at": self._started_at.isoformat(),
            "stopped_at": self._stopped_at.isoformat(),
        }
        return self

    def report(self) -> Dict:
        """
        Print a clean summary and return the raw data as a dict.

        Example output:
            ┌────────────────────────────────┐
            │  🌿 GreenAI                    │
            ├────────────────────────────────┤
            │  Duration    2m 34s            │
            │  Energy      0.234 Wh          │
            │  CO₂         0.054 g           │
            └────────────────────────────────┘
        """
        if self._result is None:
            raise RuntimeError("No data yet — call start() then stop() first.")

        r = self._result
        lines = [
            ("Duration", _fmt_duration(r["duration_seconds"])),
            ("Energy",   _fmt_energy(r["energy_kwh"])),
            ("CO₂",      _fmt_co2(r["co2_kg"])),
        ]

        label_w = max(len(l) for l, _ in lines)
        value_w = max(len(v) for _, v in lines)
        inner_w = label_w + value_w + 4          # "  Label    Value  "
        inner_w = max(inner_w, len("🌿 GreenAI") + 2)

        bar   = "─" * (inner_w + 2)
        title = "🌿 GreenAI"
        padding = inner_w - len(title)

        print(f"┌{bar}┐")
        print(f"│  {title}{' ' * padding}  │")
        print(f"├{bar}┤")
        for label, value in lines:
            gap = inner_w - label_w - len(value) - 2
            print(f"│  {label:<{label_w}}  {value}{' ' * gap}  │")
        print(f"└{bar}┘")

        return dict(r)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def duration(self) -> Optional[float]:
        return self._result["duration_seconds"] if self._result else None

    @property
    def energy_wh(self) -> Optional[float]:
        return self._result["energy_kwh"] * 1000 if self._result else None

    @property
    def co2_g(self) -> Optional[float]:
        return self._result["co2_kg"] * 1000 if self._result else None

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "GreenAI":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        if self._running:
            self.stop()

    def __repr__(self) -> str:
        if self._result:
            r = self._result
            return (
                f"GreenAI(duration={_fmt_duration(r['duration_seconds'])}, "
                f"energy={_fmt_energy(r['energy_kwh'])}, "
                f"co2={_fmt_co2(r['co2_kg'])})"
            )
        return "GreenAI(not started)" if not self._running else "GreenAI(running...)"

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    @staticmethod
    def track(
        fn: Optional[Callable] = None,
        *,
        backend: Optional[BaseBackend] = None,
        print_report: bool = True,
    ):
        """
        Decorator that tracks a function automatically.

            @GreenAI.track
            def train():
                ...

            @GreenAI.track(backend=CodeCarbonBackend(), print_report=False)
            def train():
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                g = GreenAI(backend=backend)
                g.start()
                try:
                    result = func(*args, **kwargs)
                finally:
                    g.stop()
                    if print_report:
                        g.report()
                return result
            return wrapper

        return decorator(fn) if fn is not None else decorator

    # ------------------------------------------------------------------
    # One-liner context manager
    # ------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def measure(backend: Optional[BaseBackend] = None) -> Generator["GreenAI", None, None]:
        """
        One-liner context manager.

            with GreenAI.measure() as g:
                train()
            g.report()
        """
        g = GreenAI(backend=backend)
        g.start()
        try:
            yield g
        finally:
            g.stop()
