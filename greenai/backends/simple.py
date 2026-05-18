"""
SimpleBackend — default heuristic backend.

Estimates energy from wall-clock time and an assumed average CPU/GPU wattage.
No external dependencies required.

Default assumptions:
  - Average system power draw: 150 W  (laptop-class CPU + GPU)
  - Carbon intensity: 233 g CO₂ / kWh  (European average, IEA 2023)

Both values are configurable via constructor arguments.
"""

import time
from typing import Optional

from greenai.backends.base import BaseBackend


_DEFAULT_WATTAGE = 150.0
_DEFAULT_CO2_INTENSITY_G_PER_KWH = 233.0


class SimpleBackend(BaseBackend):
    """
    Time-based energy estimator.

    Args:
        average_wattage: Assumed average system power draw in Watts.
        co2_intensity_g_per_kwh: Grid carbon intensity in g CO₂/kWh.
    """

    def __init__(
        self,
        average_wattage: float = _DEFAULT_WATTAGE,
        co2_intensity_g_per_kwh: float = _DEFAULT_CO2_INTENSITY_G_PER_KWH,
    ) -> None:
        self._wattage = average_wattage
        self._co2_intensity = co2_intensity_g_per_kwh
        self._start_time: Optional[float] = None
        self._duration: float = 0.0

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._duration = 0.0

    def stop(self) -> None:
        if self._start_time is None:
            raise RuntimeError("SimpleBackend.stop() called before start().")
        self._duration = time.perf_counter() - self._start_time
        self._start_time = None

    @property
    def energy_kwh(self) -> float:
        hours = self._duration / 3600.0
        return self._wattage * hours / 1000.0

    @property
    def name(self) -> str:
        return "simple"

    @property
    def metadata(self):
        return {
            "assumed_wattage_w": self._wattage,
            "co2_intensity_g_per_kwh": self._co2_intensity,
            "duration_seconds": self._duration,
            "note": (
                "Heuristic estimate — install codecarbon for hardware-level measurements."
            ),
        }
