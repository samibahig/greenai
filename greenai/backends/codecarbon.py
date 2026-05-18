"""
CodeCarbonBackend — wraps codecarbon for real hardware-level measurements.

Install the optional dependency to use this backend:
    pip install greenai[codecarbon]
"""

from typing import Any, Dict

from greenai.backends.base import BaseBackend


class CodeCarbonBackend(BaseBackend):
    """
    Backend that delegates to the codecarbon library for accurate measurements.

    Requires `pip install codecarbon`.

    Args:
        **codecarbon_kwargs: Passed directly to EmissionsTracker(). Useful
            options include `country_iso_code`, `project_name`, `log_level`.
    """

    def __init__(self, **codecarbon_kwargs: Any) -> None:
        try:
            from codecarbon import EmissionsTracker  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "codecarbon is not installed. "
                "Run `pip install greenai[codecarbon]` to enable this backend."
            ) from exc

        self._tracker = EmissionsTracker(
            save_to_file=False,
            log_level="error",
            **codecarbon_kwargs,
        )
        self._emissions_data: Any = None

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._tracker.start()

    def stop(self) -> None:
        self._emissions_data = self._tracker.stop()

    @property
    def energy_kwh(self) -> float:
        if self._emissions_data is None:
            return 0.0
        return float(getattr(self._emissions_data, "energy_consumed", 0.0))

    @property
    def name(self) -> str:
        return "codecarbon"

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._emissions_data is None:
            return {}
        data = self._emissions_data
        return {
            "emissions_kg": getattr(data, "emissions", None),
            "cpu_power_w": getattr(data, "cpu_power", None),
            "gpu_power_w": getattr(data, "gpu_power", None),
            "ram_power_w": getattr(data, "ram_power", None),
            "country_iso_code": getattr(data, "country_iso_code", None),
            "cloud_provider": getattr(data, "cloud_provider", None),
        }
