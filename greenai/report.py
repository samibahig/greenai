"""Standardized report returned after tracking completes."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class Report:
    """
    Unified carbon report produced after a tracked computation.

    Attributes:
        duration_seconds: Wall-clock time of the tracked run.
        energy_kwh: Estimated energy consumption in kWh.
        co2_kg: Estimated CO₂ emissions in kilograms.
        backend: Name of the backend that produced the measurements.
        metadata: Optional extra data provided by the backend.
        started_at: UTC timestamp when tracking started.
        stopped_at: UTC timestamp when tracking stopped.
    """

    duration_seconds: float
    energy_kwh: float
    co2_kg: float
    backend: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    @property
    def energy_wh(self) -> float:
        """Energy in Wh (more readable for short runs)."""
        return self.energy_kwh * 1000

    @property
    def co2_g(self) -> float:
        """CO₂ in grams (more readable for short runs)."""
        return self.co2_kg * 1000

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        return (
            f"[GreenAI] Duration: {self.duration_seconds:.2f}s | "
            f"Energy: {self.energy_wh:.4f} Wh | "
            f"CO₂: {self.co2_g:.4f} g  (backend: {self.backend})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the report to a plain dictionary."""
        return {
            "duration_seconds": self.duration_seconds,
            "energy_kwh": self.energy_kwh,
            "co2_kg": self.co2_kg,
            "energy_wh": self.energy_wh,
            "co2_g": self.co2_g,
            "backend": self.backend,
            "metadata": self.metadata,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
        }

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"Report(duration={self.duration_seconds:.2f}s, "
            f"energy={self.energy_wh:.4f}Wh, "
            f"co2={self.co2_g:.4f}g, "
            f"backend='{self.backend}')"
        )
