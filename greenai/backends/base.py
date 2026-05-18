"""Abstract base class for GreenAI backends."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseBackend(ABC):
    """
    All GreenAI backends must implement this interface.

    Backends are responsible for measuring (or estimating) energy consumption
    during a tracked run. The Tracker calls start() and stop(), then reads
    the energy_kwh property to build a unified Report.
    """

    @abstractmethod
    def start(self) -> None:
        """Begin measurement."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """End measurement and store results internally."""
        ...

    @property
    @abstractmethod
    def energy_kwh(self) -> float:
        """Return the measured/estimated energy in kWh after stop() is called."""
        ...

    @property
    def name(self) -> str:
        """Human-readable backend identifier."""
        return self.__class__.__name__

    @property
    def metadata(self) -> Dict[str, Any]:
        """Optional extra data to include in the Report (override as needed)."""
        return {}
