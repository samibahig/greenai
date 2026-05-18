"""GreenAI backend system — plug in the measurement engine you need."""

from greenai.backends.base import BaseBackend
from greenai.backends.simple import SimpleBackend

__all__ = ["BaseBackend", "SimpleBackend"]

try:
    from greenai.backends.codecarbon import CodeCarbonBackend
    __all__.append("CodeCarbonBackend")
except ImportError:
    pass
