"""
GreenAI — the simplest way to track ML carbon footprint.

    from greenai import GreenAI

    g = GreenAI()
    g.start()
    train()
    g.stop()
    g.report()
"""

from greenai.core import GreenAI
from greenai.tracker import Tracker        # kept for backward compatibility
from greenai.report import Report
from greenai.recommender import Recommender

__version__ = "0.2.0"
__all__ = ["GreenAI", "Tracker", "Report", "Recommender"]
