"""
GreenAI — Lightweight carbon footprint tracking for ML workflows.

Usage:
    from greenai import Tracker

    tracker = Tracker()
    tracker.start()
    # ... your ML code ...
    report = tracker.stop()
    print(report)
"""

from greenai.tracker import Tracker
from greenai.report import Report
from greenai.recommender import Recommender

__version__ = "0.1.0"
__all__ = ["Tracker", "Report", "Recommender"]
