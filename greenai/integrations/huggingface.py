"""
GreenAICallback — drop-in HuggingFace Trainer integration.

Usage:
    from transformers import Trainer, TrainingArguments
    from greenai.integrations import GreenAICallback

    trainer = Trainer(
        model=model,
        args=TrainingArguments(...),
        callbacks=[GreenAICallback()],
    )
    trainer.train()
    # GreenAI report prints automatically when training ends.

Requires:
    pip install transformers
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from greenai.core import GreenAI
from greenai.backends.base import BaseBackend

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


class GreenAICallback:
    """
    HuggingFace Trainer callback that tracks the carbon cost of a training run.

    Args:
        backend: Optional GreenAI backend (defaults to SimpleBackend).
        print_report: Whether to print the report at the end of training.
        print_epoch_report: Whether to print a mini-report after each epoch.

    Example:
        from greenai.integrations import GreenAICallback

        trainer = Trainer(
            model=model,
            args=args,
            callbacks=[GreenAICallback()],
        )
        trainer.train()

        # Access the report programmatically after training:
        cb = GreenAICallback()
        trainer = Trainer(..., callbacks=[cb])
        trainer.train()
        print(cb.tracker.co2_g, "g CO₂")
    """

    def __init__(
        self,
        backend: Optional[BaseBackend] = None,
        print_report: bool = True,
        print_epoch_report: bool = False,
    ) -> None:
        try:
            from transformers import TrainerCallback  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "transformers is not installed. "
                "Run `pip install transformers` to use GreenAICallback."
            ) from e

        self.tracker = GreenAI(backend=backend)
        self._print_report = print_report
        self._print_epoch_report = print_epoch_report
        self._epoch_tracker: Optional[GreenAI] = None

    # ------------------------------------------------------------------
    # TrainerCallback interface (duck-typed — no hard import needed)
    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        """Start tracking when training begins."""
        self.tracker.start()

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Optionally track each epoch individually."""
        if self._print_epoch_report:
            self._epoch_tracker = GreenAI(backend=None)
            self._epoch_tracker.start()

    def on_epoch_end(self, args, state, control, **kwargs):
        """Print per-epoch report if enabled."""
        if self._print_epoch_report and self._epoch_tracker is not None:
            self._epoch_tracker.stop()
            epoch = int(state.epoch) if state else "?"
            print(f"\n  Epoch {epoch} footprint:")
            self._epoch_tracker.report()
            self._epoch_tracker = None

    def on_train_end(self, args, state, control, **kwargs):
        """Stop tracking and print the full training report."""
        if self.tracker.is_running:
            self.tracker.stop()
        if self._print_report:
            self.tracker.report()

    def on_init_end(self, args, state, control, **kwargs):
        pass

    def on_evaluate(self, args, state, control, **kwargs):
        pass

    def on_save(self, args, state, control, **kwargs):
        pass

    def on_log(self, args, state, control, **kwargs):
        pass
