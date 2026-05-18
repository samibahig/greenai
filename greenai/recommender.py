"""
Recommender — heuristic optimization suggestions based on a Report.

These are intentionally lightweight and interpretable.
They give developers actionable next steps without complex analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from greenai.report import Report
from typing import Dict, Any


@dataclass
class Recommendation:
    """A single optimization suggestion."""

    title: str
    description: str
    estimated_savings: str
    priority: str  # "high" | "medium" | "low"

    def __str__(self) -> str:
        return f"[{self.priority.upper()}] {self.title}: {self.description} ({self.estimated_savings} savings)"


class Recommender:
    """
    Analyses a Report and proposes optimisations to reduce the carbon footprint.

    Args:
        model_size_params: Number of parameters in the model (optional).
        carbon_budget_kg: Maximum acceptable CO₂ in kg (optional).
        framework: ML framework in use — "pytorch", "tensorflow", "huggingface", or None.

    Example:
        report = tracker.stop()
        rec = Recommender(model_size_params=175_000_000, carbon_budget_kg=0.001)
        suggestions = rec.analyze(report)
        for s in suggestions:
            print(s)
    """

    def __init__(
        self,
        model_size_params: Optional[int] = None,
        carbon_budget_kg: Optional[float] = None,
        framework: Optional[str] = None,
    ) -> None:
        self._model_size = model_size_params
        self._budget = carbon_budget_kg
        self._framework = (framework or "").lower()

    def analyze(self, report: Report) -> List[Recommendation]:
        """
        Return a prioritised list of recommendations for the given report.

        Args:
            report: A completed Report from Tracker.stop().

        Returns:
            List of Recommendation objects, ordered by priority.
        """
        suggestions: List[Recommendation] = []

        # --- budget exceeded ---
        if self._budget is not None and report.co2_kg > self._budget:
            excess_pct = (report.co2_kg - self._budget) / self._budget * 100
            suggestions.append(Recommendation(
                title="Carbon budget exceeded",
                description=(
                    f"Emissions ({report.co2_g:.4f} g CO₂) exceed your budget by "
                    f"{excess_pct:.1f}%. Apply the suggestions below."
                ),
                estimated_savings="varies",
                priority="high",
            ))

        # --- long runs ---
        if report.duration_seconds > 3600:
            suggestions.append(Recommendation(
                title="Use mixed-precision training (FP16 / BF16)",
                description=(
                    "Long runs on modern GPUs benefit strongly from FP16/BF16. "
                    "Speeds up training 1.5–3× with minimal accuracy loss."
                    + self._framework_hint("fp16"),
                ),
                estimated_savings="30–60% energy",
                priority="high",
            ))

        # --- large models ---
        if self._model_size is not None and self._model_size > 100_000_000:
            suggestions.append(Recommendation(
                title="Apply post-training quantization (INT8 / INT4)",
                description=(
                    "Models with >100 M parameters can be quantized to INT8/INT4, "
                    "reducing inference cost by 2–4× with negligible accuracy drop."
                    + self._framework_hint("quantization"),
                ),
                estimated_savings="50–75% inference energy",
                priority="high" if self._model_size > 1_000_000_000 else "medium",
            ))

        # --- moderate energy use ---
        if report.energy_kwh > 0.01:
            suggestions.append(Recommendation(
                title="Reduce model size / prune redundant layers",
                description=(
                    "Consider magnitude pruning or structured pruning to cut 20–40% "
                    "of parameters without significant accuracy loss."
                ),
                estimated_savings="20–40% energy",
                priority="medium",
            ))

        # --- short but repeated runs ---
        if report.duration_seconds < 60 and report.energy_kwh > 0.0001:
            suggestions.append(Recommendation(
                title="Batch your workloads",
                description=(
                    "Short, frequent runs have high startup overhead. "
                    "Combine them into fewer, larger batches to improve GPU utilisation."
                ),
                estimated_savings="10–30% energy",
                priority="medium",
            ))

        # --- always suggest green scheduling ---
        suggestions.append(Recommendation(
            title="Schedule training during low-carbon hours",
            description=(
                "Grid carbon intensity varies up to 3–5× throughout the day. "
                "Use tools like Electricity Maps to identify the cleanest window."
            ),
            estimated_savings="10–50% CO₂ (same energy, cleaner grid)",
            priority="low",
        ))

        # Sort: high → medium → low
        order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda r: order.get(r.priority, 99))
        return suggestions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _framework_hint(self, topic: str) -> str:
        hints = {
            ("pytorch", "fp16"): " Use `torch.autocast('cuda')` or `torch.cuda.amp.GradScaler`.",
            ("pytorch", "quantization"): " See `torch.quantization.quantize_dynamic`.",
            ("tensorflow", "fp16"): " Set `tf.keras.mixed_precision.set_global_policy('mixed_float16')`.",
            ("tensorflow", "quantization"): " Use TensorFlow Lite's `TFLiteConverter` with `DEFAULT` optimizations.",
            ("huggingface", "fp16"): " Pass `fp16=True` to `TrainingArguments`.",
            ("huggingface", "quantization"): " Use `bitsandbytes` via `load_in_8bit=True` in `from_pretrained`.",
        }
        return hints.get((self._framework, topic), "")
