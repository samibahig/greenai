# 🌿 GreenAI

> Lightweight carbon footprint tracking for ML workflows — visible, measurable, actionable.

GreenAI makes the energy cost of machine learning **visible in a few lines of code**. It acts as a unified, simplified layer over existing green-AI tools, giving developers and researchers an instant picture of what their models cost without any complex setup.

---

## Features

- **Minimal API** — `start()` / `stop()` is all you need
- **Unified report** — duration, energy (kWh / Wh), and CO₂ (kg / g) in one object
- **Modular backend system** — default heuristic estimator, or plug in [CodeCarbon](https://github.com/mlco2/codecarbon) for hardware-level measurements
- **Smart recommendations** — FP16, quantization, pruning, carbon-aware scheduling
- **Framework-agnostic** — works with PyTorch, TensorFlow, HuggingFace, or plain Python
- **Zero required dependencies** — the default backend needs nothing extra

---

## Installation

```bash
pip install greenai
```

With CodeCarbon backend support:

```bash
pip install greenai[codecarbon]
```

---

## Quickstart

### Basic usage

```python
from greenai import Tracker

tracker = Tracker()
tracker.start()

# ... your ML code here ...
train_model(data)

report = tracker.stop()
print(report)
# [GreenAI] Duration: 142.30s | Energy: 5.9292 Wh | CO₂: 1.3815 g  (backend: simple)
```

### Context manager

```python
from greenai import Tracker

with Tracker() as tracker:
    train_model(data)

print(tracker.report)
```

### Decorator

```python
from greenai import Tracker

@Tracker.track
def train_model(data):
    ...

train_model(data)  # prints report automatically
```

### One-liner context manager

```python
from greenai import Tracker

with Tracker.measure() as t:
    train_model(data)

print(t.report.to_dict())
```

---

## Report object

```python
report = tracker.stop()

report.duration_seconds  # float — wall-clock seconds
report.energy_kwh        # float — kWh
report.energy_wh         # float — Wh  (shortcut)
report.co2_kg            # float — kg CO₂
report.co2_g             # float — grams CO₂  (shortcut)
report.backend           # str — which backend was used
report.metadata          # dict — extra backend data
report.started_at        # datetime (UTC)
report.stopped_at        # datetime (UTC)

report.summary()         # human-readable one-liner string
report.to_dict()         # serialize to plain dict (JSON-friendly)
```

---

## Backends

### SimpleBackend (default)

Heuristic estimator based on wall-clock time and assumed system wattage. No external dependencies.

```python
from greenai import Tracker
from greenai.backends import SimpleBackend

tracker = Tracker(
    backend=SimpleBackend(
        average_wattage=200,           # Watts — tune to your hardware
        co2_intensity_g_per_kwh=400,   # g CO₂/kWh — tune to your grid
    )
)
```

### CodeCarbonBackend

Real hardware-level measurements via [CodeCarbon](https://github.com/mlco2/codecarbon).

```bash
pip install greenai[codecarbon]
```

```python
from greenai import Tracker
from greenai.backends import CodeCarbonBackend

tracker = Tracker(
    backend=CodeCarbonBackend(country_iso_code="FRA")
)
tracker.start()
train_model(data)
report = tracker.stop()
```

### Custom backend

Implement `BaseBackend` to plug in any measurement tool:

```python
from greenai.backends import BaseBackend

class MyBackend(BaseBackend):
    def start(self): ...
    def stop(self): ...

    @property
    def energy_kwh(self) -> float:
        return ...  # your measurement here

tracker = Tracker(backend=MyBackend())
```

---

## Recommendations

```python
from greenai import Tracker, Recommender

tracker = Tracker()
tracker.start()
train_model(data)
report = tracker.stop()

rec = Recommender(
    model_size_params=175_000_000,  # optional
    carbon_budget_kg=0.01,          # optional — triggers alert if exceeded
    framework="pytorch",            # optional — adds framework-specific hints
)

for suggestion in rec.analyze(report):
    print(suggestion)

# [HIGH] Carbon budget exceeded: Emissions (15.2300 g CO₂) exceed your budget by 52.3%. Apply the suggestions below. (varies savings)
# [HIGH] Use mixed-precision training (FP16 / BF16): ... (30–60% energy savings)
# [HIGH] Apply post-training quantization (INT8 / INT4): ... (50–75% inference energy savings)
# [LOW]  Schedule training during low-carbon hours: ... (10–50% CO₂ savings)
```

---

## Running tests

```bash
pip install greenai[dev]
pytest
```

---

## Architecture

```
greenai/
├── tracker.py        ← Tracker class (start / stop / context manager / decorator)
├── report.py         ← Report dataclass
├── recommender.py    ← Heuristic optimization suggestions
└── backends/
    ├── base.py       ← BaseBackend abstract class
    ├── simple.py     ← Default time-based estimator
    └── codecarbon.py ← CodeCarbon adapter (optional)
```

**Design principles:**
1. **Simple to use** — one import, two method calls
2. **Extensible** — swap or add backends without touching user code
3. **Universal** — works in training loops, notebooks, and production scripts

---

## Roadmap

- Carbon-aware scheduling (pick the greenest training window automatically)
- Model comparator (side-by-side energy cost of multiple model variants)
- Multi-objective optimization (accuracy vs. energy vs. cost)
- CLI tool (`greenai run python train.py`)

---

## License

MIT
