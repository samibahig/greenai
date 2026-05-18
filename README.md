# 🌿 GreenAI

> The simplest way to track ML carbon footprint.

```python
from greenai import GreenAI

g = GreenAI()
g.start()
train()
g.stop()
g.report()
```

```
┌──────────────────────────┐
│  🌿 GreenAI              │
├──────────────────────────┤
│  Duration    2m 34s      │
│  Energy      0.234 Wh    │
│  CO₂         0.054 g     │
└──────────────────────────┘
```

Three metrics. One call. No configuration needed.

---

## Install

```bash
pip install greenai
```

With real hardware measurements via [CodeCarbon](https://github.com/mlco2/codecarbon):

```bash
pip install greenai[codecarbon]
```

---

## Usage

### Basic

```python
from greenai import GreenAI

g = GreenAI()
g.start()
train_model(data)
g.stop()
g.report()          # prints the summary, returns a dict
```

### Context manager

```python
with GreenAI() as g:
    train_model(data)
g.report()
```

### Decorator

```python
from greenai import GreenAI

@GreenAI.track
def train_model(data):
    ...

train_model(data)   # prints report automatically after the call
```

### Jupyter / notebooks

```python
from greenai import GreenAI

with GreenAI.measure() as g:
    train_model(data)

g.report()          # clean output, works inline in any cell
```

---

## What you get

`g.report()` prints a clean 3-line box and returns a dict:

| Key               | Description              |
|-------------------|--------------------------|
| `duration_seconds`| Wall-clock time          |
| `energy_kwh`      | Energy in kWh            |
| `co2_kg`          | CO₂ emitted in kg        |
| `backend`         | Which backend was used   |
| `started_at`      | ISO timestamp (UTC)      |
| `stopped_at`      | ISO timestamp (UTC)      |

Quick access properties (no dict needed):

```python
g.duration    # seconds (float)
g.energy_wh   # Wh (float)
g.co2_g       # grams CO₂ (float)
```

---

## Backends

### Default — SimpleBackend

Works out of the box. Estimates energy from wall-clock time and an assumed system wattage.

```python
from greenai import GreenAI
from greenai.backends import SimpleBackend

g = GreenAI(
    backend=SimpleBackend(
        average_wattage=200,          # tune to your hardware (default: 150 W)
        co2_intensity_g_per_kwh=400,  # tune to your grid   (default: 233 g/kWh)
    )
)
```

### CodeCarbonBackend — real hardware measurements

```bash
pip install greenai[codecarbon]
```

```python
from greenai import GreenAI
from greenai.backends import CodeCarbonBackend

g = GreenAI(backend=CodeCarbonBackend(country_iso_code="FRA"))
g.start()
train_model(data)
g.stop()
g.report()
```

### Custom backend

```python
from greenai.backends import BaseBackend

class MyBackend(BaseBackend):
    def start(self): ...
    def stop(self): ...

    @property
    def energy_kwh(self) -> float:
        return ...  # your measurement

g = GreenAI(backend=MyBackend())
```

---

## Recommendations

```python
from greenai import GreenAI, Recommender

g = GreenAI()
g.start()
train_model(data)
g.stop()
g.report()

rec = Recommender(
    model_size_params=175_000_000,  # optional
    carbon_budget_kg=0.01,          # optional — alerts if exceeded
    framework="pytorch",            # optional — framework-specific hints
)

for tip in rec.analyze_dict(g.report()):
    print(tip)
```

---

## Design principles

1. **Zero friction** — works without reading the docs
2. **Three metrics only** — duration, energy, CO₂. Nothing else by default.
3. **Pluggable** — swap backends without changing your code
4. **Universal** — PyTorch, TensorFlow, HuggingFace, plain Python, notebooks, scripts

---

## Running tests

```bash
pip install pytest
pytest
```

---

## Architecture

```
greenai/
├── core.py          ← GreenAI  (main simple class)
├── tracker.py       ← Tracker  (power-user API, kept for compatibility)
├── report.py        ← Report   (dataclass for Tracker output)
├── recommender.py   ← Recommender (optimization suggestions)
└── backends/
    ├── base.py      ← BaseBackend abstract class
    ├── simple.py    ← Default heuristic estimator
    └── codecarbon.py ← CodeCarbon adapter (optional)
```

---

## Roadmap

- Carbon-aware scheduling (pick the greenest window automatically)
- Model comparator (energy cost of several models side by side)
- CLI: `greenai run python train.py`
- Multi-objective suggestions: accuracy vs. energy vs. cost

---

## License

MIT
