"""GreenAI integrations — drop-in callbacks for popular ML frameworks."""

__all__ = []

try:
    from greenai.integrations.huggingface import GreenAICallback
    __all__.append("GreenAICallback")
except ImportError:
    pass
