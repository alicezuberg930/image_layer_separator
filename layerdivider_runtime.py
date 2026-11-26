"""Process-wide SAM 3 cache, outside the source folders watched by Gradio."""

from types import SimpleNamespace
from threading import RLock

sam3_cache = SimpleNamespace(
    model=None, processor=None, checkpoint_path=None, inference_lock=RLock()
)
