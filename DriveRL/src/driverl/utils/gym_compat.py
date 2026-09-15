"""Small compatibility layer for Gymnasium spaces.

DriveRL uses ``gymnasium``. Some nuPlan-ready environments on this
machine do not have it installed, but eval-only model construction only needs
``Box``/``Discrete`` shape and bounds metadata. Prefer the real implementation
when available and fall back to these minimal classes otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - exercised when gymnasium is installed.
    from gymnasium.spaces import Box, Discrete, Space
except ImportError:  # pragma: no cover - fallback is environment dependent.

    class Space:
        """Minimal base class matching the type checks used by DriveRL agents."""

        shape: tuple[int, ...] | None = None

        def sample(self) -> Any:
            raise NotImplementedError

    class Box(Space):
        """Minimal continuous space with ``low``, ``high``, ``shape`` and dtype."""

        def __init__(self, low: Any, high: Any, dtype: Any = np.float32) -> None:
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            self.dtype = dtype
            self.shape = self.low.shape

        def sample(self) -> np.ndarray:
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class Discrete(Space):
        """Minimal discrete space with ``n``."""

        def __init__(self, n: int) -> None:
            self.n = int(n)
            self.shape = ()

        def sample(self) -> int:
            return int(np.random.randint(self.n))


__all__ = ["Box", "Discrete", "Space"]
