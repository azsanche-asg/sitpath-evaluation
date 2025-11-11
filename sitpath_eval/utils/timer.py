from __future__ import annotations

import time
from contextlib import ContextDecorator
from typing import Callable, Optional


class Timer(ContextDecorator):
    def __init__(self, name: str | None = None, logger: Optional[Callable[[str], None]] = None) -> None:
        self.name = name
        self.logger = logger
        self.start: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        if self.start is None:
            return False
        self.elapsed = time.perf_counter() - self.start
        self._log()
        return False

    def _log(self) -> None:
        if self.elapsed is None or self.name is None:
            return
        message = f"{self.name} took {self.elapsed:.3f}s"
        if self.logger:
            if hasattr(self.logger, "info"):
                self.logger.info(message)  # type: ignore[attr-defined]
            else:
                self.logger(message)
        else:
            print(message)

    def stop(self) -> float:
        if self.start is None:
            raise RuntimeError("Timer was never started")
        self.elapsed = time.perf_counter() - self.start
        self._log()
        return self.elapsed


__all__ = ["Timer"]
