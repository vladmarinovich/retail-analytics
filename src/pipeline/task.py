"""Lightweight task orchestration utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence

from utils.io import logger


@dataclass
class Task:
    name: str
    run: Callable[[], None]
    inputs: Sequence[str] | None = None
    outputs: Sequence[str] | None = None
    requires: Iterable["Task"] | None = None
    _has_run: bool = field(default=False, init=False, repr=False)

    def execute(self, force: bool = False) -> None:
        if self._has_run and not force:
            logger.debug("Skipping task %s (already completed)", self.name)
            return
        for dependency in self.requires or []:
            dependency.execute(force=force)
        logger.info("Running task: %s", self.name)
        self.run()
        self._has_run = True

    def __call__(self) -> None:
        self.execute()


__all__ = ["Task"]
