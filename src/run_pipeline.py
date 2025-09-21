"""Pipeline runner for retail analytics."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import yaml

from pipeline.task import Task
from utils.io import get_paths, logger

PATHS = get_paths()
SRC_DIR = PATHS.base / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CONFIG_PATH = PATHS.configs / "artifacts.yml"


def _resolve_entrypoint(entrypoint: str) -> Callable[[], None]:
    if ":" not in entrypoint:
        raise ValueError(f"Entrypoint '{entrypoint}' must be 'module:function'")
    module_name, func_name = entrypoint.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func


def _load_config(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return config.get("artifacts", {})


def _build_tasks(artifact_specs: Dict[str, dict]) -> Dict[str, Task]:
    # Map outputs to artifact name for dependency resolution
    output_map: Dict[str, str] = {}
    for name, spec in artifact_specs.items():
        for output in spec.get("outputs", []) or []:
            output_map[output] = name

    tasks: Dict[str, Task] = {}
    for name, spec in artifact_specs.items():
        entrypoint = spec.get("entrypoint")
        if not entrypoint:
            raise ValueError(f"Artifact '{name}' missing 'entrypoint'")
        run_callable = _resolve_entrypoint(entrypoint)
        tasks[name] = Task(
            name=name,
            run=run_callable,
            inputs=spec.get("inputs", []),
            outputs=spec.get("outputs", []),
            requires=[],
        )

    # Attach dependencies
    for name, spec in artifact_specs.items():
        deps: List[Task] = []
        for input_path in spec.get("inputs", []) or []:
            dependency_name = output_map.get(input_path)
            if dependency_name and dependency_name != name:
                deps.append(tasks[dependency_name])
        tasks[name].requires = deps
    return tasks


def run_pipeline(targets: Iterable[str] | None = None) -> None:
    artifact_specs = _load_config(CONFIG_PATH)
    if not artifact_specs:
        raise FileNotFoundError(f"No artifacts defined in {CONFIG_PATH}")

    tasks = _build_tasks(artifact_specs)
    if targets:
        for target in targets:
            if target not in tasks:
                raise KeyError(f"Unknown artifact '{target}'")
            tasks[target].execute()
    else:
        for name in artifact_specs.keys():
            tasks[name].execute()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run retail analytics pipeline")
    parser.add_argument(
        "artifacts",
        nargs="*",
        help="Specific artifacts to build (default: all).",
    )
    args = parser.parse_args(argv)

    logger.info("Starting pipeline (targets=%s)", args.artifacts or "ALL")
    run_pipeline(args.artifacts or None)
    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()
