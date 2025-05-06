from __future__ import annotations
import os

try:
    from clearml import Task
except ModuleNotFoundError:
    Task = None

_TASK = None


def _is_disabled(cfg) -> bool:
    if os.getenv("CLEARML_DISABLED", "0") == "1":
        return True
    return not cfg.clearml.get("enabled", False)


def init_clearml(cfg):
    global _TASK

    if os.getenv("CLEARML_DISABLED", "0") == "1":
        print("CLEARML_DISABLED=1 – ClearML disabled.")
        return None

    if not cfg.clearml.get("enabled", False):
        return None

    if Task is None:
        print("clearml package not found – skipping ClearML logging.")
        return None

    if _TASK is None:
        _TASK = Task.init(
            project_name=cfg.clearml.project_name,
            task_name=cfg.clearml.task_name,
            output_uri=cfg.clearml.get("output_uri", None),
        )
        _TASK.connect(cfg, name="hydra_cfg")
    return _TASK


def get_logger():
    if _TASK is None:
        raise RuntimeError("ClearML task not initialised (or disabled).")
    return _TASK.get_logger()
