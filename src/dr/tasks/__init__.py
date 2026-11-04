"""Task registry. A task owns the environment, evaluation protocol, baselines and task-specific plots."""
from .connect4 import Connect4Task
from .dr import DemandResponseTask

TASKS = {"dr": DemandResponseTask, "connect4": Connect4Task}


def make_task(cfg):
    """Instantiate the task named by `cfg['task']` (default: demand response)."""
    return TASKS[cfg.get("task", "dr")](cfg)
