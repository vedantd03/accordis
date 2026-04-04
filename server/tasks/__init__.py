"""Tasks package — task definitions for all curriculum levels."""

from accordis.server.tasks.base_task import BaseTask
from accordis.server.tasks.task_easy import EasyTask
from accordis.server.tasks.task_medium import MediumTask
from accordis.server.tasks.task_hard import HardTask

__all__ = ["BaseTask", "EasyTask", "MediumTask", "HardTask"]
