"""Tasks package — task definitions for all curriculum levels."""

from server.tasks.base_task import BaseTask
from server.tasks.task_easy import EasyTask
from server.tasks.task_medium import MediumTask
from server.tasks.task_hard import HardTask

__all__ = ["BaseTask", "EasyTask", "MediumTask", "HardTask"]
