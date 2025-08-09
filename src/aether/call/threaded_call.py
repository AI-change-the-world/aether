import threading
import uuid
from typing import Any, Callable, Optional


class ThreadTask:
    def __init__(self, func: Callable, *args, **kwargs):
        self.task_id = str(uuid.uuid4())
        self.func = func
        self.args = args
        self.kwargs = kwargs

        self._thread = threading.Thread(target=self._run_wrapper)
        self._status = "pending"  # pending, running, done, failed
        self._result: Optional[Any] = None
        self._error: Optional[Exception] = None

    def _run_wrapper(self):
        self._status = "running"
        try:
            self._result = self.func(*self.args, **self.kwargs)
            self._status = "done"
        except Exception as e:
            self._error = e
            self._status = "failed"

    def start(self):
        self._thread.start()

    def join(self, timeout=None):
        self._thread.join(timeout)

    def is_alive(self):
        return self._thread.is_alive()

    @property
    def status(self):
        return self._status

    @property
    def result(self):
        return self._result

    @property
    def error(self):
        return self._error


class ThreadTaskManager:
    def __init__(self):
        self._tasks: dict[str, ThreadTask] = {}

    def submit(self, func: Callable, *args, **kwargs) -> str:
        task = ThreadTask(func, *args, **kwargs)
        self._tasks[task.task_id] = task
        task.start()
        return task.task_id

    def get_status(self, task_id: str) -> Optional[str]:
        task = self._tasks.get(task_id)
        return task.status if task else None

    def get_result(self, task_id: str) -> Optional[Any]:
        task = self._tasks.get(task_id)
        if task and task.status == "done":
            return task.result
        return None

    def get_error(self, task_id: str) -> Optional[Exception]:
        task = self._tasks.get(task_id)
        if task and task.status == "failed":
            return task.error
        return None

    def get_task(self, task_id: str) -> Optional[ThreadTask]:
        return self._tasks.get(task_id)


GLOBAL_THREAD_TASK_MANAGER = ThreadTaskManager()
