import logging
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from ltx_api.config import Settings
from ltx_api.pipeline import PipelineManager
from ltx_api.schemas import GenerateRequest, TaskResponse, TaskStatus

logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self, pipeline: PipelineManager, settings: Settings) -> None:
        self._pipeline = pipeline
        self._settings = settings
        self._lock = threading.Lock()
        self._tasks: OrderedDict[str, TaskResponse] = OrderedDict()
        self._queue: list[str] = []
        self._active_task_id: str | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._output_dir = Path(settings.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="task-worker")
        self._worker_thread.start()
        logger.info("Task worker started")

    def stop(self) -> None:
        self._stop_event.set()

    def submit(self, request: GenerateRequest) -> str:
        task_id = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc)
        task = TaskResponse(
            task_id=task_id,
            status=TaskStatus.queued,
            created_at=now,
            request=request,
        )
        with self._lock:
            self._tasks[task_id] = task
            self._queue.append(task_id)
        logger.info("Task %s submitted: prompt=%.60s", task_id, request.prompt)
        return task_id

    def get_task(self, task_id: str) -> TaskResponse | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_active_task_id(self) -> str | None:
        with self._lock:
            return self._active_task_id

    def get_queue_length(self) -> int:
        with self._lock:
            return len(self._queue)

    def _update_status(self, task_id: str, status: TaskStatus, **kwargs: object) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            update_kwargs: dict = {"status": status}
            if status == TaskStatus.completed:
                update_kwargs["completed_at"] = datetime.now(timezone.utc)
            if status == TaskStatus.failed:
                update_kwargs["completed_at"] = datetime.now(timezone.utc)
            update_kwargs.update(kwargs)
            self._tasks[task_id] = task.model_copy(update=update_kwargs)

    def _cleanup_old_outputs(self) -> None:
        max_age = self._settings.max_output_age_hours * 3600
        now = time.time()
        for f in self._output_dir.iterdir():
            if f.is_file() and f.suffix == ".mp4":
                age = now - f.stat().st_mtime
                if age > max_age:
                    try:
                        f.unlink()
                        logger.info("Cleaned up old output: %s", f.name)
                    except OSError:
                        pass

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            task_id: str | None = None
            with self._lock:
                if self._queue:
                    task_id = self._queue.pop(0)
                    self._active_task_id = task_id

            if task_id is None:
                self._cleanup_old_outputs()
                time.sleep(1)
                continue

            task = self.get_task(task_id)
            if task is None:
                with self._lock:
                    self._active_task_id = None
                continue

            try:
                self._update_status(task_id, TaskStatus.running, progress=0.0)
                logger.info("Starting task %s", task_id)

                output_filename = f"{task_id}.mp4"
                output_path = str(self._output_dir / output_filename)

                self._pipeline.generate(
                    request=task.request,
                    output_path=output_path,
                    progress_callback=lambda p: self._update_status(task_id, TaskStatus.running, progress=p),
                )

                self._update_status(
                    task_id,
                    TaskStatus.completed,
                    progress=1.0,
                    output_filename=output_filename,
                )
                logger.info("Task %s completed: %s", task_id, output_filename)

            except Exception as e:
                logger.exception("Task %s failed", task_id)
                self._update_status(task_id, TaskStatus.failed, error=str(e))

            finally:
                with self._lock:
                    self._active_task_id = None
