import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ltx_api.config import settings
from ltx_api.pipeline import PipelineManager
from ltx_api.schemas import (
    ErrorResponse,
    GenerateRequest,
    HealthResponse,
    TaskResponse,
    TaskStatus,
)
from ltx_api.tasks import TaskManager

logger = logging.getLogger(__name__)

pipeline_manager = PipelineManager(settings)
task_manager = TaskManager(pipeline_manager, settings)

app = FastAPI(
    title="LTX-2 Distilled Inference API",
    description="FastAPI wrapper around Lightricks' LTX-2 DistilledPipeline for text/image-to-video generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Loading pipeline...")
    pipeline_manager.load()
    task_manager.start()
    logger.info("API ready on %s:%s", settings.host, settings.port)


@app.on_event("shutdown")
def on_shutdown() -> None:
    logger.info("Shutting down...")
    task_manager.stop()
    pipeline_manager.unload()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail=str(exc), code="INTERNAL_ERROR").model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        queue_length=task_manager.get_queue_length(),
        active_task=task_manager.get_active_task_id(),
    )


@app.post(
    "/api/v1/generate",
    response_model=TaskResponse,
    status_code=202,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def generate(request: GenerateRequest) -> TaskResponse:
    if pipeline_manager.device_str != "cuda":
        raise HTTPException(
            status_code=503,
            detail="No CUDA GPU available. The pipeline requires a CUDA-capable GPU.",
        )
    task_id = task_manager.submit(request)
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=500, detail="Failed to create task")
    return task


@app.get(
    "/api/v1/tasks/{task_id}",
    response_model=TaskResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_task(task_id: str) -> TaskResponse:
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found",
        )
    return task


@app.get(
    "/api/v1/tasks/{task_id}/download",
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
def download_task(task_id: str) -> FileResponse:
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task.status == TaskStatus.running or task.status == TaskStatus.queued:
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} is {task.status.value}. Wait for completion before downloading.",
        )
    if task.status == TaskStatus.failed:
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} failed and has no output. Error: {task.error}",
        )
    if task.status == TaskStatus.completed and task.output_filename:
        output_path = Path(settings.output_dir) / task.output_filename
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Output file not found on disk")
        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=f"ltx-generation-{task_id}.mp4",
        )
    raise HTTPException(status_code=500, detail="Unexpected task state")


@app.delete(
    "/api/v1/tasks/{task_id}",
    responses={404: {"model": ErrorResponse}},
)
def delete_task(task_id: str) -> dict:
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task.output_filename:
        output_path = Path(settings.output_dir) / task.output_filename
        if output_path.exists():
            output_path.unlink()
    return {"status": "deleted", "task_id": task_id}
