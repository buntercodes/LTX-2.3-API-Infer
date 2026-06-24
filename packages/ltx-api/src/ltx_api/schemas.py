from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class ImageConditioningInput(BaseModel):
    path: str = Field(..., description="Path or URL to the conditioning image")
    frame_idx: int = Field(0, ge=0, description="Target frame index for conditioning")
    strength: float = Field(1.0, ge=0.0, le=2.0, description="Conditioning strength")
    crf: int = Field(33, ge=0, le=51, description="H.264 compression quality (0=lossless)")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for video generation")
    seed: int = Field(42, description="Random seed for reproducibility")
    height: int = Field(512, description="Output video height (must be divisible by 64)")
    width: int = Field(768, description="Output video width (must be divisible by 64)")
    num_frames: int = Field(121, description="Number of frames (must satisfy frames % 8 == 1)")
    frame_rate: float = Field(24.0, description="Frame rate of the output video")
    images: list[ImageConditioningInput] = Field(default_factory=list, description="Optional image conditioning inputs")
    enhance_prompt: bool = Field(False, description="Automatically enhance the prompt via Gemma")

    @field_validator("height", "width")
    @classmethod
    def validate_divisible_by_64(cls, v: int) -> int:
        if v % 64 != 0:
            msg = f"Value must be divisible by 64, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("num_frames")
    @classmethod
    def validate_frames(cls, v: int) -> int:
        if v < 1 or (v - 1) % 8 != 0:
            msg = f"num_frames must satisfy (frames - 1) % 8 == 0, got {v}"
            raise ValueError(msg)
        return v


class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current status of the task")
    created_at: datetime = Field(..., description="Timestamp when the task was created")
    completed_at: datetime | None = Field(None, description="Timestamp when the task completed")
    progress: float | None = Field(None, description="Progress estimate 0.0-1.0")
    error: str | None = Field(None, description="Error message if task failed")
    output_filename: str | None = Field(None, description="Output video filename when completed")
    request: GenerateRequest = Field(..., description="Original generation request")


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Service health status")
    gpu_available: bool = Field(..., description="Whether a CUDA GPU is available")
    gpu_name: str | None = Field(None, description="GPU device name")
    queue_length: int = Field(0, description="Number of pending tasks in the queue")
    active_task: str | None = Field(None, description="ID of the currently running task")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error description")
    code: str | None = Field(None, description="Machine-readable error code")
