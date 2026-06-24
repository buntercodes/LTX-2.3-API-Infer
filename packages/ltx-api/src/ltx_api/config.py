import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from ltx_core.quantization import QuantizationPolicy

_env_path = Path(".env")
if not _env_path.exists():
    _env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(_env_path)


@dataclass(frozen=True)
class Settings:
    distilled_checkpoint_path: str = field(
        default_factory=lambda: os.environ.get(
            "LTX_DISTILLED_CHECKPOINT_PATH",
            "/models/ltx-2.3-22b-distilled.safetensors",
        )
    )
    spatial_upsampler_path: str = field(
        default_factory=lambda: os.environ.get(
            "LTX_SPATIAL_UPSAMPLER_PATH",
            "/models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        )
    )
    gemma_root: str = field(
        default_factory=lambda: os.environ.get(
            "LTX_GEMMA_ROOT",
            "/models/gemma-3-12b-it-qat-q4_0-unquantized",
        )
    )

    lora_paths: list[str] = field(default_factory=lambda: _parse_list_env("LTX_LORA_PATHS", []))
    lora_strengths: list[float] = field(default_factory=lambda: _parse_float_list_env("LTX_LORA_STRENGTHS", []))

    quantization: str | None = field(
        default_factory=lambda: os.environ.get("LTX_QUANTIZATION", None)
    )

    host: str = field(default_factory=lambda: os.environ.get("LTX_API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("LTX_API_PORT", "8000")))

    output_dir: str = field(
        default_factory=lambda: os.environ.get("LTX_OUTPUT_DIR", "/tmp/ltx-output")
    )
    max_output_age_hours: int = field(
        default_factory=lambda: int(os.environ.get("LTX_MAX_OUTPUT_AGE_HOURS", "24"))
    )
    max_concurrent_tasks: int = field(
        default_factory=lambda: int(os.environ.get("LTX_MAX_CONCURRENT_TASKS", "1"))
    )

    def build_quantization_policy(self) -> QuantizationPolicy | None:
        if self.quantization is None:
            return None
        q = self.quantization.strip().lower()
        if q == "fp8-cast":
            return QuantizationPolicy.fp8_cast()
        if q == "fp8-scaled-mm":
            return QuantizationPolicy.fp8_scaled_mm(None)
        return None


def _parse_list_env(key: str, default: list[str]) -> list[str]:
    val = os.environ.get(key, "")
    return [x.strip() for x in val.split(",") if x.strip()] if val else default


def _parse_float_list_env(key: str, default: list[float]) -> list[float]:
    val = os.environ.get(key, "")
    return [float(x.strip()) for x in val.split(",") if x.strip()] if val else default


settings = Settings()
