# LTX-2 API

FastAPI inference server wrapping Lightricks' **DistilledPipeline** for text/image-to-video generation.

## Quick Start

```bash
uv run python -m ltx_api
```

## Configuration

All settings via environment variables:

| Variable | Description | Default |
|---|---|---|
| `LTX_DISTILLED_CHECKPOINT_PATH` | Path to distilled checkpoint | `/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors` |
| `LTX_SPATIAL_UPSAMPLER_PATH` | Path to spatial upsampler | `/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors` |
| `LTX_GEMMA_ROOT` | Path to Gemma text encoder | `/models/gemma-3-12b-it-qat-q4_0-unquantized` |
| `LTX_QUANTIZATION` | none, fp8-cast, or fp8-scaled-mm | none |
| `LTX_API_HOST` | Listen address | `0.0.0.0` |
| `LTX_API_PORT` | Listen port | `8000` |
| `LTX_OUTPUT_DIR` | Generated video storage | `/tmp/ltx-output` |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | GPU status, queue depth |
| `POST` | `/api/v1/generate` | Submit generation (returns task ID) |
| `GET` | `/api/v1/tasks/{id}` | Poll task status |
| `GET` | `/api/v1/tasks/{id}/download` | Download generated MP4 |
| `DELETE` | `/api/v1/tasks/{id}` | Remove task and output |

See [DEPLOY.md](DEPLOY.md) for full deployment instructions.
